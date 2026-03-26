// cube-lambertian.cpp
// Build a Lambertian (SH9) and GGX (Prefiltered Specular) cubemap.
// Restored terminal logging while keeping MSVC /W4 /WX compatibility.

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "external/tinyobjloader/stb_image.h"
#include "external/tinyobjloader/stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct float3 { float x, y, z; };
static inline float3 add(float3 a, float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline float3 mul(float3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float3 cross(float3 a, float3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static inline float3 normalize(float3 v) {
	float d = std::sqrt(std::max(0.0f, dot(v, v)));
	if (d <= 0.0f) return { 0,0,1 };
	float inv = 1.0f / d;
	return { v.x * inv, v.y * inv, v.z * inv };
}

static inline void rgbe_to_float4(uint8_t R, uint8_t G, uint8_t B, uint8_t E, float out[4]) {
	if (E == 0) { out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f; return; }
	float scale = (float)std::ldexp(1.0f, (int)E - 128) / 256.0f;
	out[0] = (float(R) + 0.5f) * scale;
	out[1] = (float(G) + 0.5f) * scale;
	out[2] = (float(B) + 0.5f) * scale;
	out[3] = 1.0f;
}

static inline void float3_to_rgbe(float3 c, uint8_t out[4]) {
	c.x = std::max(0.0f, c.x); c.y = std::max(0.0f, c.y); c.z = std::max(0.0f, c.z);
	float m = std::max(c.x, std::max(c.y, c.z));
	if (m < 1e-32f) { out[0] = out[1] = out[2] = out[3] = 0; return; }
	int e = 0;
	float f = std::frexp(m, &e);
	float v = (f * 256.0f) / m;
	int R = int(c.x * v + 0.5f); int G = int(c.y * v + 0.5f); int B = int(c.z * v + 0.5f);
	out[0] = uint8_t(std::max(0, std::min(255, R)));
	out[1] = uint8_t(std::max(0, std::min(255, G)));
	out[2] = uint8_t(std::max(0, std::min(255, B)));
	out[3] = uint8_t(e + 128);
}

static bool split_cube_faces_vertical_strip_rgbe(uint8_t const* pixels, int w, int h, std::array<std::vector<float>, 6>& faces_out, int& faceSizeOut) {
	if (h != 6 * w) return false;
	int s = w; faceSizeOut = s;
	for (auto& f : faces_out) f.assign(size_t(s) * size_t(s) * 4, 0.0f);
	for (int face = 0; face < 6; ++face) {
		auto& dst = faces_out[face]; int y0 = face * s;
		for (int y = 0; y < s; ++y) {
			for (int x = 0; x < s; ++x) {
				uint8_t const* p = pixels + 4 * ((y0 + y) * w + x);
				float rgba[4]; rgbe_to_float4(p[0], p[1], p[2], p[3], rgba);
				size_t o = 4 * (size_t(y) * size_t(s) + size_t(x));
				dst[o + 0] = rgba[0]; dst[o + 1] = rgba[1]; dst[o + 2] = rgba[2]; dst[o + 3] = rgba[3];
			}
		}
	}
	return true;
}

static void pack_cube_faces_vertical_strip_rgbe_png(std::array<std::vector<float>, 6> const& faces_f, int s, std::string const& out_path) {
	int w = s; int h = 6 * s;
	std::vector<uint8_t> out(size_t(w) * size_t(h) * 4, 0);
	for (int face = 0; face < 6; ++face) {
		int y0 = face * s; auto const& src = faces_f[face];
		for (int y = 0; y < s; ++y) {
			for (int x = 0; x < s; ++x) {
				size_t si = 4 * (size_t(y) * size_t(s) + size_t(x));
				uint8_t rgbe[4]; float3_to_rgbe({ src[si + 0], src[si + 1], src[si + 2] }, rgbe);
				size_t di = 4 * (size_t(y0 + y) * size_t(w) + size_t(x));
				out[di + 0] = rgbe[0]; out[di + 1] = rgbe[1]; out[di + 2] = rgbe[2]; out[di + 3] = rgbe[3];
			}
		}
	}
	stbi_write_png(out_path.c_str(), w, h, 4, out.data(), w * 4);
}

static float3 dir_from_face_uv(int face, float u, float v) {
	switch (face) {
	case 0: return normalize({ 1.0f, -v, -u }); case 1: return normalize({ -1.0f, -v,  u });
	case 2: return normalize({ u,  1.0f,  v }); case 3: return normalize({ u, -1.0f, -v });
	case 4: return normalize({ u, -v,  1.0f }); case 5: return normalize({ -u, -v, -1.0f });
	default: return { 0,0,1 };
	}
}

static void face_uv_from_dir(float3 d, int& face, float& u, float& v) {
	float ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
	if (ax >= ay && ax >= az) { if (d.x > 0) { face = 0; u = -d.z / ax; v = -d.y / ax; } else { face = 1; u = d.z / ax; v = -d.y / ax; } }
	else if (ay >= ax && ay >= az) { if (d.y > 0) { face = 2; u = d.x / ay; v = d.z / ay; } else { face = 3; u = d.x / ay; v = -d.z / ay; } }
	else { if (d.z > 0) { face = 4; u = d.x / az; v = -d.y / az; } else { face = 5; u = -d.x / az; v = -d.y / az; } }
	u = std::max(-1.0f, std::min(1.0f, u)); v = std::max(-1.0f, std::min(1.0f, v));
}

static float3 sample_cubemap_linear(std::array<std::vector<float>, 6> const& faces, int s, float3 dir) {
	dir = normalize(dir); int face = 0; float u = 0.0f, v = 0.0f; face_uv_from_dir(dir, face, u, v);
	float fx = (u * 0.5f + 0.5f) * (float)s - 0.5f; float fy = (v * 0.5f + 0.5f) * (float)s - 0.5f;
	int x0 = std::max(0, std::min(s - 1, (int)std::floor(fx))); int y0 = std::max(0, std::min(s - 1, (int)std::floor(fy)));
	int x1 = std::max(0, std::min(s - 1, x0 + 1)); int y1 = std::max(0, std::min(s - 1, y0 + 1));
	float tx = fx - (float)x0; float ty = fy - (float)y0;
	auto get_p = [&](int x, int y) -> float3 { size_t i = 4 * (size_t(y) * size_t(s) + size_t(x)); return { faces[size_t(face)][i + 0], faces[size_t(face)][i + 1], faces[size_t(face)][i + 2] }; };
	float3 top = add(mul(get_p(x0, y0), 1.0f - tx), mul(get_p(x1, y0), tx));
	float3 bot = add(mul(get_p(x0, y1), 1.0f - tx), mul(get_p(x1, y1), tx));
	return add(mul(top, 1.0f - ty), mul(bot, ty));
}

static float3 importanceSampleGGX(float u1, float u2, float roughness, float3 N) {
	float a = roughness * roughness;
	float phi = 2.0f * (float)M_PI * u1;
	float cosTheta = std::sqrt((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
	float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
	float3 Ht{ std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta };
	float3 up = (std::fabs(N.z) < 0.999f) ? float3{ 0,0,1 } : float3{ 1,0,0 };
	float3 T = normalize(cross(up, N)); float3 B = cross(N, T);
	return normalize(add(add(mul(T, Ht.x), mul(B, Ht.y)), mul(N, Ht.z)));
}

static void hammersley(uint32_t i, uint32_t N, float& u, float& v) {
	u = (float)i / (float)N;
	uint32_t bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	v = (float)bits * 2.3283064365386963e-10f;
}

static inline void sh9_basis(float3 d, float out[9]) {
	float x = d.x, y = d.y, z = d.z;
	out[0] = 0.282095f; out[1] = 0.488603f * y; out[2] = 0.488603f * z; out[3] = 0.488603f * x;
	out[4] = 1.092548f * x * y; out[5] = 1.092548f * y * z; out[6] = 0.315392f * (3.0f * z * z - 1.0f);
	out[7] = 1.092548f * x * z; out[8] = 0.546274f * (x * x - y * y);
}

struct SH9_RGB { std::array<float3, 9> coeff{}; };

static SH9_RGB project_env_to_sh9(std::array<std::vector<float>, 6> const& in_faces, int inSize) {
	SH9_RGB sh{}; float du = 2.0f / (float)inSize; float dv = 2.0f / (float)inSize;
	for (int face = 0; face < 6; ++face) {
		for (int y = 0; y < inSize; ++y) {
			for (int x = 0; x < inSize; ++x) {
				float3 dir = dir_from_face_uv(face, ((x + 0.5f) / (float)inSize) * 2.0f - 1.0f, ((y + 0.5f) / (float)inSize) * 2.0f - 1.0f);
				float basis[9]; sh9_basis(dir, basis);
				float dOmega = (du * dv) / std::pow(1.0f + std::pow(((x + 0.5f) / (float)inSize) * 2.0f - 1.0f, 2.0f) + std::pow(((y + 0.5f) / (float)inSize) * 2.0f - 1.0f, 2.0f), 1.5f);
				size_t i = 4 * (size_t(y) * inSize + x);
				for (int k = 0; k < 9; ++k) {
					float w = basis[k] * dOmega;
					sh.coeff[k].x += in_faces[face][i + 0] * w; sh.coeff[k].y += in_faces[face][i + 1] * w; sh.coeff[k].z += in_faces[face][i + 2] * w;
				}
			}
		}
	}
	return sh;
}

static SH9_RGB convolve_sh9_lambertian(SH9_RGB sh) {
	sh.coeff[0] = mul(sh.coeff[0], (float)M_PI);
	for (int i = 1; i <= 3; ++i) sh.coeff[i] = mul(sh.coeff[i], 2.0f * (float)M_PI / 3.0f);
	for (int i = 4; i <= 8; ++i) sh.coeff[i] = mul(sh.coeff[i], (float)M_PI / 4.0f);
	return sh;
}

static float3 eval_sh9(SH9_RGB const& sh, float3 dir) {
	float b[9]; sh9_basis(dir, b); float3 out{ 0,0,0 };
	for (int i = 0; i < 9; ++i) { out.x += sh.coeff[i].x * b[i]; out.y += sh.coeff[i].y * b[i]; out.z += sh.coeff[i].z * b[i]; }
	return { std::max(0.0f, out.x), std::max(0.0f, out.y), std::max(0.0f, out.z) };
}

static std::array<std::vector<float>, 6> build_lambertian_cube_sh9(std::array<std::vector<float>, 6> const& in_faces, int inSize, int outSize) {
	SH9_RGB sh_diffuse = convolve_sh9_lambertian(project_env_to_sh9(in_faces, inSize));
	std::array<std::vector<float>, 6> out_faces;
	for (auto& f : out_faces) f.assign(size_t(outSize) * size_t(outSize) * 4, 0.0f);
	for (int face = 0; face < 6; ++face) {
		for (int y = 0; y < outSize; ++y) {
			for (int x = 0; x < outSize; ++x) {
				float3 n = dir_from_face_uv(face, ((x + 0.5f) / (float)outSize) * 2.0f - 1.0f, ((y + 0.5f) / (float)outSize) * 2.0f - 1.0f);
				float3 E = eval_sh9(sh_diffuse, n);
				size_t o = 4 * (size_t(y) * outSize + x);
				out_faces[face][o + 0] = E.x; out_faces[face][o + 1] = E.y; out_faces[face][o + 2] = E.z; out_faces[face][o + 3] = 1.0f;
			}
		}
	}
	return out_faces;
}

static std::array<std::vector<float>, 6> prefilter_ggx_cube(std::array<std::vector<float>, 6> const& in_faces, int inSize, int outSize, float roughness, uint32_t sampleCount) {
	std::array<std::vector<float>, 6> out_faces;
	for (auto& f : out_faces) f.assign(size_t(outSize) * size_t(outSize) * 4, 0.0f);
	if (roughness <= 0.0f && inSize == outSize) { for (int face = 0; face < 6; ++face) out_faces[face] = in_faces[face]; return out_faces; }

	for (int face = 0; face < 6; ++face) {
		for (int y = 0; y < outSize; ++y) {
			for (int x = 0; x < outSize; ++x) {
				float3 N = dir_from_face_uv(face, ((x + 0.5f) / (float)outSize) * 2.0f - 1.0f, ((y + 0.5f) / (float)outSize) * 2.0f - 1.0f);
				float3 V = N; float3 prefiltered{ 0,0,0 }; float totalWeight = 0.0f;
				for (uint32_t i = 0; i < sampleCount; ++i) {
					float u1, u2; hammersley(i, sampleCount, u1, u2);
					float3 H = importanceSampleGGX(u1, u2, roughness, N);
					float3 L = normalize(add(mul(H, 2.0f * dot(V, H)), mul(V, -1.0f)));
					float NdotL = std::max(0.0f, dot(N, L));
					if (NdotL > 0.0f) {
						float3 Li = sample_cubemap_linear(in_faces, inSize, L);
						float maxRadiance = 20.0f + (1.0f - roughness) * 150.0f;
						float lum = std::max(Li.x, std::max(Li.y, Li.z));
						if (lum > maxRadiance) Li = mul(Li, maxRadiance / lum);
						prefiltered = add(prefiltered, mul(Li, NdotL)); totalWeight += NdotL;
					}
				}
				if (totalWeight > 0.0f) prefiltered = mul(prefiltered, 1.0f / totalWeight);
				size_t o = 4 * (size_t(y) * outSize + x);
				out_faces[face][o + 0] = prefiltered.x; out_faces[face][o + 1] = prefiltered.y; out_faces[face][o + 2] = prefiltered.z; out_faces[face][o + 3] = 1.0f;
			}
		}
	}
	return out_faces;
}

int main(int argc, char** argv) {
	if (argc < 2) return 1;
	if (std::string(argv[1]) == "ggx") {
		std::string in_p = argv[2]; std::string out_b = argv[3];
		uint32_t samples = (argc >= 6) ? (uint32_t)std::atoi(argv[5]) : 4096;
		int w, h, n; stbi_uc* pix = stbi_load(in_p.c_str(), &w, &h, &n, 4);
		if (!pix) return 1;
		std::array<std::vector<float>, 6> in_f; int inS; split_cube_faces_vertical_strip_rgbe(pix, w, h, in_f, inS);
		stbi_image_free(pix);
		int baseS = (argc >= 5 && std::atoi(argv[4]) > 0) ? std::atoi(argv[4]) : inS;
		int maxM = 0; for (int t = baseS; t > 1; t >>= 1) maxM++;

		std::cout << "[GGX] Starting convolution with " << samples << " samples...\n";
		for (int m = 0; m <= maxM; ++m) {
			int outS = baseS >> m; float rough = (float)m / (float)maxM;
			std::cout << "  Processing Mip " << m << " (Roughness: " << rough << ")..." << std::endl;
			auto out_f = prefilter_ggx_cube(in_f, inS, outS, rough, samples);
			pack_cube_faces_vertical_strip_rgbe_png(out_f, outS, out_b.substr(0, out_b.find_last_of('.')) + "_" + std::to_string(m) + ".png");
		}
		std::cout << "[GGX] Finished all mips." << std::endl;
	}
	else {
		std::string in_p = argv[1]; std::string out_p = argv[2]; int outS = (argc >= 4) ? std::atoi(argv[3]) : 64;
		int w, h, n; stbi_uc* pix = stbi_load(in_p.c_str(), &w, &h, &n, 4);
		if (!pix) return 1;
		std::array<std::vector<float>, 6> in_f; int inS; split_cube_faces_vertical_strip_rgbe(pix, w, h, in_f, inS);
		stbi_image_free(pix);
		std::cout << "[Lambertian] Starting SH9 projection..." << std::endl;
		auto out_f = build_lambertian_cube_sh9(in_f, inS, outS);
		pack_cube_faces_vertical_strip_rgbe_png(out_f, outS, out_p);
		std::cout << "[Lambertian] Finished." << std::endl;
	}
	return 0;
}