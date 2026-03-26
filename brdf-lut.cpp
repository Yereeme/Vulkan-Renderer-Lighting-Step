// brdf-lut.cpp
// Build a BRDF integration LUT for split-sum specular IBL.
// Optimized for the LearnOpenGL / Epic Games math.

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "external/tinyobjloader/stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct float3 { float x, y, z; };
static inline float saturate(float x) { return std::max(0.0f, std::min(1.0f, x)); }
static inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float3 normalize(float3 v) {
    float d = std::sqrt(std::max(0.0f, dot(v, v)));
    if (d <= 0.0f) return { 0,0,1 };
    return { v.x / d, v.y / d, v.z / d };
}

// Low-discrepancy sequence helpers
static float radicalInverse_VdC(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

static void hammersley(uint32_t i, uint32_t N, float& u, float& v) {
    u = float(i) / float(N);
    v = radicalInverse_VdC(i);
}

static float3 importanceSampleGGX(float u1, float u2, float roughness, float3 N) {
    float a = roughness * roughness;
    float phi = 2.0f * float(M_PI) * u1;
    float cosTheta = std::sqrt((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 H = { std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta };

    // Basis construction (N is assumed to be 0,0,1 for the LUT)
    float3 up = std::abs(N.z) < 0.999f ? float3{ 0,0,1 } : float3{ 1,0,0 };
    float3 tangent = normalize({ -up.y, up.x, 0 }); // Simplified for N={0,0,1}
    float3 bitangent = { 0, -1, 0 }; // Simplified for N={0,0,1}

    // To world space (simplified since N is up)
    return H;
}

// 🚨 THE CRITICAL IBL FIX 🚨
// For IBL, k = (roughness^2) / 2
static float GeometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

static float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

static void integrateBRDF(float NdotV, float roughness, uint32_t sampleCount, float& outA, float& outB) {
    float3 V = { std::sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV };
    float3 N = { 0.0f, 0.0f, 1.0f };

    float A = 0.0f;
    float B = 0.0f;

    for (uint32_t i = 0; i < sampleCount; ++i) {
        float u1, u2;
        hammersley(i, sampleCount, u1, u2);
        float3 H = importanceSampleGGX(u1, u2, roughness, N);
        float3 L = normalize({ 2.0f * dot(V, H) * H.x - V.x, 2.0f * dot(V, H) * H.y - V.y, 2.0f * dot(V, H) * H.z - V.z });

        float NdotL = saturate(L.z);
        float NdotH = saturate(H.z);
        float VdotH = saturate(dot(V, H));

        if (NdotL > 0.0f) {
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = std::pow(1.0f - VdotH, 5.0f);

            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    outA = A / float(sampleCount);
    outB = B / float(sampleCount);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string path = argv[1];
    int size = (argc >= 3) ? std::atoi(argv[2]) : 512;
    uint32_t samples = (argc >= 4) ? std::atoi(argv[3]) : 4096;

    std::vector<uint8_t> img(size * size * 4);
    for (int y = 0; y < size; ++y) {
        float roughness = (float(y) + 0.5f) / float(size);
        for (int x = 0; x < size; ++x) {
            float NdotV = (float(x) + 0.5f) / float(size);
            float A, B;
            integrateBRDF(NdotV, roughness, samples, A, B);

            size_t i = 4 * (size_t(y) * size_t(size) + size_t(x));
            img[i + 0] = uint8_t(saturate(A) * 255.0f);
            img[i + 1] = uint8_t(saturate(B) * 255.0f);
            img[i + 2] = 0;
            img[i + 3] = 255;
        }
    }
    stbi_write_png(path.c_str(), size, size, 4, img.data(), size * 4);
    return 0;
}