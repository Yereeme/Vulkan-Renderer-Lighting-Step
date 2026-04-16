#include "LightHelpers.hpp"

#include <cmath>
#include <iostream>

S72::vec3 make_s72_vec3(float x, float y, float z) {
	return { x, y, z };
}

mat4 mat4_inverse_rigid(mat4 const& M) {
	// Assumes M is rotation + translation only (no scale/shear).

	float r00 = M[0], r01 = M[4], r02 = M[8];
	float r10 = M[1], r11 = M[5], r12 = M[9];
	float r20 = M[2], r21 = M[6], r22 = M[10];

	// transpose rotation
	float t00 = r00, t01 = r10, t02 = r20;
	float t10 = r01, t11 = r11, t12 = r21;
	float t20 = r02, t21 = r12, t22 = r22;

	float tx = M[12], ty = M[13], tz = M[14];

	float ntx = -(t00 * tx + t01 * ty + t02 * tz);
	float nty = -(t10 * tx + t11 * ty + t12 * tz);
	float ntz = -(t20 * tx + t21 * ty + t22 * tz);

	return mat4{
		t00, t10, t20, 0.0f,
		t01, t11, t21, 0.0f,
		t02, t12, t22, 0.0f,
		ntx, nty, ntz, 1.0f
	};
}

mat4 make_local_from_node(S72::Node const* node) {
	if (!node) {
		return mat4{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
	}

	float sx = node->scale.x;
	float sy = node->scale.y;
	float sz = node->scale.z;

	mat4 S = {
		sx,   0.0f, 0.0f, 0.0f,
		0.0f, sy,   0.0f, 0.0f,
		0.0f, 0.0f, sz,   0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	float x = node->rotation.x;
	float y = node->rotation.y;
	float z = node->rotation.z;
	float w = node->rotation.w;

	mat4 R = {
		1.0f - 2.0f * y * y - 2.0f * z * z,  2.0f * x * y + 2.0f * w * z,         2.0f * x * z - 2.0f * w * y,         0.0f,
		2.0f * x * y - 2.0f * w * z,         1.0f - 2.0f * x * x - 2.0f * z * z,  2.0f * y * z + 2.0f * w * x,         0.0f,
		2.0f * x * z + 2.0f * w * y,         2.0f * y * z - 2.0f * w * x,         1.0f - 2.0f * x * x - 2.0f * y * y,  0.0f,
		0.0f,                        0.0f,                        0.0f,                        1.0f
	};

	mat4 T = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		node->translation.x, node->translation.y, node->translation.z, 1.0f
	};

	return T * R * S;
}

mat4 make_spot_light_matrix(Tutorial::LoadedLight const& l) {
	S72::vec3 F = l.world_direction;
	float flen = std::sqrt(F.x * F.x + F.y * F.y + F.z * F.z);
	if (flen <= 0.0001f) {
		F = { 0.0f, 0.0f, -1.0f };
	}
	else {
		F.x /= flen;
		F.y /= flen;
		F.z /= flen;
	}

	S72::vec3 up = { 0.0f, 1.0f, 0.0f };
	float d = F.x * up.x + F.y * up.y + F.z * up.z;
	if (std::abs(d) > 0.99f) {
		up = { 1.0f, 0.0f, 0.0f };
	}

	S72::vec3 R{
		up.y * F.z - up.z * F.y,
		up.z * F.x - up.x * F.z,
		up.x * F.y - up.y * F.x
	};
	float rlen = std::sqrt(R.x * R.x + R.y * R.y + R.z * R.z);
	R.x /= rlen;
	R.y /= rlen;
	R.z /= rlen;

	S72::vec3 U{
		F.y * R.z - F.z * R.y,
		F.z * R.x - F.x * R.z,
		F.x * R.y - F.y * R.x
	};

	S72::vec3 P = l.world_position;

	mat4 view{
		R.x, U.x, -F.x, 0.0f,
		R.y, U.y, -F.y, 0.0f,
		R.z, U.z, -F.z, 0.0f,
		-(R.x * P.x + R.y * P.y + R.z * P.z),
		-(U.x * P.x + U.y * P.y + U.z * P.z),
		 (F.x * P.x + F.y * P.y + F.z * P.z),
		1.0f
	};

	float aspect = 1.0f;
	float near = 0.1f;
	float far = 100.0f;
	float fov = l.fov;

	mat4 proj = perspective(fov, aspect, near, far);
	return proj * view;
}

S72::vec3 get_translation_from_mat4(mat4 const& m) {
	return { m[12], m[13], m[14] };
}

S72::vec3 get_world_neg_z_direction(mat4 const& m) {
	float x = -m[8];
	float y = -m[9];
	float z = -m[10];

	float len = std::sqrt(x * x + y * y + z * z);
	if (len > 0.0f) {
		x /= len;
		y /= len;
		z /= len;
	}
	else {
		x = 0.0f;
		y = 0.0f;
		z = -1.0f;
	}

	return { x, y, z };
}

void collect_loaded_lights_from_node(
	S72::Node const* node,
	mat4 const& parent_world_from_local,
	std::vector<Tutorial::LoadedLight>& out
) {
	if (!node) {
		std::cout << "node is null\n";
		return;
	}

	//std::cout << "visiting node: " << node->name << std::endl;

	mat4 local_from_node = make_local_from_node(node);
	// compose transforms in parent->child order:
// WORLD_FROM_LOCAL = PARENT_WORLD_FROM_LOCAL * LOCAL_FROM_NODE
	mat4 world_from_local = parent_world_from_local * local_from_node;

	if (node->light) {
		//std::cout << "node " << node->name << " has light ptr " << node->light << std::endl;

		// safety check (prevents crash)
		if (!node->light) {
			std::cout << "INVALID LIGHT POINTER\n";
			return;
		}

		Tutorial::LoadedLight light{};

		light.name = node->light->name;
		light.tint = { node->light->tint.r, node->light->tint.g, node->light->tint.b };
		light.shadow = float(node->light->shadow);
		light.world_from_local = world_from_local;

		light.world_position = get_translation_from_mat4(world_from_local);
		light.world_direction = get_world_neg_z_direction(world_from_local);
		//std::cout << "light " << light.name
			//<< " pos: " << light.world_position.x << ", "
			//<< light.world_position.y << ", "
			//<< light.world_position.z << std::endl;

		if (std::holds_alternative<S72::Light::Sun>(node->light->source)) {
			auto const& sun = std::get<S72::Light::Sun>(node->light->source);
			light.type = Tutorial::LoadedLight::Type::Sun;
			light.angle = sun.angle;
			light.strength = sun.strength;
		}
		else if (std::holds_alternative<S72::Light::Sphere>(node->light->source)) {
			auto const& sphere = std::get<S72::Light::Sphere>(node->light->source);
			light.type = Tutorial::LoadedLight::Type::Sphere;
			light.radius = sphere.radius;
			light.power = sphere.power;
			light.limit = sphere.limit;
		}
		else if (std::holds_alternative<S72::Light::Spot>(node->light->source)) {
			auto const& spot = std::get<S72::Light::Spot>(node->light->source);
			light.type = Tutorial::LoadedLight::Type::Spot;
			light.radius = spot.radius;
			light.power = spot.power;
			light.limit = spot.limit;
			light.fov = spot.fov;
			light.blend = spot.blend;
		}

		out.emplace_back(light);
	}

	for (auto* child : node->children) {
		if (!child) {
			std::cout << "NULL CHILD in node " << node->name << std::endl;
			continue;
		}

		collect_loaded_lights_from_node(child, world_from_local, out);
	}
}