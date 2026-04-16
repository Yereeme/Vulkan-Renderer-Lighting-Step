#pragma once

#include "Tutorial.hpp"

S72::vec3 make_s72_vec3(float x, float y, float z);
mat4 make_local_from_node(S72::Node const* node);
S72::vec3 get_translation_from_mat4(mat4 const& m);
S72::vec3 get_world_neg_z_direction(mat4 const& m);
//mat4 make_spot_light_matrix(LoadedLight const& l);
mat4 mat4_inverse_rigid(mat4 const& M);
mat4 make_spot_light_matrix(Tutorial::LoadedLight const& l);


void collect_loaded_lights_from_node(
	S72::Node const* node,
	mat4 const& parent_world_from_local,
	std::vector<Tutorial::LoadedLight>& out
);