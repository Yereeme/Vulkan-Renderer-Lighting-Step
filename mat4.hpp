#pragma once

#include <array>
#include <cmath>
#include <cstdint>

//NOTE: column-major storage order (like in OpenGL/GLSL):
using mat4 = std::array< float, 16 >;
static_assert(sizeof(mat4) == 16*4, "mat4 is exactly 16 32-bit floats.");

using vec4 = std::array< float, 4 >;
static_assert(sizeof(vec4) == 4*4, "vec4 is exactly 4 32-bit floats.");

//applying the linear function tabulated in a matrix to a vector, i.e, matrix-vector multiply:
inline vec4 operator*(mat4 const &A, vec4 const &b) {
	vec4 ret;
	//compute ret = A * b:
	for (uint32_t r = 0; r < 4; ++r) {
		ret[r] = A[0 * 4 + r] * b[0];
		for (uint32_t k = 1; k < 4; ++k) {
			ret[r] += A[k * 4 + r] * b[k];
		}
	}
	return ret;
}

//composition of two linear functions tabulated in matrices; matrix-matrix multiplication 
inline mat4 operator*(mat4 const &A, mat4 const &B) {
	mat4 ret;
	//compute ret A * B
	for (uint32_t c = 0; c < 4; ++c) {
		for (uint32_t r = 0; r < 4; ++r) {
			ret[c * 4 + r] = A[0 * 4 + r] * B[c * 4 + 0];
			for (uint32_t k = 1; k < 4; ++k) {
				ret[c * 4 + r] += A[k * 4 + r] * B[c * 4 + k];
			}
		}
	}
	return ret;
}

//perspective projection matrix.
// - vfov is fov *in radians*
// - near maps to 0, far maps to 1
//looks down -z with +y up and +x right
inline mat4 perspective(float vfov, float aspect, float near, float far) {
	//as per https://www.terathon.com/gdc07_lengyel.pdf
	// (with modifications for Vulkan-style coordinate system)
	//  notably: flip y (vulkan device coords are y-down)
	//       and rescale z (vulkan device coords are z-[0,1])
	const float e = 1.0f / std::tan(vfov / 2.0f); //e = focal length = 1 / tan(FOV / 2)
	const float a = aspect; // viewport height / width
	const float n = near; //n = distance to near plane
	const float f = far; //f = distance to far plane
	return mat4{ //note: column-major storage order! 
		//gdc07_lengyel 05/43
		e/a, 0.0f,                       0.0f, 0.0f,
		0.0f,   -e,                      0.0f, 0.0f,
		0.0f, 0.0f,-0.5f - 0.5f * (f+n)/(f-n),-1.0f,
		0.0f, 0.0f,             -(f*n)/(f-n), 0.0f,
	};
}

//look at matrix:
//make a camera-space-from-world matrix for a camera at eye looking toward
// target with up-vector pointing (as-close-as-possible) along up.
//that is, it maps:
// - eye_xyz to the origin
// - the unit length vector from eye_xyz to target_xyz to -z
// - an as-close-as-possible unit-length vector to up to +y
inline mat4 look_at(
	float eye_x, float eye_y, float eye_z,
	float target_x, float target_y, float target_z,
	float up_x, float up_y, float up_z ) {
	//note this would be a lot cleaner with a vec3 type and some overloads

	//compute vector from eye to target
	//formula is subtraction form target to start
	float in_x = target_x - eye_x;
	float in_y = target_y - eye_y;
	float in_z = target_z - eye_z;

	//normalize "in" vector
	float  inv_in_len = 1.0f / std::sqrt(in_x*in_x + in_y*in_y + in_z*in_z);
	in_x *= inv_in_len;
	in_y *= inv_in_len;
	in_z *= inv_in_len;

	//make "up" orthogonal to "in"
	//gram smidth
	float in_dot_up = in_x*up_x + in_y*up_y + in_z*up_z; //dot product formula
	up_x -= in_dot_up * in_x;
	up_y -= in_dot_up * in_y;
	up_z -= in_dot_up * in_z;

	//normalize "up" vector:
	float inv_up_len = 1.0f / std::sqrt(up_x*up_x + up_y*up_y + up_z*up_z);
	up_x *= inv_up_len;
	up_y *= inv_up_len;
	up_z *= inv_up_len;

	//compute 'right' vector as 'in' x 'up'
	//cross product A × B = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1).
	float right_x = in_y * up_z - in_z * up_y;
	float right_y = in_z * up_x - in_x * up_z;
	float right_z = in_x * up_y - in_y * up_x;

	//compute dot of right, in, up with eye
	float right_dot_eye = right_x*eye_x + right_y*eye_y + right_z*eye_z;
	float in_dot_eye = in_x*eye_x + in_y*eye_y + in_z*eye_z;
	float up_dot_eye = up_x*eye_x + up_y*eye_y + up_z*eye_z;

	//final matrix: (computes (right . (v - eye), up . (v - eye), -in . (v-eye), v.w )
	return mat4{ //note: column-major storage order
		right_x, up_x, -in_x, 0.0f,
		right_y, up_y, -in_y, 0.0f,
		right_z, up_z, -in_z, 0.0f,
		-right_dot_eye, -up_dot_eye, in_dot_eye, 1.0f,
	};
}

//orbit camera matrix:
	//makes a camera-from-world matrix for a camera orbiting target_{x,y,z}
	//  at distance radius with angles azimuth and elevation.
	// azimuth is counterclockwise angle in the xy plane from the x axis
	// elevation is angle up from xy plane
	// both are in radians
inline mat4 orbit(
	float target_x, float target_y, float target_z,
	float azimuth, float elevation, float radius
   ) {

	//shorthand for some useful trig values:
	float ca = std::cos(azimuth);
	float sa = std::sin(azimuth);
	float ce = std::cos(elevation);
	float se = std::sin(elevation);

	//camera's right direction (vector) is azimuth rotated by 90 degrees:
	float right_x = -sa;
	float right_y = ca;
	float right_z = 0.0f;

	//TODO:camera's up direction (vector) is elevation rotated by 90 degrees:
	// (and points in the same xy direction as azimuth)
	float up_x = -se * ca;
	float up_y = -se * sa;
	float up_z = ce;

	//direction to the camera from the target:
	float out_x = ce * ca;
	float out_y = ce * sa;
	float out_z = se;

	//camera position
	float eye_x = target_x + radius * out_x;
	float eye_y = target_y + radius * out_y;
	float eye_z = target_z + radius * out_z;


	//assemble and return camera-from-world matrix
	//camera's position position projected onto the various vectors:
	float right_dot_eye = right_x*eye_x + right_y*eye_y + right_z*eye_z;
	float up_dot_eye = up_x * eye_x + up_y * eye_y + up_z * eye_z;
	float out_dot_eye = out_x * eye_x + out_y * eye_y + out_z * eye_z;

	//the final local-from-world transformation (column-major):
	return mat4{
		right_x, up_x, out_x, 0.0f,
		right_y, up_y, out_y, 0.0f,
		right_z, up_z, out_z, 0.0f,
		-right_dot_eye, -up_dot_eye, -out_dot_eye, 1.0f,
	};
}

