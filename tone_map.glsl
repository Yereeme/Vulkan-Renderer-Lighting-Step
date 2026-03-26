#ifndef TONE_MAP_GLSL
#define TONE_MAP_GLSL

// Exposure is applied before calling these functions.
// These expect linear HDR input.

vec3 tonemap_linear(vec3 x) {
    return x;
}

vec3 tonemap_reinhard(vec3 x) {
    return x / (1.0 + x);
}

// Optional helper for one unified entry point:
vec3 apply_tone_map(vec3 color, int op) {
    if (op == 0) {
        return tonemap_linear(color);
    } else {
        return tonemap_reinhard(color);
    }
}

#endif