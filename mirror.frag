#version 450
#include "tone_map.glsl"

layout(set=0, binding=0) uniform samplerCube envTex;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_world_nrm;

layout(location = 0) out vec4 outColor;

// keep this simple + aligned
layout(push_constant) uniform Push {
    vec3 camera_ws;
    float exposure;

    int tone_op;
    int _pad0;
    int _pad1;
    int _pad2;
} pc;

void main() {
    vec3 N = normalize(v_world_nrm); //N is surface normal
    vec3 V = normalize(pc.camera_ws - v_world_pos); //V is vector from surface to  camera
    vec3 R = reflect(-V, N); //-V is Incident vector camera -> ssurface 

    vec3 col = texture(envTex, R).rgb;
    col *= exp2(pc.exposure);
    col = apply_tone_map(col, pc.tone_op);

    outColor = vec4(col, 1.0);
}