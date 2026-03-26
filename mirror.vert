#version 450

struct Transform {
    mat4 CLIP_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL_NORMAL;
};

layout(set=1, binding=0, std140) readonly buffer Transforms {
    Transform TRANSFORMS[];
};

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoord; // keep to match PosNorTexVertex
layout(location = 3) in vec4 Tangent;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_world_nrm;

void main() {
    Transform T = TRANSFORMS[gl_InstanceIndex];

    gl_Position = T.CLIP_FROM_LOCAL * vec4(Position, 1.0);

    vec4 wp = T.WORLD_FROM_LOCAL * vec4(Position, 1.0);
    v_world_pos = wp.xyz;

    v_world_nrm = mat3(T.WORLD_FROM_LOCAL_NORMAL) * Normal;
}