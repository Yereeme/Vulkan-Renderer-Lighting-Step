#version 450

struct Transform {
    mat4 CLIP_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL_NORMAL;
};

layout(set = 0, binding = 0) readonly buffer Transforms {
    Transform TRANSFORMS[];
};

layout(push_constant) uniform ShadowPush {
    mat4 LIGHT_CLIP_FROM_WORLD;
    int OBJECT_INDEX;
    int _pad0;
    int _pad1;
    int _pad2;
} push_;

layout(location = 0) in vec3 Position;

void main() {
mat4 world_from_local = TRANSFORMS[push_.OBJECT_INDEX].WORLD_FROM_LOCAL;
    vec4 world_pos = world_from_local * vec4(Position, 1.0);
    gl_Position = push_.LIGHT_CLIP_FROM_WORLD * world_pos;
}