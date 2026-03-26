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
// --- PERFECTLY MATCHING C++ LOCATIONS ---
layout(location = 2) in vec2 TexCoord; // C++ offsetof TexCoord is at location 2
layout(location = 3) in vec4 Tangent;  // C++ offsetof Tangent is at location 3

// --- OUTPUTS TO FRAGMENT SHADER ---
layout(location = 0) out vec3 position; 
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 texCoord;  
layout(location = 3) out vec4 tangent;

void main() {
    Transform transform = TRANSFORMS[gl_InstanceIndex];
    
    gl_Position = transform.CLIP_FROM_LOCAL * vec4(Position, 1.0);
    position = (transform.WORLD_FROM_LOCAL * vec4(Position, 1.0)).xyz;
    
    // Transform Normal using the Normal Matrix (Inverse-Transpose)
    normal = mat3(transform.WORLD_FROM_LOCAL_NORMAL) * Normal;
    
    // Transform Tangent using standard Model Matrix
    vec3 T_world = mat3(transform.WORLD_FROM_LOCAL) * Tangent.xyz;
    tangent = vec4(T_world, Tangent.w);
    
    texCoord = TexCoord;
}