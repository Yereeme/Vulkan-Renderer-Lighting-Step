#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent; // The 'w' is the handedness

layout(push_constant) uniform Push {
    mat4 CLIP_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL;
    vec3 camera_ws;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out mat3 outTBN;

void main() {
outTexCoord = inTexCoord;
   
    

   // Inside pbr.vert main
// Inside pbr.vert main()
vec4 worldPos = pc.WORLD_FROM_LOCAL * vec4(inPosition, 1.0);
outWorldPos = worldPos.xyz;

// Normal matrix math for TBN
vec3 N = normalize(mat3(pc.WORLD_FROM_LOCAL) * inNormal);
vec3 T = normalize(mat3(pc.WORLD_FROM_LOCAL) * inTangent.xyz);
T = normalize(T - dot(T, N) * N); // Re-orthogonalize
vec3 B = cross(N, T) * inTangent.w;

outTBN = mat3(T, B, N);    gl_Position = pc.CLIP_FROM_LOCAL * vec4(inPosition, 1.0);
}