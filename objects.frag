#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord; 
layout(location = 3) in vec4 tangent;

layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION; float _pad0;
    vec3 SKY_ENERGY;    float _pad1;
    vec3 SUN_DIRECTION; float _pad2;
    vec3 SUN_ENERGY;    float _pad3;

    vec3 CAMERA_POSITION; float _pad4;
};

layout(set=2,binding=0) uniform sampler2D TEXTURE;
layout(set=2,binding=1) uniform sampler2D NORMAL_MAP;
layout(set=3,binding=0) uniform samplerCube ENV_LAMBERTIAN;

layout(location = 0) out vec4 outColor;

void main() {
    // ----- base material color -----
    vec3 albedo = texture(TEXTURE, texCoord).rgb;

    // ----- build TBN frame in world space -----
    vec3 N = normalize(normal);
    vec3 T = normalize(tangent.xyz);

    // Gram-Schmidt orthogonalize
    T = normalize(T - N * dot(N, T));

    // bitangent from cross product + handedness sign
    vec3 B = normalize(cross(N, T)) * tangent.w;
    mat3 TBN = mat3(T, B, N);

    // ----- decode normal map -----
    vec3 n_ts = texture(NORMAL_MAP, texCoord).xyz * 2.0 - 1.0;
    
    // CRITICAL VULKAN FIX: Flip the Y axis so shadows cast correctly!
    n_ts.y = -n_ts.y; 
    
    n_ts = normalize(n_ts);

    // convert tangent-space normal -> world-space normal
    vec3 N_ws = normalize(TBN * n_ts);

    // ----- lambertian irradiance lookup -----
    vec3 E = texture(ENV_LAMBERTIAN, N_ws).rgb;

    // keep World block "used" so compiler doesn't warn:
    E += SKY_ENERGY * 0.0;

    // lambert diffuse: albedo * (E / pi)
    vec3 diffuse = albedo * (E * (1.0 / 3.14159265359));

    outColor = vec4(diffuse, 1.0);
}