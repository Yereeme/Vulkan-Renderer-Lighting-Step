#version 450
#include "tone_map.glsl"

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec2 texCoord; 
layout(location = 2) in mat3 inTBN;      

layout(set=0, binding=0, std140) uniform World {
    vec3 SKY_DIRECTION;  float _pad0;
    vec3 SKY_ENERGY;     float _pad1;
    vec3 SUN_DIRECTION;  float _pad2;
    vec3 SUN_ENERGY;     float _pad3;
};

layout(push_constant) uniform Push {
    mat4 CLIP_FROM_LOCAL;
    mat4 WORLD_FROM_LOCAL;
    vec3 camera_ws;
    float exposure;
    int tone_op;
} pc;

layout(set=2, binding=0) uniform sampler2D TEXTURE;
layout(set=2, binding=1) uniform sampler2D NORMAL_MAP;
layout(set=2, binding=2) uniform sampler2D ROUGHNESS_MAP;
layout(set=2, binding=3) uniform sampler2D METALNESS_MAP;

layout(set=3, binding=0) uniform samplerCube ENV_LAMBERTIAN;
layout(set=3, binding=1) uniform samplerCube ENV_GGX;
layout(set=3, binding=2) uniform sampler2D   BRDF_LUT;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// --- MATH HELPERS ---
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// 🚨 THE FIX: Geometry terms for Direct vs IBL
float GeometrySchlickGGX(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    // For Direct Lighting: k = (roughness + 1)^2 / 8
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return GeometrySchlickGGX(max(dot(N, V), 0.0), k) * GeometrySchlickGGX(max(dot(N, L), 0.0), k);
}

void main() {
    // 1. RE-NORMALIZE VECTORS 
    vec3 V = normalize(pc.camera_ws - inPosition);
    vec3 Ngeom = normalize(inTBN[2]);
    
    // 2. NORMAL MAPPING
    vec3 n_ts = texture(NORMAL_MAP, texCoord).xyz * 2.0 - 1.0;
    n_ts.y = -n_ts.y; 
    vec3 N = normalize(inTBN * n_ts);

    float NdotV = clamp(dot(N, V), 0.0001, 1.0);
    vec3 R = reflect(-V, N);

    // 3. MATERIAL PROPERTIES
    vec3 albedo = texture(TEXTURE, texCoord).rgb;
    float roughness = clamp(texture(ROUGHNESS_MAP, texCoord).r, 0.04, 1.0);
    float metallic  = clamp(texture(METALNESS_MAP, texCoord).r, 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // 4. DIRECT LIGHTING (SUN)
    vec3 L = normalize(SUN_DIRECTION);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kD_sun = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 specularMath = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    vec3 diffuseSun  = (kD_sun * albedo / PI) * SUN_ENERGY * NdotL;
    vec3 specularSun = specularMath * SUN_ENERGY * NdotL;

    // 5. INDIRECT LIGHTING (IBL)
    vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);

    // Irradiance (No /PI here because the fixed SH9 tool outputs raw irradiance)
    vec3 irradiance = texture(ENV_LAMBERTIAN, N).rgb;
    vec3 diffuseIBL = albedo * (irradiance * (1.0 / PI));
    
    float maxMip = 9.0; 
    vec3 prefiltered = textureLod(ENV_GGX, R, roughness * maxMip).rgb;
    
    // Sample the fixed LUT (R=Scale, G=Bias)
    //vec2 brdfEnv = texture(BRDF_LUT, vec2(NdotV, roughness)).rg; 
    vec2 brdfEnv = texture(BRDF_LUT, vec2(NdotV, 1.0 - roughness)).rg;
    vec3 specularIBL = prefiltered * (F0 * brdfEnv.x + brdfEnv.y);

    // 6. FINAL COMBINATION
    vec3 ambient = (kD_ibl * diffuseIBL) + specularIBL;
    vec3 finalColor = ambient + diffuseSun + specularSun;

    outColor = vec4(apply_tone_map(finalColor * exp2(pc.exposure), pc.tone_op), 1.0);
}