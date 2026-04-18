#version 450
#include "tone_map.glsl"

layout(location = 0) in vec3 inWorldPos; 
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
    mat4 LIGHT_CLIP_FROM_WORLD;

    vec3 camera_ws;
    float exposure;

    int tone_op;
    int SHADOW_LIGHT_INDEX;
    int _pad0;
    int _pad1;
} pc;

layout(set=2, binding=0) uniform sampler2D TEXTURE;
layout(set=2, binding=1) uniform sampler2D NORMAL_MAP;
layout(set=2, binding=2) uniform sampler2D ROUGHNESS_MAP;
layout(set=2, binding=3) uniform sampler2D METALNESS_MAP;

layout(set=3, binding=0) uniform samplerCube ENV_LAMBERTIAN;
layout(set=3, binding=1) uniform samplerCube ENV_GGX;
layout(set=3, binding=2) uniform sampler2D   BRDF_LUT;


struct GPULight {
    vec4 position;
    vec4 direction;
    vec4 tint;
    vec4 params;
};

layout(set = 4, binding = 0) readonly buffer Lights {
    GPULight lights[];
};

// Shadow map written in the shadow pass, read in the lighting pass.
 
 const int MAX_SHADOW_SPOT_LIGHTS = 16;
layout(set = 5, binding = 0) uniform sampler2D SHADOW_MAPS[MAX_SHADOW_SPOT_LIGHTS];


layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

const float SHADOW_NEAR_PLANE = 0.1;
const float SHADOW_FAR_PLANE = 100.0;
const float SHADOW_A = -0.5 - 0.5 * (SHADOW_FAR_PLANE + SHADOW_NEAR_PLANE) / (SHADOW_FAR_PLANE - SHADOW_NEAR_PLANE);
const float SHADOW_B = -(SHADOW_FAR_PLANE * SHADOW_NEAR_PLANE) / (SHADOW_FAR_PLANE - SHADOW_NEAR_PLANE);

float linearize_shadow_depth(float depth01) {
    return 1.0 / max(SHADOW_A - depth01 * SHADOW_B, 0.0001);
}

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

//   Geometry terms for Direct vs IBL
float GeometrySchlickGGX(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    // For Direct Lighting: k = (roughness + 1)^2 / 8
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return GeometrySchlickGGX(max(dot(N, V), 0.0), k) * GeometrySchlickGGX(max(dot(N, L), 0.0), k);
}


// Returns 1.0 when lit, 0.0 when shadowed.
 
 // Uses a 4-tap PCF kernel for softer shadow edges.
mat4 make_spot_light_matrix(GPULight light) {
    vec3 F = normalize(light.direction.xyz);
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(F, up)) > 0.99) up = vec3(1.0, 0.0, 0.0);

    vec3 R = normalize(cross(up, F));
    vec3 U = cross(F, R);
    vec3 P = light.position.xyz;

    mat4 view = mat4(
        R.x, U.x, -F.x, 0.0,
        R.y, U.y, -F.y, 0.0,
        R.z, U.z, -F.z, 0.0,
        -dot(R, P), -dot(U, P), dot(F, P), 1.0
    );

    float near_ = 0.1;
    float far_ = 100.0;
    float e = 1.0 / tan(light.params.w * 0.5);
    float A = -0.5 - 0.5 * (far_ + near_) / (far_ - near_);
    float B = -(far_ * near_) / (far_ - near_);

    mat4 proj = mat4(
        e,   0.0, 0.0,  0.0,
        0.0, -e,  0.0,  0.0,
        0.0, 0.0, A,   -1.0,
        0.0, 0.0, B,    0.0
    );
    return proj * view;
}

float sample_shadow(
    vec3 worldPos,
    vec3 N,
    vec3 lightDir,
    mat4 light_clip_from_world,
    int shadowSlot,
    float lightRadius,
    float lightFov
)
{
    vec4 lightClip = light_clip_from_world * vec4(worldPos, 1.0);
    if (lightClip.w <= 0.0) return 1.0;
    vec3 lightNDC = lightClip.xyz / lightClip.w;

    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;
    float currentDepth = lightNDC.z  ;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }

    

    float bias = max(0.0005, 0.002 * (1.0 - max(dot(N, lightDir), 0.0)));

    float receiverDepth = currentDepth - bias;

    vec2 texelSize = 1.0 / vec2(textureSize(SHADOW_MAPS[shadowSlot], 0));

    //16-sample Poisson disk.
   float zReceiver = linearize_shadow_depth(receiverDepth);

    const int POISSON_COUNT = 16;
    vec2 poisson[POISSON_COUNT] = vec2[](
        vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
        vec2(-0.09418410, -0.92938870), vec2(0.34495938,  0.29387760),
        vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
        vec2(-0.38277543,  0.27676845), vec2(0.97484398,  0.75648379),
        vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
        vec2(-0.26496911, -0.41893023), vec2(0.79197514,  0.19090188),
        vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
        vec2(0.19984126,  0.78641367), vec2(0.14383161, -0.14100790)
    );

     // 1) Blocker search (PCSS reference style):
    // search width grows with light size and receiver distance from light.
    float lightSizeUV = lightRadius / max(tan(lightFov * 0.5) * SHADOW_FAR_PLANE, 0.0001);
    float searchRadiusUVScalar = lightSizeUV * max((zReceiver - SHADOW_NEAR_PLANE) / max(zReceiver, SHADOW_NEAR_PLANE), 0.0);
    vec2 searchRadiusUV = vec2(max(searchRadiusUVScalar, texelSize.x));
    float blockerSum = 0.0;
    float blockerCount = 0.0;
    for (int i = 0; i < POISSON_COUNT; ++i) {
        float sampleDepth = texture(SHADOW_MAPS[shadowSlot], shadowUV + poisson[i] * searchRadiusUV).r;
        if (sampleDepth < receiverDepth) {
            blockerSum += sampleDepth;
            blockerCount += 1.0;
        }
    }
   if (blockerCount < 0.5) return 1.0;
    float avgBlockerDepth = blockerSum / blockerCount;
    float zBlocker = linearize_shadow_depth(avgBlockerDepth);

    // 2) Penumbra size estimation from PCSS:
    // wPenumbra = ((dReceiver - dBlocker) / dBlocker) * wLight
    float penumbraUV = max((zReceiver - zBlocker) / max(zBlocker, 0.0001), 0.0) * lightSizeUV;
    vec2 filterRadiusUV = vec2(clamp(penumbraUV, max(texelSize.x, texelSize.y), 40.0 * max(texelSize.x, texelSize.y)));

    // 3) Visibility estimate with variable-radius PCF.
    float lit = 0.0;
    for (int i = 0; i < POISSON_COUNT; ++i) {
        float closestDepth = texture(SHADOW_MAPS[shadowSlot], shadowUV + poisson[i] * filterRadiusUV).r;
        lit += (receiverDepth > closestDepth) ? 0.0 : 1.0;
    }
    return lit / float(POISSON_COUNT);
    
}

void main() {
 
    // 1. RE-NORMALIZE VECTORS
   
    vec3 worldPos = inWorldPos;
    vec3 V = normalize(pc.camera_ws - worldPos);
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

    
   

    // 4. INDIRECT LIGHTING (IBL)
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

    // 5. FINAL COMBINATION
    vec3 directLights = vec3(0.0);

    int shadowSlot = 0;

for (int i = 0; i < lights.length(); ++i) {
    float lightType = lights[i].position.w;

    int myShadowSlot = -1;
    if (lightType == 2.0 && lights[i].tint.w > 0.0 && shadowSlot < MAX_SHADOW_SPOT_LIGHTS) {
        myShadowSlot = shadowSlot;
        shadowSlot += 1;
    }
   

    // for now: treat sphere + spot as point-style direct lights

    if (lightType == 0.0) {
        vec3 sunDir = normalize(lights[i].direction.xyz);
        float sunAngle = max(lights[i].params.x, 0.0001);
        float sunStrength = lights[i].params.y;

        float NdotLsun = max(dot(N, sunDir), 0.0);
        if (NdotLsun <= 0.0) continue;

        vec3 Rsun = normalize(reflect(-V, N));
        float cosSun = cos(sunAngle);
        float cosToCenter = dot(Rsun, sunDir);
        vec3 sunSpecDir = Rsun;

        if (cosToCenter < cosSun) {
            vec3 axis = cross(sunDir, Rsun);
            float axisLen = length(axis);
            if (axisLen > 0.0001) {
                axis /= axisLen;
                vec3 tangent = normalize(cross(axis, sunDir));
                float sinSun = sqrt(max(1.0 - cosSun * cosSun, 0.0));
                sunSpecDir = normalize(sunDir * cosSun + tangent * sinSun);
            } else {
                sunSpecDir = sunDir;
            }
        }

        float NdotLsunSpec = max(dot(N, sunSpecDir), 0.0);
        if (NdotLsunSpec <= 0.0) continue;

        vec3 Hsun = normalize(V + sunSpecDir);
        float NDFsun = DistributionGGX(N, Hsun, roughness);
        float Gsun = GeometrySmith(N, V, sunSpecDir, roughness);
        vec3 Fsun = fresnelSchlick(max(dot(Hsun, V), 0.0), F0);
        vec3 kD_sun = (vec3(1.0) - Fsun) * (1.0 - metallic);
        vec3 diffuseSun = kD_sun * albedo / PI;
        vec3 specularSun = (NDFsun * Gsun * Fsun) / (4.0 * NdotV * NdotLsunSpec + 0.0001);

        vec3 sunRadiance = lights[i].tint.rgb * sunStrength;
        directLights += (diffuseSun * NdotLsun + specularSun * NdotLsunSpec) * sunRadiance;
        continue;
    }

    // treat sphere + spot as point/area approximations

    if (lightType == 1.0 || lightType == 2.0) {
        vec3 Lvec = lights[i].position.xyz - worldPos;
        float d = length(Lvec);
        vec3 Ldyn = Lvec / max(d, 0.0001);

        

// sphere representative point for specular
vec3 Lspec = -Ldyn;

if (lightType == 1.0 || lightType == 2.0) {
    vec3 lightCenter = lights[i].position.xyz;
    float radius = lights[i].params.x;

    vec3 Rdir = normalize(reflect(-V, N));
    vec3 centerToSurface = worldPos - lightCenter;

    vec3 repPoint = lightCenter + normalize(Rdir * radius - centerToSurface) * radius;

    Lspec = normalize(repPoint - worldPos);
}

       
//cone
float spotFactor = 1.0;

if (lightType == 2.0) {
    vec3 lightDir = normalize(lights[i].direction.xyz);
    vec3 LtoSurface = -Ldyn;

    float cosTheta = dot(lightDir, LtoSurface);
    // params.w stores the full spotlight FOV angle.
    // Cone tests should use the half-angle from center to edge.
   // params.w stores the full spotlight FOV angle.
    // Cone tests should use the half-angle from center to edge.
    float cosOuter = cos(lights[i].params.w * 0.5);
    // direction.w controls penumbra blend [0,1]:
    // 0.0 = hard edge (inner == outer), 1.0 = widest soft edge.
    float blend = clamp(lights[i].direction.w, 0.0, 1.0);

      float cosInner = mix(cosOuter, 1.0, blend);

    spotFactor = clamp(
        (cosTheta - cosOuter) / max(cosInner - cosOuter, 0.0001),
        0.0,
        1.0
    );

    if (spotFactor <= 0.0) {
        continue;
    }
}

        float NdotLdyn = max(dot(N, Ldyn), 0.0);
float NdotLspec = max(dot(N, Lspec), 0.0);
        if (NdotLdyn <= 0.0) continue;

        vec3 Hdyn = normalize(V + Lspec);

        float NDFdyn = DistributionGGX(N, Hdyn, roughness);
        float Gdyn = GeometrySmith(N, V, Lspec, roughness);
        vec3 Fdyn = fresnelSchlick(max(dot(Hdyn, V), 0.0), F0);

        vec3 kDdyn = (vec3(1.0) - Fdyn) * (1.0 - metallic);
        vec3 diffuseDyn = kDdyn * albedo / PI;
        vec3 specularDyn = (NDFdyn * Gdyn * Fdyn) / (4.0 * NdotV * NdotLspec + 0.0001);

        float physicalFalloff = 1.0 / max(d * d, 0.0001);

        float limit = lights[i].params.z;
        float limitFalloff = 1.0;
        if (limit > 0.0) {
            float x = d / limit;
            limitFalloff = max(0.0, 1.0 - pow(x, 4.0));
        }

        float attenuation = physicalFalloff * limitFalloff;
        //vec3 radiance = lights[i].tint.rgb * lights[i].params.y * attenuation * spotFactor;
        vec3 radiance = lights[i].tint.rgb * lights[i].params.y * attenuation * spotFactor;
float shadow = 1.0;

if (myShadowSlot >= 0) {
    mat4 light_clip_from_world = make_spot_light_matrix(lights[i]);
   shadow = sample_shadow(worldPos, N, Ldyn, light_clip_from_world, myShadowSlot, lights[i].params.x, lights[i].params.w);
}

directLights += shadow * (diffuseDyn + specularDyn) * radiance * NdotLdyn;
    }
}

vec3 ambient = (kD_ibl * diffuseIBL) + specularIBL;
vec3 finalColor = ambient + directLights;

outColor = vec4(apply_tone_map(finalColor * exp2(pc.exposure), pc.tone_op), 1.0);
}