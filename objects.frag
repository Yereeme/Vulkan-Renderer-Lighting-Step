#version 450

// ===== INPUTS FROM VERTEX SHADER =====
layout(location = 0) in vec3 position;   // world-space position (confirmed earlier)
layout(location = 1) in vec3 normal;     // world-space normal (pre-normal-map)
layout(location = 2) in vec2 texCoord;   // UVs
layout(location = 3) in vec4 tangent;    // world-space tangent + handedness


// ===== GLOBAL WORLD DATA =====
layout(set=0,binding=0,std140) uniform World {
    vec3 SKY_DIRECTION; float _pad0;
    vec3 SKY_ENERGY;    float _pad1;
    vec3 SUN_DIRECTION; float _pad2;
    vec3 SUN_ENERGY;    float _pad3;

    vec3 CAMERA_POSITION; float _pad4;
};



// ===== MATERIAL TEXTURES =====
layout(set=2,binding=0) uniform sampler2D TEXTURE;     // albedo
layout(set=2,binding=1) uniform sampler2D NORMAL_MAP;  // normal map


// ===== ENVIRONMENT (IBL) =====
layout(set=3,binding=0) uniform samplerCube ENV_LAMBERTIAN;




// ===== GPU LIGHT STRUCT (must match CPU layout) =====
struct GPULight {
    vec4 position;   // xyz = world position, w = type
    vec4 direction;  // xyz = direction, w = shadow (unused)
    vec4 tint;       // rgb = color
    vec4 params;     // x=radius, y=power, z=limit
};


// ===== LIGHT BUFFER =====
layout(set = 4, binding = 0) readonly buffer Lights {
    GPULight lights[];
};

// Shadow map generated from the light's point of view.
// We sample it in the lighting pass to decide whether this fragment is occluded.
//layout(set = 5, binding = 0) uniform sampler2D SHADOW_MAP;
const int MAX_SHADOW_SPOT_LIGHTS = 16;
layout(set = 5, binding = 0) uniform sampler2D SHADOW_MAPS[MAX_SHADOW_SPOT_LIGHTS];

layout(push_constant) uniform Push {
    mat4 LIGHT_CLIP_FROM_WORLD;
    int SHADOW_LIGHT_INDEX;
    int _pad0;
    int _pad1;
    int _pad2;
} pc;


// ===== OUTPUT =====
layout(location = 0) out vec4 outColor;


// ===== CONSTANT =====
const float PI = 3.14159265359;

const float SHADOW_NEAR_PLANE = 0.1;
const float SHADOW_FAR_PLANE = 100.0;
const float SHADOW_A = -0.5 - 0.5 * (SHADOW_FAR_PLANE + SHADOW_NEAR_PLANE) / (SHADOW_FAR_PLANE - SHADOW_NEAR_PLANE);
const float SHADOW_B = -(SHADOW_FAR_PLANE * SHADOW_NEAR_PLANE) / (SHADOW_FAR_PLANE - SHADOW_NEAR_PLANE);

float linearize_shadow_depth(float depth01) {
    return 1.0 / max(SHADOW_A - depth01 * SHADOW_B, 0.0001);
}

// Returns 1.0 if the fragment is lit by the shadow-casting light,
// 0.0 if it is in shadow.
//
// Core idea:
// - transform current world-space point into the light's clip space
// - convert that to shadow-map UV + depth
// - compare current depth vs depth stored in shadow map
//
// Interview phrasing:
// "Shadow mapping is a depth comparison in light space."
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
    if (lightClip.w <= 0.0) return 1.0; // behind light camera
    vec3 lightNDC = lightClip.xyz / lightClip.w;
    

    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;
   float currentDepth = lightNDC.z;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }

    float bias = max(0.0005, 0.002 * (1.0 - max(dot(N, lightDir), 0.0)));

   float receiverDepth = currentDepth - bias;
    vec2 texelSize = 1.0 / vec2(textureSize(SHADOW_MAPS[shadowSlot], 0));
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

  // 1) blocker search (PCSS reference style)
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

    // 2) penumbra estimate
    float penumbraUV = max((zReceiver - zBlocker) / max(zBlocker, 0.0001), 0.0) * lightSizeUV;
    vec2 filterRadiusUV = vec2(clamp(penumbraUV, max(texelSize.x, texelSize.y), 40.0 * max(texelSize.x, texelSize.y)));

    // 3) variable-radius PCF
    float lit = 0.0;
    for (int i = 0; i < POISSON_COUNT; ++i) {
        float closestDepth = texture(SHADOW_MAPS[shadowSlot], shadowUV + poisson[i] * filterRadiusUV).r;
        lit += (receiverDepth > closestDepth) ? 0.0 : 1.0;
    }
    return lit / float(POISSON_COUNT);
}
void main() {


    // ===== 1. ALBEDO =====
    vec3 albedo = texture(TEXTURE, texCoord).rgb;


    // ===== 2. BUILD TBN (WORLD SPACE) =====
    // This lets us convert tangent-space normal maps into world space
    vec3 N = normalize(normal);
    vec3 T = normalize(tangent.xyz);

    // Gram-Schmidt to ensure orthogonality
    T = normalize(T - N * dot(N, T));

    vec3 B = normalize(cross(N, T)) * tangent.w;
    mat3 TBN = mat3(T, B, N);


    // ===== 3. NORMAL MAP =====
    vec3 n_ts = texture(NORMAL_MAP, texCoord).xyz * 2.0 - 1.0;

    // Vulkan normal maps need Y flipped
    n_ts.y = -n_ts.y;

    n_ts = normalize(n_ts);

    // Convert from tangent space to world space
    vec3 N_ws = normalize(TBN * n_ts);


    // ===== 4. ENVIRONMENT DIFFUSE (IBL) =====
    // Lookup irradiance from environment map
    vec3 E = texture(ENV_LAMBERTIAN, N_ws).rgb;

    // Prevent compiler removing SKY block
    E += SKY_ENERGY * 0.0;

    // Lambert diffuse = albedo * (irradiance / PI)
    vec3 diffuseIBL = albedo * (E * (1.0 / PI));


    // ===== 5. DYNAMIC LIGHTS  =====
   

    

vec3 directLights = vec3(0.0);

vec3 lambertBRDF = albedo * (1.0 / PI);

int shadowSlot = 0;

for (int i = 0; i < lights.length(); ++i) {
 
    float lightType = lights[i].position.w;

    if (lightType == 0.0) {
        vec3 Lsun = normalize(lights[i].direction.xyz);
        float NdotLsun = max(dot(N_ws, Lsun), 0.0);
        if (NdotLsun <= 0.0) continue;

        float sunStrength = lights[i].params.y;
        directLights += lambertBRDF * lights[i].tint.rgb * sunStrength * NdotLsun;
        continue;
    }

     int myShadowSlot = -1;
    if (lightType == 2.0 && lights[i].tint.w > 0.0 && shadowSlot < MAX_SHADOW_SPOT_LIGHTS) {
        myShadowSlot = shadowSlot;
        shadowSlot += 1;
    }

    if (lightType == 1.0 || lightType == 2.0) {
        vec3 Lvec = lights[i].position.xyz - position;
        float d = length(Lvec);
        vec3 L = Lvec / max(d, 0.0001);

        float spotFactor = 1.0;
        if (lightType == 2.0) {
            vec3 lightDir = normalize(lights[i].direction.xyz);
            vec3 LtoSurface = -L;

            float cosTheta = dot(lightDir, LtoSurface);
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

            if (spotFactor <= 0.0) continue;
        }

        float NdotL = max(dot(N_ws, L), 0.0);
        if (NdotL <= 0.0) continue;

        float physicalFalloff = 1.0 / max(d * d, 0.0001);

        float limit = lights[i].params.z;
        float limitFalloff = 1.0;
        if (limit > 0.0) {
            float x = d / limit;
            limitFalloff = max(0.0, 1.0 - pow(x, 4.0));
        }

        float attenuation = physicalFalloff * limitFalloff;
        float power = lights[i].params.y;

       float shadow = 1.0;
        if (myShadowSlot >= 0) {
            mat4 light_clip_from_world = make_spot_light_matrix(lights[i]);
             shadow = sample_shadow(position, N_ws, L, light_clip_from_world, myShadowSlot, lights[i].params.x, lights[i].params.w);
        }

        directLights += shadow * lambertBRDF * lights[i].tint.rgb * power * NdotL * attenuation * spotFactor;
    }
}

 outColor = vec4(diffuseIBL + directLights, 1.0);
}
 