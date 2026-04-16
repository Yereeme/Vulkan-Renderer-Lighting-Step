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
layout(set = 5, binding = 0) uniform sampler2D SHADOW_MAP;

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
float sample_shadow(vec3 worldPos, vec3 N, vec3 lightDir)
{
    vec4 lightClip = pc.LIGHT_CLIP_FROM_WORLD * vec4(worldPos, 1.0);
    vec3 lightNDC = lightClip.xyz / max(lightClip.w, 0.0001);

    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;
   float currentDepth = lightNDC.z * 0.5 + 0.5;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth < 0.0 || currentDepth > 1.0) {
        return 1.0;
    }

    float bias = max(0.0005, 0.002 * (1.0 - max(dot(N, lightDir), 0.0)));

    vec2 texel = 1.0 / vec2(textureSize(SHADOW_MAP, 0));
    float sum = 0.0;

    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            vec2 offset = vec2(x - 0.5, y - 0.5) * texel;
            float closestDepth = texture(SHADOW_MAP, shadowUV + offset).r;
            sum += (currentDepth - bias > closestDepth) ? 0.0 : 1.0;
        }
    }

    return sum * 0.25;
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

for (int i = 0; i < lights.length(); ++i) {
 
    float lightType = lights[i].position.w;

    if (lightType == 1.0 || lightType == 2.0) {
        vec3 Lvec = lights[i].position.xyz - position;
        float d = length(Lvec);
        vec3 L = Lvec / max(d, 0.0001);

        float spotFactor = 1.0;
        if (lightType == 2.0) {
            vec3 lightDir = normalize(lights[i].direction.xyz);
            vec3 LtoSurface = -L;

            float cosTheta = dot(lightDir, LtoSurface);
            float cosOuter = cos(lights[i].params.w);
            float blend = lights[i].direction.w;
            float cosInner = mix(cosOuter, 1.0, 1.0 - blend);

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
if (lightType == 2.0 && i == pc.SHADOW_LIGHT_INDEX) {
    shadow = sample_shadow(position, N_ws, L);
}

        directLights += shadow * albedo * lights[i].tint.rgb * power * NdotL * attenuation * spotFactor;
    }
}

//outColor = vec4(0.08 * albedo + directLights, 1.0);
vec4 lightClip = pc.LIGHT_CLIP_FROM_WORLD * vec4(position, 1.0);
vec3 lightNDC = lightClip.xyz / max(lightClip.w, 0.0001);
vec2 shadowUV = vec2(
    lightNDC.x * 0.5 + 0.5,
    1.0 - (lightNDC.y * 0.5 + 0.5)
);
float z01 = lightNDC.z * 0.5 + 0.5;

bool inside =
    shadowUV.x >= 0.0 && shadowUV.x <= 1.0 &&
    shadowUV.y >= 0.0 && shadowUV.y <= 1.0 &&
    z01 >= 0.0 && z01 <= 1.0;

if (!inside) {
    outColor = vec4(1.0, 0.0, 0.0, 1.0); // red = outside shadow map
} else {
    float d = texture(SHADOW_MAP, shadowUV).r;
    outColor = vec4(vec3(d), 1.0);       // grayscale depth only for valid samples
}
}