#version 450
#include "tone_map.glsl"

layout(set=0, binding=0) uniform samplerCube envTex;
layout(location=0) in vec2 v_ndc;
layout(location=0) out vec4 outColor;

layout(push_constant) uniform Push {
    float time;

     // pad time up to 16 bytes so the next thing is aligned nicely
    float _pad0;

    //math
    mat4 inv_view_rot;
    mat4 inv_proj;

    float exposure;
    int tone_op;
    ivec2 _pad1;
    
} pc;

//linear tone map
    

//reinhard tone map compresses highlights into [0..1)
 
void main() {
 // treat tone_op like an integer ID:
    // 0 = linear, 1 = reinhard
    int tone_op = int(pc.tone_op);

    // grab the controls
    float exposure = pc.exposure;

     // reconstruct view direction from NDC
    vec4 view = pc.inv_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 viewDir = normalize(view.xyz / view.w);

     // rotate into world (no translation)
    vec3 worldDir = mat3(pc.inv_view_rot) * viewDir;

     // sample the environment map (this is linear HDR data)
    vec3 col = texture(envTex, worldDir).rgb;  //   linear HDR

    // exposure scaling in stops (multiply by 2^E)
    //col *= pow(2.0, exposure);
    col *= exp2(exposure);

    //tone mapping
    col = apply_tone_map(col, pc.tone_op);

    //old behavor
    //if(tone_op == 0){
    //col = tonemap_linear(col);
    //} else { 
        //col = tonemap_reinhard(col);
    //}

    //output
    outColor = vec4(col, 1.0);
}