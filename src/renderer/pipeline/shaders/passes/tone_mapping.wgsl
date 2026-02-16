{$ include 'full_screen_vertex.wgsl' $}

{$ include 'tone_mapping_pars' $}

struct Uniforms{
    exposure: f32,
};

// bindings
@group(0) @binding(0)
var colorTex: texture_2d<f32>;
@group(0) @binding(1)
var texSampler: sampler;
@group(0) @binding(2)
var<uniform> u_effect: Uniforms;


@fragment
fn fs_main(varyings: VertexOutput) -> @location(0) vec4<f32> {
    let texCoord = varyings.uv;
    var color = textureSample(colorTex, texSampler, texCoord);

    // Apply tone mapping to RGB channels
    return vec4<f32>(toneMapping(color.rgb * u_effect.exposure), color.a);
}