struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, 1.0);
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(2.0, 1.0);
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, -1.0);
    }
    return out;

}

{$ include 'tone_mapping_pars' $}

struct Unifromss{
    exposure: f32,
};

// bindings
@group(0) @binding(0)
var colorTex: texture_2d<f32>;
@group(0) @binding(1)
var texSampler: sampler;
@group(0) @binding(2)
var<uniform> u_effect: Unifromss;


@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texCoord = varyings.uv;
    var color = textureSample(colorTex, texSampler, texCoord);

    // Apply tone mapping to RGB channels
    return vec4<f32>(toneMapping(color.rgb * u_effect.exposure), color.a);
}