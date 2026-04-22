// Gaussian Splatting Composite Shader
//
// Resolves the isolated Gaussian accumulation buffer back into Myth's linear
// HDR scene color. When the accumulation buffer stores standard 3DGS sRGB
// values, the shader first unpremultiplies by alpha, converts the color to
// linear space, and then premultiplies again before blending over the scene.

{$ include 'core/full_screen_vertex' $}

struct CompositeSettings {
    flags: vec4<u32>,
};

@group(0) @binding(0)
var accumulation_tex: texture_2d<f32>;
@group(0) @binding(1)
var accumulation_sampler: sampler;
@group(0) @binding(2)
var<uniform> settings: CompositeSettings;

fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let accumulation = textureSampleLevel(accumulation_tex, accumulation_sampler, in.uv, 0.0);
    let alpha = clamp(accumulation.a, 0.0, 1.0);

    if alpha <= 1e-6 {
        return vec4<f32>(0.0);
    }

    var premultiplied_linear = accumulation.rgb;
    if settings.flags.x != 0u {
        let srgb = clamp(accumulation.rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
        premultiplied_linear = srgb_to_linear(srgb) * alpha;
    }

    return vec4<f32>(premultiplied_linear, alpha);
}