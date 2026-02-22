// Bloom Downsample Pass (13-tap filter)
//
// Implements the progressive downsample from "Next Generation Post Processing
// in Call of Duty: Advanced Warfare". The 13-tap sampling pattern minimizes
// aliasing while the optional Karis average on the first mip suppresses fireflies.

{$ include 'full_screen_vertex.wgsl' $}

{{ struct_definitions }}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> u_bloom: DownsampleUniforms;

fn rgb_to_luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn karis_weight(color: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + rgb_to_luminance(color));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // 13-tap sampling pattern (bilinear-friendly offsets)
    //
    // a - b - c
    // - j - k -
    // d - e - f
    // - l - m -
    // g - h - i
    //
    // 'e' is the center texel.

    let a = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(-2, 2)).rgb;
    let b = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(0, 2)).rgb;
    let c = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(2, 2)).rgb;

    let d = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(-2, 0)).rgb;
    let e = textureSampleLevel(src_texture, src_sampler, uv, 0.0).rgb;
    let f = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(2, 0)).rgb;

    let g = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(-2, -2)).rgb;
    let h = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(0, -2)).rgb;
    let i = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(2, -2)).rgb;

    let j = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(-1, 1)).rgb;
    let k = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(1, 1)).rgb;
    let l = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(-1, -1)).rgb;
    let m = textureSampleLevel(src_texture, src_sampler, uv, 0.0, vec2i(1, -1)).rgb;

    var result: vec3<f32>;

    if (u_bloom.use_karis_average != 0u) {
        // First downsample: use Karis average to suppress firefly artifacts.
        // Five 2×2 sample groups with luminance-weighted averaging.
        let g0 = (a + b + d + e) * 0.25;
        let g1 = (b + c + e + f) * 0.25;
        let g2 = (d + e + g + h) * 0.25;
        let g3 = (e + f + h + i) * 0.25;
        let g4 = (j + k + l + m) * 0.25;

        let w0 = karis_weight(g0);
        let w1 = karis_weight(g1);
        let w2 = karis_weight(g2);
        let w3 = karis_weight(g3);
        let w4 = karis_weight(g4);

        let w_sum = w0 + w1 + w2 + w3 + w4;
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / max(w_sum, 0.0001);
    } else {
        // Standard weighted distribution (energy-preserving):
        // 0.125 × 5 + 0.03125 × 4 + 0.0625 × 4 = 1.0
        result = e * 0.125
               + (a + c + g + i) * 0.03125
               + (b + d + f + h) * 0.0625
               + (j + k + l + m) * 0.125;
    }

    result = max(result, vec3<f32>(0.0));
    return vec4<f32>(result, 1.0);
}
