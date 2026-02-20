// Bloom Composite Pass
//
// Blends the accumulated bloom result (from mip 0 of the bloom chain)
// with the original HDR scene color using linear interpolation.
// Output replaces the scene color buffer for subsequent passes (e.g., tone mapping).

{$ include 'full_screen_vertex.wgsl' $}

struct CompositeUniforms {
    bloom_strength: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var original_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> u_bloom: CompositeUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let original = textureSampleLevel(original_texture, tex_sampler, uv, 0.0);
    let bloom = textureSampleLevel(bloom_texture, tex_sampler, uv, 0.0);

    // Additive blend: original + bloom × strength
    let result = original.rgb + bloom.rgb * u_bloom.bloom_strength;

    // The original LearnOpenGL text(https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom) uses "mix", 
    // but I believe "additive" seems to be more appropriate. 
    // TODO: confirm the correct approach.
    // let result = mix(original.rgb, bloom.rgb, u_bloom.bloom_strength);

    return vec4<f32>(result, original.a);
}
