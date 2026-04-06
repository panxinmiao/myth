// Bloom Upsample Pass (3×3 tent filter with additive blending)
//
// Performs progressive upsampling with a 3×3 tent (bilinear) filter.
// Output is additively blended with the target mip via hardware blend state,
// accumulating bloom contributions from coarser mip levels.

{$ include 'core/full_screen_vertex' $}

{{ struct_definitions }}

// Group 0: Persistent feature resources (Feature-owned, long-lived)
@group(0) @binding(0) var src_sampler: sampler;
@group(0) @binding(1) var<uniform> u_bloom: UpsampleUniforms;

// Group 1: Transient RDG textures (PassNode-owned, per-frame)
@group(1) @binding(0) var src_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let x = u_bloom.filter_radius;
    let y = u_bloom.filter_radius;

    // 3×3 tent filter:
    //  1   | 1 2 1 |
    // -- × | 2 4 2 |
    // 16   | 1 2 1 |

    let a = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-x,  y), 0.0).rgb;
    let b = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.0, y), 0.0).rgb;
    let c = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( x,  y), 0.0).rgb;

    let d = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-x, 0.0), 0.0).rgb;
    let e = textureSampleLevel(src_texture, src_sampler, uv, 0.0).rgb;
    let f = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( x, 0.0), 0.0).rgb;

    let g = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>(-x, -y), 0.0).rgb;
    let h = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( 0.0, -y), 0.0).rgb;
    let i = textureSampleLevel(src_texture, src_sampler, uv + vec2<f32>( x, -y), 0.0).rgb;

    var result = e * 4.0;
    result += (b + d + f + h) * 2.0;
    result += (a + c + g + i);
    result *= (1.0 / 16.0);

    result = max(result, vec3<f32>(0.0));
    return vec4<f32>(result, 1.0);
}
