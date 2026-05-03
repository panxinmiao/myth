{$ include 'core/full_screen_vertex' $}

@group(0) @binding(0)
var lit_tex: texture_2d<f32>;
@group(0) @binding(1)
var lit_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(lit_tex, lit_sampler, in.uv, 0.0);
}