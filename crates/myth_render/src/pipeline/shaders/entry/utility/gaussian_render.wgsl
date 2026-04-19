// Gaussian Splatting Render Shader
//
// Draws screen-space 2D Gaussian splats with standard back-to-front
// premultiplied-alpha compositing, while preserving Myth's reverse-Z
// depth testing against opaque geometry.

const CUTOFF: f32 = 2.3539888583335364;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct Splat {
    v_0: u32,
    v_1: u32,
    pos: u32,
    depth: f32,
    color_0: u32,
    color_1: u32,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    let sorted_idx = sort_indices[instance_idx];
    let splat = splats[sorted_idx];

    let axis_0 = unpack2x16float(splat.v_0);
    let axis_1 = unpack2x16float(splat.v_1);
    let center_ndc = unpack2x16float(splat.pos);
    let color_rg = unpack2x16float(splat.color_0);
    let color_ba = unpack2x16float(splat.color_1);

    let x = f32(vertex_idx % 2u == 0u) * 2.0 - 1.0;
    let y = f32(vertex_idx < 2u) * 2.0 - 1.0;
    let local_pos = vec2<f32>(x, y) * CUTOFF;
    let delta_ndc = 2.0 * mat2x2<f32>(axis_0, axis_1) * local_pos;

    var out: VertexOutput;
    out.position = vec4<f32>(center_ndc + delta_ndc, splat.depth, 1.0);
    out.local_pos = local_pos;
    out.color = vec4<f32>(color_rg, color_ba);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius_sq = dot(in.local_pos, in.local_pos);
    if radius_sq > 2.0 * CUTOFF {
        discard;
    }

    let alpha = min(0.99, exp(-radius_sq) * in.color.a);
    if alpha < 1.0 / 255.0 {
        discard;
    }

    return vec4<f32>(in.color.rgb, 1.0) * alpha;
}
