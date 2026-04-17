// Gaussian Splatting — Final Rendering Shader
//
// Vertex-pulled quad rendering of 2D Gaussian splats. Each splat is
// rendered as a screen-aligned quad; the fragment shader evaluates the
// Gaussian kernel and applies front-to-back alpha blending.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic_and_opacity: vec4<f32>,
    @location(2) center: vec2<f32>,
};

struct Splat {
    v_0: u32,
    v_1: u32,
    pos: u32,
    color_0: u32,
    color_1: u32,
};

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;

// Front-to-back quad rendering
// Each quad consists of 4 vertices. The sort_indices array
// contains the order to render in (sorted by depth).
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    // Each quad: 4 vertices (triangle strip)
    let quad_idx = vertex_idx / 4u;
    let vert_in_quad = vertex_idx % 4u;

    let sorted_idx = sort_indices[quad_idx];
    let splat = splats[sorted_idx];

    let v0 = unpack2x16float(splat.v_0);
    let v1 = unpack2x16float(splat.v_1);
    let center_ndc = unpack2x16float(splat.pos);
    let color_rg = unpack2x16float(splat.color_0);
    let color_ba = unpack2x16float(splat.color_1);

    // Quad offsets: [-1,-1], [1,-1], [-1,1], [1,1]
    let offset = vec2<f32>(
        select(-1.0, 1.0, (vert_in_quad & 1u) != 0u),
        select(-1.0, 1.0, (vert_in_quad & 2u) != 0u),
    );

    let pos2d = center_ndc + v0 * offset.x + v1 * offset.y;

    var out: VertexOutput;
    out.position = vec4<f32>(pos2d, 0.0, 1.0);
    out.color = vec4<f32>(color_rg, color_ba);

    // Compute conic from v0, v1 for the Gaussian kernel
    let det = v0.x * v1.y - v0.y * v1.x;
    let inv_det = 1.0 / (det + 1e-10);
    out.conic_and_opacity = vec4<f32>(
        v1.y * inv_det,
        -v0.y * inv_det,
        v0.x * inv_det,
        color_ba.y
    );
    out.center = center_ndc;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let offset = (in.position.xy / vec2<f32>(camera.viewport)) * 2.0 - 1.0 - in.center;
    let conic = vec3<f32>(in.conic_and_opacity.xyz);

    let power = -0.5 * (conic.x * offset.x * offset.x + 2.0 * conic.y * offset.x * offset.y + conic.z * offset.y * offset.y);

    if power > 0.0 {
        discard;
    }

    let alpha = min(0.99, in.color.a * exp(power));
    if alpha < 1.0 / 255.0 {
        discard;
    }

    // Front-to-back blending: pre-multiply
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
