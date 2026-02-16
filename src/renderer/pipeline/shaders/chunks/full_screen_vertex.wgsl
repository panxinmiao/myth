// --- Vertex output ---
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// --- Vertex Shader: Fullscreen Triangle ---
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var output : VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.uv = pos[vertex_index] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}