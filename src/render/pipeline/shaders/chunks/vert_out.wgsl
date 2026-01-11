struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    $$ endif
};