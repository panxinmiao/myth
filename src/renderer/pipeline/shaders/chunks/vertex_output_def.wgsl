struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
    $$ if HAS_UV
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if HAS_NORMAL
    @location({{ loc.next() }}) normal: vec3<f32>,
    @location({{ loc.next() }}) geometry_normal: vec3<f32>,
    $$ if HAS_TANGENT
    @location({{ loc.next() }}) v_tangent: vec3<f32>,
    @location({{ loc.next() }}) v_bitangent: vec3<f32>,
    $$ endif
    $$ endif
    $$ if HAS_VERTEX_COLOR
    @location({{ loc.next() }}) color: vec4<f32>,
    $$ endif
    {$ include 'uv_vetex_output' $}
};