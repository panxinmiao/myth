struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    @location({{ loc.next() }}) geometry_normal: vec3<f32>,
    $$ if use_tangent is defined
    @location({{ loc.next() }}) v_tangent: vec3<f32>,
    @location({{ loc.next() }}) v_bitangent: vec3<f32>,
    $$ endif
    $$ endif
    $$ if use_vertex_color
    @location({{ loc.next() }}) color: vec4<f32>,
    $$ endif
    {$ include 'uv_vetex_output' $}
};