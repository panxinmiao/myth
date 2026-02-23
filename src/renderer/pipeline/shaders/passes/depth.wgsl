{{ vertex_input_code }}
{{ binding_code }}
{$ include 'morph_pars' $}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    $$ if HAS_UV
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
};

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = in.position;
    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);
    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;
    out.position = u_shadow_light.view_projection * world_pos;

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    return out;
}

@fragment
fn fs_main(varyings: VertexOutput) {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    {$ include 'alpha_test' $}
}
