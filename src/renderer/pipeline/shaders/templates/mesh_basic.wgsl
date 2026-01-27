{{ vertex_input_code }} 
{{ binding_code }}      
{$ include 'vertex_output_def' $}



{$ include 'morph_pars' $}


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = in.position;

    $$ if HAS_NORMAL
    var local_normal = in.normal;
    $$ endif

    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);
    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_VERTEX_COLOR
        out.color = in.color;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    $$ if HAS_NORMAL
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    {$ include 'uv_vertex' $}
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var diffuse_color = u_material.color;
    {$ if HAS_MAP $}
    let tex_color = textureSample(t_map, s_map, in.uv);
    diffuse_color = diffuse_color * tex_color;
    {$ endif $}

    {$ include 'alpha_test' $}
    return diffuse_color;
}