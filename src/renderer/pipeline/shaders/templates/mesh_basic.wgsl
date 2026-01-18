{{ vertex_input_code }} 
{{ binding_code }}      
{$ include 'vertex_output_def' $}



{$ include 'morph_pars' $}


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_pos = vec4<f32>(in.position, 1.0);
    var local_normal = in.normal;

    {$ include 'morph_vertex' $}
    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if use_vertex_color
        out.color = in.color;
    $$ endif

    $$ if has_uv
    out.uv = in.uv;
    $$ endif

    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    {$ include 'uv_vertex' $}
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = u_material.color;
    {$ if use_map $}
    let tex_color = textureSample(t_map, s_map, in.uv);
    final_color = final_color * tex_color;
    {$ endif $}
    return final_color;
}