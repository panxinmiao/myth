
{{ vertex_input_code }} 

{{ binding_code }}      

{$ include 'morph' $}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    @location({{ loc.next() }}) geometry_normal: vec3<f32>,
    $$ endif
    $$ if use_vertex_color
    @location({{ loc.next() }}) color: vec4<f32>,
    $$ endif
    {$ include 'uv_vetex_output' $}
};


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_pos = vec4<f32>(in.position, 1.0);
    var local_normal = in.normal;

    $$ if use_morphing
    // 应用 Morph Target 变形
    let morph_result = apply_morph_targets(vertex_index, in.position, in.normal);
    local_pos = vec4<f32>(morph_result.position, 1.0);
    local_normal = morph_result.normal;
    $$ endif

    {$ include 'skin' $}

    let world_pos = u_model.world_matrix * local_pos;


    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if use_vertex_color
        out.color = in.color;
    $$ endif

    out.uv = in.uv;
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    {$ include 'uv' $}
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