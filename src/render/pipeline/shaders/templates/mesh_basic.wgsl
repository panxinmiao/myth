
{{ vertex_input_code }} 

{{ binding_code }}      

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    $$ endif
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model_matrix = u_model.world_matrix; 
    let world_pos = model_matrix * vec4<f32>(in.position, 1.0);
    
    out.position = u_render_state.view_projection * world_pos;
    $$ if has_uv
        out.uv = in.uv;
    $$ endif 
    $$ if has_normal
        let normal_matrix = u_model.normal_matrix;
        out.normal = normal_matrix * in.normal;
    $$ endif
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