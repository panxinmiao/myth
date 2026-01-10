
{{ binding_code }}      

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

$$ if SHADER_STAGE == "VERTEX"
{{ vertex_struct_code }} 

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let model_matrix = u_model.model_matrix; 
    let world_pos = model_matrix * vec4<f32>(in.position, 1.0);
    
    out.position = u_global.view_projection * world_pos;
    
    $$ if has_uv
    out.uv = in.uv;
    $$ endif 

    let normal_matrix = u_model.normal_matrix;

    $$ if has_normal
    out.normal = normal_matrix * in.normal;
    $$ endif

    return out;
}

$$ endif

$$ if SHADER_STAGE == "FRAGMENT"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = u_material.color;

    $$ if use_map
        let tex_color = textureSample(t_map, s_map, in.uv);
        final_color = final_color * tex_color;
    $$ endif

    return final_color;
}
$$ endif