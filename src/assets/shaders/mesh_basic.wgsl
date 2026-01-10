struct GlobalFrameUniforms {
    view_projection: mat4x4<f32>,
    view_projection_inverse: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> u_global: GlobalFrameUniforms;

{{ binding_code }}      

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>, // 如果有 UV
};

$$ if SHADER_STAGE == "VERTEX"
{{ vertex_struct_code }} 
// ========================================================================
// 2. 顶点着色器
// ========================================================================

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // 使用 Group 2 (Object) 和 Group 0 (Global)
    let model_matrix = u_model.model_matrix; 
    let world_pos = model_matrix * vec4<f32>(in.position, 1.0);
    
    out.clip_position = u_global.view_projection * world_pos;
    
    // 处理 UV (如果 geometry 有的话)
    // out.uv = in.uv; 

    return out;
}

$$ endif


$$ if SHADER_STAGE == "FRAGMENT"
// ========================================================================
// 3. 片元着色器
// ========================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 使用 Group 1 (Material)
    // 你的 ResourceBuilder 生成的变量名是 "material"
    
    var final_color = u_material.color;

    // 简单的贴图测试 (如果有贴图宏)
    $$ if use_map
        let tex_color = textureSample(t_map, s_map, in.uv);
        final_color = final_color * tex_color;
    $$ endif

    return final_color;
}
$$ endif