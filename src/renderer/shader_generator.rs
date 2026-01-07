use crate::core::material::Material;
use crate::core::binding::{Bindable, BindingType};
use crate::renderer::vertex_layout::GeneratedVertexLayout;
use serde_json::{Map, Value};
use super::shader_manager::get_env;

/// Shader 上下文构建器
pub struct ShaderContext {
    pub defines: Map<String, Value>,
}

impl ShaderContext {
    pub fn new() -> Self {
        Self {
            defines: Map::new(),
        }
    }

    pub fn define(mut self, name: &str, value: bool) -> Self {
        self.defines.insert(name.to_string(), Value::Bool(value));
        self
    }

    pub fn set_value(mut self, name: &str, value: impl Into<Value>) -> Self {
        self.defines.insert(name.to_string(), value.into());
        self
    }

    pub fn merge(&mut self, other: &ShaderContext) {
        for (k, v) in &other.defines {
            self.defines.insert(k.clone(), v.clone());
        }
    }
}

pub struct ShaderGenerator;

impl ShaderGenerator {

    // ========================================================================
    // 1. 顶点着色器生成器
    // ========================================================================
    pub fn generate_vertex(
        global_context: &ShaderContext,
        geometry_layout: &GeneratedVertexLayout,
        template_name: &str,
    ) -> String {

        let env = get_env();
        let mut context = global_context.defines.clone();

        // 注入 Vertex Buffer 结构定义
        context.insert("vertex_struct_code".to_string(), Value::String(geometry_layout.vertex_input_code.clone()));

        // Group 2: Model Matrix Binding (目前这部分还没有 Bindable 化，暂时硬编码，后续也可重构)
        let binding_code = r#"
            struct ModelUniforms {
                model_matrix: mat4x4<f32>,
                model_matrix_inverse: mat4x4<f32>,
                normal_matrix: mat3x3<f32>,
            };
            @group(2) @binding(0) var<storage, read> mesh_models: array<ModelUniforms>;
        "#;
        context.insert("binding_code".to_string(), Value::String(binding_code.to_string()));

        let vs_template = env.get_template(template_name)
            .unwrap_or_else(|e| panic!("Vertex template '{}' not found: {}", template_name, e));
        let vs_code = vs_template.render(&context)
            .unwrap_or_else(|e| panic!("Failed to render vertex shader: {}", e));

        format!(
            "// === Auto-generated Vertex Shader ===\n\n{}\n",
            vs_code
        )
    }

    // ========================================================================
    // 2. 片元着色器生成器 (完全动态化)
    // ========================================================================
    pub fn generate_fragment(
        global_context: &ShaderContext,
        material: &Material,
        template_name: &str,
    ) -> String {
    
        let env = get_env();
        let mut context = global_context.defines.clone();

        // 1. 注入 Material Uniform 结构体定义 (来自 Material 自描述)
        context.insert(
            "uniform_struct_code".to_string(), 
            Value::String(material.wgsl_struct_def().to_string())
        );

        // 2. 动态生成 Bindings 代码
        // 我们遍历 Material 提供的 Layout，生成 WGSL
        let (bindings, _) = material.get_bindings();
        let mut binding_code = String::new();

        for desc in bindings {
            // A. 设置宏 (Defines)
            // 如果 Binding 名字叫 "map"，我们就定义 "use_map = true"
            // 这样 Shader 模板里 {% if use_map %} 就能正常工作
            // 注意：因为 Material 可能同时返回 "map" 的 texture 和 sampler，这里会重复设置，但这没关系
            context.insert(format!("use_{}", desc.name), Value::Bool(true));

            // B. 生成 WGSL 变量声明
            // @group(1) @binding(N) var prefix_name: type;
            let wgsl = match desc.bind_type {
                BindingType::UniformBuffer => {
                    // 假设 UniformBuffer 绑定的结构体名字就是 desc.name (例如 "MaterialUniforms")
                    // 变量名我们固定叫 material，或者也可以根据 desc.name 生成
                    format!("@group(1) @binding({}) var<uniform> material: {};\n", desc.index, desc.name)
                },
                BindingType::Texture { sample_type, view_dimension, .. } => {
                    // 根据 Texture 类型生成对应的 WGSL 类型
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { .. }) => "texture_2d<f32>",
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => "texture_depth_2d",
                        // ... 其他类型可按需添加
                        _ => "texture_2d<f32>", 
                    };
                    // 约定：纹理变量名为 t_{name}
                    format!("@group(1) @binding({}) var t_{}: {};\n", desc.index, desc.name, type_str)
                },
                BindingType::Sampler { type_ } => {
                    let type_str = match type_ {
                        wgpu::SamplerBindingType::Filtering => "sampler",
                        wgpu::SamplerBindingType::NonFiltering => "sampler",
                        wgpu::SamplerBindingType::Comparison => "sampler_comparison",
                    };
                    // 约定：采样器变量名为 s_{name}
                    format!("@group(1) @binding({}) var s_{}: {};\n", desc.index, desc.name, type_str)
                },
                BindingType::StorageBuffer { read_only } => {
                    let access = if read_only { "read" } else { "read_write" };
                    format!("@group(1) @binding({}) var<storage, {}> {}: array<f32>;\n", desc.index, access, desc.name)
                }
            };
            
            binding_code.push_str(&wgsl);
        }

        context.insert("binding_code".to_string(), Value::String(binding_code));

        // 3. 渲染模板
        let fs_template = env.get_template(template_name)
            .unwrap_or_else(|e| panic!("Fragment template '{}' not found: {}", template_name, e));
        let fs_code = fs_template.render(&context)
            .unwrap_or_else(|e| panic!("Failed to render fragment shader: {}", e));

        format!(
            "// === Auto-generated Fragment Shader ===\n\n{}\n", 
            fs_code
        )
    }
}