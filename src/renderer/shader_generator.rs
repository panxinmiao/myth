use crate::core::material::Material;
use crate::core::binding::{BindingType};
use crate::renderer::vertex_layout::GeneratedVertexLayout;
use serde_json::{Map, Value};
use super::shader_manager::get_env;
use crate::core::material::MaterialFeatures;
use crate::core::geometry::GeometryFeatures;


#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShaderCompilationOptions {
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    // 将来可以扩展：
    // pub alpha_test: OrderedFloat<f32>, 
    // pub use_instancing: bool,
}

impl ShaderCompilationOptions {
    /// 转换为 minijinja/Handlebars 可用的 Context Map
    pub fn to_defines(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        // 自动展开 features
        if self.mat_features.contains(MaterialFeatures::USE_MAP) { map.insert("use_map".into(), true.into()); }
        if self.mat_features.contains(MaterialFeatures::USE_NORMAL_MAP) { map.insert("use_normal_map".into(), true.into()); }
        if self.mat_features.contains(MaterialFeatures::USE_ROUGHNESS_MAP) { map.insert("use_roughness_map".into(), true.into()); }
        if self.mat_features.contains(MaterialFeatures::USE_METALNESS_MAP) { map.insert("use_metalness_map".into(), true.into()); }
        if self.mat_features.contains(MaterialFeatures::USE_EMISSIVE_MAP) { map.insert("use_emissive_map".into(), true.into()); }
        if self.mat_features.contains(MaterialFeatures::USE_AO_MAP) { map.insert("use_ao_map".into(), true.into()); }

        if self.geo_features.contains(GeometryFeatures::USE_VERTEX_COLOR) { map.insert("use_vertex_color".into(), true.into()); }
        if self.geo_features.contains(GeometryFeatures::USE_TANGENT) { map.insert("use_tangent".into(), true.into()); }
        if self.geo_features.contains(GeometryFeatures::USE_MORPHING) { map.insert("use_morphing".into(), true.into()); }
        if self.geo_features.contains(GeometryFeatures::USE_SKINNING) { map.insert("use_skinning".into(), true.into()); }
        map
    }
}

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

        // // 2. 注入 Defines (使用新系统！)
        // let defines_map = material.get_defines().to_defines();
        // for (k, v) in defines_map {
        //     context.insert(k, v);
        // }

        // 2. 动态生成 Bindings 代码
        
        let bindings_layout = material.get_layout();

        let mut binding_code = String::new();

        for desc in bindings_layout.iter() {
            // context.insert(format!("use_{}", desc.name), Value::Bool(true));

            // B. 生成 WGSL 变量声明
            // @group(1) @binding(N) var prefix_name: type;
            let wgsl = match desc.bind_type {
                BindingType::UniformBuffer{..} => {
                    // 假设 UniformBuffer 绑定的结构体名字就是 desc.name (例如 "MaterialUniforms")
                    // 变量名我们固定叫 material，或者也可以根据 desc.name 生成
                    format!("@group(1) @binding({}) var<uniform> material: {};\n", desc.index, desc.name)
                },
                BindingType::Texture { sample_type, view_dimension, .. } => {
                    // 根据 Texture 类型生成对应的 WGSL 类型
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { .. }) => "texture_2d<f32>",
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => "texture_depth_2d",
                        (wgpu::TextureViewDimension::Cube, wgpu::TextureSampleType::Float { .. }) => "texture_cube<f32>",
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