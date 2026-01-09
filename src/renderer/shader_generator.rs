use crate::core::material::Material;
use crate::renderer::vertex_layout::GeneratedVertexLayout;
use serde_json::{Map, Value};
use super::shader_manager::get_env;
use crate::core::material::MaterialFeatures;
use crate::core::geometry::{Geometry, GeometryFeatures};
use crate::core::scene::SceneFeatures;
use crate::core::binding::ResourceBuilder;
use crate::core::uniforms::DynamicModelUniforms;
use crate::core::buffer::{DataBuffer, BufferRef};
use wgpu::ShaderStages;


#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShaderCompilationOptions {
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    pub scene_features: SceneFeatures,
    pub num_dir_lights: u8,
    pub num_point_lights: u8,
    pub num_spot_lights: u8,
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


        map.insert("num_dir_lights".into(), self.num_dir_lights.into());
        map.insert("num_point_lights".into(), self.num_point_lights.into());
        map.insert("num_spot_lights".into(), self.num_spot_lights.into());

        // [新增] 处理 Scene Features (示例)
        // if self.scene_features.contains(SceneFeatures::USE_SHADOW_MAP) { map.insert("use_shadow_map".into(), true.into()); }
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
    // 核心辅助：根据 ResourceBuilder 生成 WGSL 绑定代码
    // ========================================================================
    fn generate_wgsl_bindings(builder: &ResourceBuilder, group_index: u32) -> String {
        let mut code = String::new();

        for (i, entry) in builder.layout_entries.iter().enumerate() {
            let name = &builder.names[i];
            let binding_index = entry.binding; // 使用 builder 中真实的 binding index

            let decl = match entry.ty {
                wgpu::BindingType::Buffer { ty, .. } => {
                    match ty {
                        wgpu::BufferBindingType::Uniform => {
                            // 约定：Uniform 变量名直接使用 name
                            // 结构体类型名通常在外部定义，这里我们假设 name 就是结构体实例名
                            // 对于 Material，通常结构体叫 MaterialUniforms，变量名由 builder 传入 "material"
                            // 对于 DynamicModel，结构体叫 DynamicModel，变量名 "mesh_model"
                            
                            // 这里做一个简单的启发式推断或约定：
                            // 如果是 Uniform，我们假设 WGSL 中已经定义了对应的大驼峰结构体，
                            // 或者我们需要 builder 提供类型名。
                            // 目前 ResourceBuilder.names 存的是变量名。
                            // 为了通用，我们假设 WGSL 类型名可以通过变量名推导（例如 "material" -> "MaterialUniforms"）
                            // 或者简单一点：让外部在 define_bindings 时就生成好结构体定义，
                            // 这里只生成 var<uniform> 声明，类型需要稍微处理一下。
                            
                            // 临时方案：针对已知变量名做特殊处理，或者扩展 ResourceBuilder 存类型名。
                            // 为了不改动 ResourceBuilder，我们这里用简单的映射：
                            let type_name = if name == "material" {
                                "MaterialUniforms"
                            } else if name == "mesh_model" {
                                "DynamicModel" 
                            } else {
                                // 默认尝试大驼峰转化，或者直接用 Name (如果 Name 本身就是类型)
                                // 这里假设 Name 是变量名，我们无法知道类型名。
                                // 如果是自定义 Storage Buffer，通常是 array<f32> 或 struct array。
                                "UnknownType" // 这会报错，但在我们当前的受控环境中足够了
                            };
                            
                            format!("@group({}) @binding({}) var<uniform> {}: {};", group_index, binding_index, name, type_name)
                        },
                        wgpu::BufferBindingType::Storage { read_only } => {
                            let access = if read_only { "read" } else { "read_write" };
                            // Storage Buffer 通常是 array<f32> 或者 array<Struct>
                            // 对于 Morph Target，通常是 array<f32>
                            // 对于 Skinning Matrices，是 array<mat4x4<f32>>
                            let type_str = if name == "morph_data" {
                                "array<f32>" // Morph Targets 数据通常是 float 流
                            } else if name.contains("bone") {
                                "array<mat4x4<f32>>"
                            } else {
                                "array<f32>" // 默认 fallback
                            };
                            format!("@group({}) @binding({}) var<storage, {}> {}: {};", group_index, binding_index, access, name, type_str)
                        },
                    }
                },
                wgpu::BindingType::Texture { sample_type, view_dimension, .. } => {
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { .. }) => "texture_2d<f32>",
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => "texture_depth_2d",
                        (wgpu::TextureViewDimension::Cube, wgpu::TextureSampleType::Float { .. }) => "texture_cube<f32>",
                        _ => "texture_2d<f32>",
                    };
                    // 纹理变量名约定：t_{name}
                    format!("@group({}) @binding({}) var t_{}: {};", group_index, binding_index, name, type_str)
                },
                wgpu::BindingType::Sampler(type_) => {
                    let type_str = match type_ {
                        wgpu::SamplerBindingType::Comparison => "sampler_comparison",
                        _ => "sampler",
                    };
                    // 采样器变量名约定：s_{name}
                    format!("@group({}) @binding({}) var s_{}: {};", group_index, binding_index, name, type_str)
                },
                _ => String::new(),
            };

            code.push_str(&decl);
            code.push('\n');
        }
        code
    }


    // ========================================================================
    // 1. 顶点着色器生成器
    // ========================================================================
    pub fn generate_vertex(
        global_context: &ShaderContext,
        geometry_layout: &GeneratedVertexLayout,
        geometry: &Geometry,
        template_name: &str,
    ) -> String {

        let env = get_env();
        let mut context = global_context.defines.clone();

        // 注入 Vertex Buffer 结构定义
        context.insert("vertex_struct_code".to_string(), Value::String(geometry_layout.vertex_input_code.clone()));

        // 2. Group 2 Bindings (Object Level) - 自动化生成
        let mut builder = ResourceBuilder::new();

        // 2.1 手动添加 Model Matrix (Binding 0)
        // 这是一个 Hack: 为了满足 builder.add_dynamic_uniform 需要 BufferRef 的 API，
        // 我们创建一个临时的 dummy buffer。它不会被真正使用，因为我们只用 builder 生成 layout 代码。
        let dummy_buffer = BufferRef::new(DataBuffer::new(
            &[0u8; 256], // 任意数据
            wgpu::BufferUsages::UNIFORM,
            Some("DummyGeneratorBuffer")
        ));
        
        let model_size = std::mem::size_of::<DynamicModelUniforms>() as u64;
        builder.add_dynamic_uniform("mesh_model", &dummy_buffer, model_size, ShaderStages::VERTEX);
        // 2.2 添加 Geometry 资源 (Binding 1..N)
        geometry.define_bindings(&mut builder);

        // 2.3 生成 WGSL
        // 首先需要注入 DynamicModel 的结构体定义
        let mut binding_code = String::new();
        binding_code.push_str(&DynamicModelUniforms::wgsl_struct_def("DynamicModel"));
        binding_code.push_str("\n");

        // 生成 var 定义
        binding_code.push_str(&Self::generate_wgsl_bindings(&builder, 2));

        context.insert("binding_code".to_string(), Value::String(binding_code));


        // 3. 渲染模版
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
        // let bindings_layout = material.get_layout();
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        let binding_code = Self::generate_wgsl_bindings(&builder, 1);

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