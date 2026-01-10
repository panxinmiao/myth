use crate::renderer::vertex_layout::GeneratedVertexLayout;
use serde_json::{Map, Value};
use super::shader_manager::get_env;
use crate::core::material::MaterialFeatures;
use crate::core::geometry::{GeometryFeatures};
use crate::core::scene::SceneFeatures;


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

        if self.geo_features.contains(GeometryFeatures::HAS_UV) { map.insert("has_uv".into(), true.into()); }
        if self.geo_features.contains(GeometryFeatures::HAS_NORMAL) { map.insert("has_normal".into(), true.into()); }
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

    // pub fn define(mut self, name: &str, value: bool) -> Self {
    //     self.defines.insert(name.to_string(), Value::Bool(value));
    //     self
    // }

    pub fn set_value(mut self, name: &str, value: impl Into<Value>) -> Self {
        self.defines.insert(name.to_string(), value.into());
        self
    }

    pub fn _merge(&mut self, other: &ShaderContext) {
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
        global_binding_code: &str,
        object_binding_code: &str,
        template_name: &str,
    ) -> String {

        let env = get_env();
        let mut context = global_context.defines.clone();

        context.insert("SHADER_STAGE".to_string(), Value::String("VERTEX".to_string()));

        // 注入 Vertex Buffer 结构定义
        context.insert("vertex_struct_code".to_string(), Value::String(geometry_layout.vertex_input_code.clone()));


        // 2.3 生成 WGSL
        // 首先需要注入 DynamicModel 的结构体定义
        let mut binding_code = String::new();
        binding_code.push_str("\n");

        // var 定义
        binding_code.push_str(global_binding_code);
        binding_code.push_str("\n");
        binding_code.push_str(object_binding_code);
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
        global_binding_code: &str,
        material_binding_code: &str,
        template_name: &str,
    ) -> String {
    
        let env = get_env();
        let mut context = global_context.defines.clone();

        context.insert("SHADER_STAGE".to_string(), Value::String("FRAGMENT".to_string()));

        // 2. Bindings 代码
        let mut binding_code = String::new();
        binding_code.push_str("\n");

        // var 定义
        binding_code.push_str(global_binding_code);
        binding_code.push_str("\n");
        binding_code.push_str(material_binding_code);
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