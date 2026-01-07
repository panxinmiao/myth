use crate::core::material::{Material, MaterialFeatures};
use crate::renderer::layout_generator::GeneratedLayout;
use serde_json::{Map, Value};
use super::shader_manager::get_env;

// 定义纹理的元数据结构
struct TextureDef {
    feature: MaterialFeatures, // 对应的特性开关
    name: &'static str,        // Shader 中的变量后缀 (如 "map" -> t_map, s_map)
}

// === 核心配置表 ===
// 这里的顺序非常重要！它决定了 BindGroup 中的绑定顺序。
// 任何通过 features 开启的纹理，都会按照这个顺序依次分配 binding slot。
const TEXTURE_DEFINITIONS: &[TextureDef] = &[
    TextureDef { feature: MaterialFeatures::USE_MAP,           name: "map" },
    TextureDef { feature: MaterialFeatures::USE_NORMAL_MAP,    name: "normal_map" },
    TextureDef { feature: MaterialFeatures::USE_ROUGHNESS_MAP, name: "roughness_map" },
    TextureDef { feature: MaterialFeatures::USE_METALNESS_MAP,  name: "metalness_map" },
    TextureDef { feature: MaterialFeatures::USE_EMISSIVE_MAP,  name: "emissive_map" },
    TextureDef { feature: MaterialFeatures::USE_AO_MAP,        name: "ao_map" },
    // 未来添加新贴图只需在这里加一行，例如:
    // TextureDef { feature: MaterialFeatures::USE_AO_MAP, name: "ao" },
];


/// Shader 上下文构建器
/// 用于收集所有的宏定义、代码片段和绑定信息
pub struct ShaderContext {
    pub defines: Map<String, Value>,
}

impl ShaderContext {
    pub fn new() -> Self {
        Self {
            defines: Map::new(),
        }
    }

    /// 定义一个宏开关 (boolean)
    pub fn define(mut self, name: &str, value: bool) -> Self {
        self.defines.insert(name.to_string(), Value::Bool(value));
        self
    }

    /// 定义一个宏值 (int/float/string)
    pub fn set_value(mut self, name: &str, value: impl Into<Value>) -> Self {
        self.defines.insert(name.to_string(), value.into());
        self
    }

    /// 批量合并另一个 Context (用于继承全局设置)
    pub fn merge(&mut self, other: &ShaderContext) {
        for (k, v) in &other.defines {
            self.defines.insert(k.clone(), v.clone());
        }
    }


    // pub fn apply_material_features(mut self, features: MaterialFeatures) -> Self {
    //     // 1. 生成 Texture Bindings
    //     for def in TEXTURE_DEFINITIONS {
    //         if features.contains(def.feature) {
    //             // 定义宏: use_map = true
    //             self.defines.insert(format!("use_{}", def.name), Value::Bool(true));

    //             // 生成 Binding 代码
    //             self.binding_code.push_str(&format!(
    //                 r#"
    //                 @group(1) @binding({}) var t_{}: texture_2d<f32>;
    //                 @group(1) @binding({}) var s_{}: sampler;
    //                 "#,
    //                 self.next_binding_idx,     def.name,
    //                 self.next_binding_idx + 1, def.name
    //             ));
    //             self.next_binding_idx += 2;
    //         }
    //     }
    //     self
    // }
    
    // 可以在这里添加更多处理，比如 Lighting 相关的宏
    // pub fn apply_lighting_features(mut self, ...) -> Self { ... }
}

pub struct ShaderGenerator;

impl ShaderGenerator {

    // ========================================================================
    // 1. 顶点着色器生成器
    // 关注点：几何体布局(Attributes)、模型矩阵(Model)、全局相机(Camera)
    // ========================================================================
    pub fn generate_vertex(
        global_context: &ShaderContext,
        geometry_layout: &GeneratedLayout,
        template_name: &str,
    ) -> String {

        let env = get_env();


        // 1. 克隆 Context，避免污染全局状态
        let mut context = global_context.defines.clone();

        context.insert("vertex_struct_code".to_string(), Value::String(geometry_layout.shader_code.clone()));

        // B. Vertex Bindings (Group 0 & 2)
        // 这里的绑定代码是 Vertex 独有的
        let binding_code = r#"
            // Group 2: Model (Storage Buffer)
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
    // 2. 片元着色器生成器
    // 关注点：材质参数(Material Uniforms)、纹理采样(Textures)、光照计算
    // ========================================================================
    pub fn generate_fragment(
        global_context: &ShaderContext,
        material: &Material,
        template_name: &str,
    ) -> String {
    
        let env = get_env();

        // 1. 克隆 Context
        let mut context = global_context.defines.clone();
        let features = material.features();

        // 2. 注入 Fragment 特有的系统代码
        // A. Material Uniform 结构体
        context.insert(
            "uniform_struct_code".to_string(), 
            Value::String(material.wgsl_struct_def().to_string())
        );

        // B. 动态生成 Texture Bindings (Group 1)
        let mut binding_code = String::from(
            "@group(1) @binding(0) var<uniform> material: MaterialUniforms;\n"
        );

        let mut binding_idx = 1;
        for def in TEXTURE_DEFINITIONS {
            if features.contains(def.feature) {
                // 1. 设置宏: use_map = true (供模板逻辑使用)
                context.insert(format!("use_{}", def.name), Value::Bool(true));

                // 2. 生成 WGSL Binding 代码
                binding_code.push_str(&format!(
                    r#"
                    @group(1) @binding({}) var t_{}: texture_2d<f32>;
                    @group(1) @binding({}) var s_{}: sampler;
                    "#,
                    binding_idx,     def.name,
                    binding_idx + 1, def.name
                ));
                binding_idx += 2;
            }
        }
        context.insert("binding_code".to_string(), Value::String(binding_code));

        // B. 渲染 Fragment 部分
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