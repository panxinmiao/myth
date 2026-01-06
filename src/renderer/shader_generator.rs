use crate::core::material::{Material, MaterialFeatures};
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


pub struct ShaderGenerator;

impl ShaderGenerator {
    // 生成最终的 WGSL 代码
    /// 
    /// # 参数
    /// * `material`: 材质实例，用于确定开启了哪些 Feature
    /// * `geometry_layout`: 几何体布局，包含 Vertex Input 的定义代码
    /// * `template_name`: 模板名称 (例如 "materials/mesh_standard")，对应 assets/shaders 下的文件
    pub fn generate_shader(
        material: &Material,
        geometry_layout: &crate::renderer::layout_generator::GeneratedLayout,
        template_name: &str) -> String {

        let env = get_env();
        let features = material.features();


        let mut context = Map::new();
        
        // 2. 注入 Rust 生成的代码片段 (Strings)
        context.insert(
            "vertex_struct_code".to_string(), 
            Value::String(geometry_layout.shader_code.clone())
        );
        context.insert(
            "uniform_struct_code".to_string(), 
            Value::String(material.wgsl_struct_def().to_string())
        );

        // --------------------------------------------------------
        // 生成 Binding 代码
        // --------------------------------------------------------

        
        let mut binding_code = String::new();
        // @group(1) @binding(0) var<uniform> material: MaterialUniforms;
        binding_code.push_str(
            "@group(1) @binding(0) var<uniform> material: MaterialUniforms;\n"
        );
        let mut binding_idx = 1; // 0 号位留给了 material uniform

        for def in TEXTURE_DEFINITIONS {
            if features.contains(def.feature) {
                // 自动生成 t_xxx 和 s_xxx
                binding_code.push_str(&format!(
                    r#"
                    @group(1) @binding({}) var t_{}: texture_2d<f32>;
                    @group(1) @binding({}) var s_{}: sampler;
                    "#,
                    binding_idx,     def.name, // e.g. binding(1) t_map
                    binding_idx + 1, def.name  // e.g. binding(2) s_map
                ));
                
                // 每次消耗 2 个槽位 (Texture + Sampler)
                binding_idx += 2;

                // B. 注入 Feature 开关 -> "use_map": true
                context.insert(format!("use_{}", def.name), Value::Bool(true));
            }
        }

        // 注入生成的 binding 代码
        context.insert("binding_code".to_string(), Value::String(binding_code));

        // --------------------------------------------------------
        // 3. 渲染模板
        // --------------------------------------------------------
        let template = env.get_template(template_name)
            .unwrap_or_else(|e| panic!("Failed to load template '{}': {}", template_name, e));
        
        let final_shader = template.render(&context)
            .unwrap_or_else(|e| panic!("Failed to render shader: {}", e));

        final_shader

    }

}