use crate::core::material::{Material, MaterialFeatures};

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
    TextureDef { feature: MaterialFeatures::USE_NORMAL_MAP,    name: "normal" },
    TextureDef { feature: MaterialFeatures::USE_ROUGHNESS_MAP, name: "roughness" },
    TextureDef { feature: MaterialFeatures::USE_EMISSIVE_MAP,  name: "emissive" },
    // 未来添加新贴图只需在这里加一行，例如:
    // TextureDef { feature: MaterialFeatures::USE_AO_MAP, name: "ao" },
];

pub struct ShaderGenerator;

impl ShaderGenerator {
    pub fn generate_shader(material: &Material, base_shader_source: &str) -> String {
        let mut final_code = String::new();

        // 1. 注入常量定义 (模拟 #ifdef)
        // WGPU 编译器会自动优化掉 if(false) 的代码块
        let features = material.features();
        final_code.push_str(&Self::generate_feature_flags(features));

        // 2. 注入 Uniform 结构体定义 (来自 Material 的单一事实来源)
        final_code.push_str(material.wgsl_struct_def());
        final_code.push('\n');

        // 3. 注入 BindGroup 定义
        // Group 1 Binding 0 是固定的 Material Uniform
        final_code.push_str(r#"
            @group(1) @binding(0) var<uniform> material: MaterialUniforms;
        "#);

        // 4. 注入纹理 Bindings
        // 注意：这里的 binding 索引必须与 ResourceManager 创建 BindGroup 的顺序一致
        let mut binding_idx = 1;
        
        for def in TEXTURE_DEFINITIONS {
            if features.contains(def.feature) {
                // 自动生成 t_xxx 和 s_xxx
                final_code.push_str(&format!(
                    r#"
                    @group(1) @binding({}) var t_{}: texture_2d<f32>;
                    @group(1) @binding({}) var s_{}: sampler;
                    "#,
                    binding_idx,     def.name, // e.g. binding(1) t_map
                    binding_idx + 1, def.name  // e.g. binding(2) s_map
                ));
                
                // 每次消耗 2 个槽位 (Texture + Sampler)
                binding_idx += 2;
            }
        }

        // 5. 拼接基础 Shader 模板
        final_code.push_str(base_shader_source);

        final_code
    }

    fn generate_feature_flags(features: MaterialFeatures) -> String {
        let mut s = String::new();
        // 同样遍历定义表，自动生成 const USE_XXX = true/false;
        // 注意：这里我们需要把 bitflags 的名字转为字符串，或者手动映射
        
        // 由于 Rust 无法简单地从 feature 变量反射出 "USE_MAP" 字符串，
        // 这里我们可以稍微硬编码，或者给 TextureDef 加一个 macro_name 字段。
        // 为了简单，我们这里手动写几个主要的，或者扩展 TextureDef。
        
        // 改进：为了彻底自动化，建议给 TextureDef 加一个 macro_name 字段。
        // 但目前为了不改动太大，我们还是用显式判断，或者用简单的字符串转换规则。
        
        s.push_str(&format!("const USE_MAP: bool = {};\n", features.contains(MaterialFeatures::USE_MAP)));
        s.push_str(&format!("const USE_NORMAL_MAP: bool = {};\n", features.contains(MaterialFeatures::USE_NORMAL_MAP)));
        s.push_str(&format!("const USE_ROUGHNESS_MAP: bool = {};\n", features.contains(MaterialFeatures::USE_ROUGHNESS_MAP)));
        s.push_str(&format!("const USE_EMISSIVE_MAP: bool = {};\n", features.contains(MaterialFeatures::USE_EMISSIVE_MAP)));
        
        s
    }
}