//! 着色器代码生成器
//!
//! 使用模板引擎生成最终的 WGSL 代码

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::resources::shader_defines::ShaderDefines;
use super::shader_manager::{get_env, LocationAllocator};
use minijinja::value::Value;
use serde::Serialize;

/// Shader 编译选项
/// 
/// 包含生成 Shader 所需的所有宏定义。
/// 使用 `ShaderDefines` 存储所有来自材质、几何体和场景的宏定义。
#[derive(Debug, Clone, Default)]
pub struct ShaderCompilationOptions {
    pub(crate) defines: ShaderDefines,
}

impl ShaderCompilationOptions {
    /// 创建新的编译选项
    pub fn new() -> Self {
        Self {
            defines: ShaderDefines::new(),
        }
    }

    /// 从材质、几何体和场景的宏定义合并创建
    pub fn from_merged(
        mat_defines: &ShaderDefines,
        geo_defines: &ShaderDefines,
        scene_defines: &ShaderDefines,
        item_defines: &ShaderDefines,
    ) -> Self {
        let mut defines = mat_defines.clone();
        defines.merge(geo_defines);
        defines.merge(scene_defines);
        defines.merge(item_defines);
        Self { defines }
    }

    /// 获取宏定义的引用
    #[inline]
    pub fn defines(&self) -> &ShaderDefines {
        &self.defines
    }

    /// 获取可变的宏定义引用
    #[inline]
    pub fn defines_mut(&mut self) -> &mut ShaderDefines {
        &mut self.defines
    }

    /// 计算编译选项的哈希值（用于缓存）
    pub fn compute_hash(&self) -> u64 {
        self.defines.compute_hash()
    }

    /// 转换为模板渲染所需的 Map
    fn to_template_map(&self) -> BTreeMap<String, String> {
        self.defines
            .iter_strings()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }
}

impl Hash for ShaderCompilationOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.defines.as_slice().hash(state);
    }
}

impl PartialEq for ShaderCompilationOptions {
    fn eq(&self, other: &Self) -> bool {
        self.defines == other.defines
    }
}

impl Eq for ShaderCompilationOptions {}

#[derive(Serialize)]
struct ShaderContext<'a> {
    #[serde(flatten)]
    defines: BTreeMap<String, String>,
    vertex_input_code: Option<&'a str>,
    binding_code: &'a str,
    loc: Value,
}

pub struct ShaderGenerator;

impl ShaderGenerator {
    pub fn generate_shader(
        vertex_input_code: &str,
        binding_code: &str,
        template_name: &str,
        options: &ShaderCompilationOptions,
    ) -> String {
        let env = get_env();
        let allocator = LocationAllocator::new();
        let loc_value = Value::from_object(allocator);

        let ctx = ShaderContext {
            defines: options.to_template_map(),
            vertex_input_code: Some(vertex_input_code),
            binding_code: &binding_code,
            loc: loc_value,
        };

        let template = env.get_template(&template_name)
            .expect("Shader template not found");

        let source = template.render(&ctx)
            .expect("Shader render failed");

        format!("// === Auto-generated Unified Shader ===\n{}", source)
    }

}
