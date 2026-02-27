//! Shader Code Generator
//!
//! Uses a template engine to generate final WGSL code

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use super::shader_manager::{LocationAllocator, get_env};
use crate::resources::shader_defines::ShaderDefines;
use minijinja::value::Value;
use serde::Serialize;

/// Shader compilation options.
///
/// Contains all macro definitions needed to generate a shader.
/// Uses `ShaderDefines` to store all defines from materials, geometries, and scenes.
#[derive(Debug, Clone, Default)]
pub struct ShaderCompilationOptions {
    pub(crate) defines: ShaderDefines,
}

impl ShaderCompilationOptions {
    /// Creates new compilation options.
    #[must_use]
    pub fn new() -> Self {
        Self {
            defines: ShaderDefines::new(),
        }
    }

    /// Creates from merged material, geometry, and scene defines.
    #[must_use]
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

    /// Returns a reference to the shader defines.
    #[inline]
    #[must_use]
    pub fn defines(&self) -> &ShaderDefines {
        &self.defines
    }

    /// Returns a mutable reference to the shader defines.
    #[inline]
    pub fn defines_mut(&mut self) -> &mut ShaderDefines {
        &mut self.defines
    }

    pub fn add_define(&mut self, key: &str, value: &str) {
        self.defines.set(key, value);
    }

    /// Computes the hash of the compilation options (used for caching).
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        self.defines.compute_hash()
    }

    /// Converts to the Map required for template rendering.
    fn to_template_map(&self) -> BTreeMap<String, String> {
        self.defines
            .iter_strings()
            .map(|(k, v)| (k.clone(), v.clone()))
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
    #[must_use]
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
            binding_code,
            loc: loc_value,
        };

        let template = env
            .get_template(template_name)
            .expect("Shader template not found");

        let source = template.render(&ctx).expect("Shader render failed");

        format!("// === Auto-generated Unified Shader ===\n{source}")
    }
}
