//! Shader Code Generator
//!
//! Renders final WGSL source from minijinja templates, substituting feature
//! defines and injected code blocks into the shader output.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use super::shader_manager::{LocationAllocator, get_env};
use minijinja::value::Value;
use myth_resources::shader_defines::ShaderDefines;
use serde::Serialize;

/// Shader compilation options.
///
/// Collects all macro definitions and injected code blocks needed to generate
/// a shader. Defines originate from materials, geometries, and scenes; code
/// blocks are arbitrary WGSL snippets injected by pipeline code (e.g. vertex
/// input declarations, bind group definitions).
#[derive(Debug, Clone, Default)]
pub struct ShaderCompilationOptions {
    pub(crate) defines: ShaderDefines,
    pub(crate) code_blocks: BTreeMap<String, String>,
}

impl ShaderCompilationOptions {
    /// Creates new compilation options with empty defines and no code blocks.
    #[must_use]
    pub fn new() -> Self {
        Self {
            defines: ShaderDefines::new(),
            code_blocks: BTreeMap::new(),
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
        Self {
            defines,
            code_blocks: BTreeMap::new(),
        }
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

    /// Injects a named code block into the template context.
    ///
    /// The block becomes available as `{{ key }}` inside templates. Common
    /// keys include `"vertex_input_code"` and `"binding_code"`.
    pub fn inject_code(&mut self, key: impl Into<String>, code: impl Into<String>) {
        self.code_blocks.insert(key.into(), code.into());
    }

    /// Computes the hash of the compilation options (used for caching).
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        Hash::hash(self, &mut h);
        h.finish()
    }

    /// Converts defines to the map required for template rendering.
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
        self.code_blocks.hash(state);
    }
}

impl PartialEq for ShaderCompilationOptions {
    fn eq(&self, other: &Self) -> bool {
        self.defines == other.defines && self.code_blocks == other.code_blocks
    }
}

impl Eq for ShaderCompilationOptions {}

/// Template rendering context passed to minijinja.
///
/// Both `defines` and `code_blocks` are flattened into the top-level namespace
/// so that templates access them directly as `{{ key }}`. The `loc` helper
/// provides auto-incrementing `@location(N)` indices for vertex outputs.
#[derive(Serialize)]
struct ShaderContext<'a> {
    #[serde(flatten)]
    defines: BTreeMap<String, String>,
    #[serde(flatten)]
    code_blocks: &'a BTreeMap<String, String>,
    loc: Value,
}

pub struct ShaderGenerator;

impl ShaderGenerator {
    /// Builds a [`ShaderContext`] from the compilation options.
    fn build_context<'a>(options: &'a ShaderCompilationOptions) -> ShaderContext<'a> {
        let allocator = LocationAllocator::new();
        ShaderContext {
            defines: options.to_template_map(),
            code_blocks: &options.code_blocks,
            loc: Value::from_object(allocator),
        }
    }

    /// Generates WGSL from a **built-in** template registered in the shader environment.
    #[must_use]
    pub fn generate_shader(
        template_name: &str,
        options: &ShaderCompilationOptions,
    ) -> String {
        let env = get_env();
        let ctx = Self::build_context(options);

        let template = env
            .get_template(template_name)
            .expect("Shader template not found");

        let source = template.render(&ctx).expect("Shader render failed");

        format!("// === Auto-generated Unified Shader ===\n{source}")
    }

    /// Generates WGSL from a **custom** template source string.
    ///
    /// The source is parsed as a one-off template while still resolving
    /// `{$ include $}` directives through the global shader loader. This
    /// allows custom shaders to reuse built-in chunks.
    #[must_use]
    pub fn generate_custom_shader(
        template_name: &str,
        template_source: &str,
        options: &ShaderCompilationOptions,
    ) -> String {
        let env = get_env();
        let ctx = Self::build_context(options);

        let source = env
            .render_named_str(template_name, template_source, &ctx)
            .expect("Custom shader render failed");

        format!("// === Auto-generated Unified Shader ===\n{source}")
    }
}
