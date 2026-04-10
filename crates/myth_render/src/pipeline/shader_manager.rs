//! Shader Template Manager
//!
//! Manages WGSL shaders using the minijinja template engine and provides
//! a centralized `ShaderModule` cache shared by all pipeline-creation paths.
//!
//! All shaders — whether loaded from built-in template files or provided as
//! inline WGSL strings — are compiled through the unified
//! [`ShaderManager::get_or_compile`] method, which accepts a [`ShaderSource`]
//! discriminant to select the source kind.

use minijinja::value::{Object, Value};
use minijinja::{Environment, Error, ErrorKind, syntax::SyntaxConfig};
use rust_embed::RustEmbed;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use xxhash_rust::xxh3::xxh3_128;

use super::shader_gen::{ShaderCompilationOptions, ShaderGenerator};

pub static SHADER_ENV: OnceLock<Environment<'static>> = OnceLock::new();

#[derive(RustEmbed)]
#[folder = "src/pipeline/shaders"]
struct ShaderAssets;

pub fn get_env() -> &'static Environment<'static> {
    SHADER_ENV.get_or_init(|| {
        let mut env = Environment::new();

        let syntax = SyntaxConfig::builder()
            .block_delimiters("{$", "$}")
            .variable_delimiters("{{", "}}")
            .line_statement_prefix("$$")
            .build()
            .expect("Failed to configure Jinja2 syntax");

        env.set_syntax(syntax);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::SemiStrict);

        env.set_loader(shader_loader);

        env.add_function("next_loc", next_location);

        env
    })
}

fn shader_loader(name: &str) -> Result<Option<String>, Error> {
    let filename = if std::path::Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wgsl"))
    {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{name}.wgsl"))
    };

    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    {
        let path = std::path::Path::new("src/renderer/pipeline/shaders").join(filename.as_ref());
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(source) => return Ok(Some(source)),
                Err(e) => {
                    return Err(Error::new(
                        ErrorKind::TemplateNotFound,
                        format!("Failed to read file: {e}"),
                    ));
                }
            }
        }
    }

    if let Some(file) = ShaderAssets::get(&filename)
        && let Ok(source) = std::str::from_utf8(file.data.as_ref())
    {
        return Ok(Some(source.to_string()));
    }

    Ok(None)
}

fn next_location(allocator: &LocationAllocator) -> u32 {
    allocator.next()
}

#[derive(Debug, Serialize)]
pub struct LocationAllocator {
    #[serde(skip)]
    counter: AtomicU32,
}

impl Default for LocationAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl LocationAllocator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }

    pub fn next(&self) -> u32 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl Object for LocationAllocator {
    fn call_method(
        self: &Arc<Self>,
        _state: &minijinja::State,
        name: &str,
        _args: &[Value],
    ) -> Result<Value, Error> {
        if name == "next" {
            Ok(Value::from(self.next()))
        } else {
            Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("method {name} not found"),
            ))
        }
    }
}

// ─── ShaderSource ─────────────────────────────────────────────────────────────

/// Identifies the origin of a shader's WGSL source.
///
/// * [`File`](Self::File) — a built-in template resolved through the embedded
///   shader asset loader (e.g. `"entry/main/physical"` or `"entry/utility/skybox"`).
/// * [`Inline`](Self::Inline) — a raw WGSL string supplied at call-time, often
///   via `include_str!()`. If a custom template with the same `name` was
///   registered, the custom source takes priority.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum ShaderSource<'a> {
    /// Load from a built-in template file.
    File(&'a str),
    /// Use an inline WGSL source string identified by `name` for labels.
    Inline { name: &'a str, source: &'a str },
}

// ─── ShaderManager ────────────────────────────────────────────────────────────

/// Centralized shader module cache.
///
/// Deduplicates compiled `wgpu::ShaderModule`s by hashing the **final** WGSL
/// source with xxh3-128. Shaders are compiled through the single
/// [`get_or_compile`](Self::get_or_compile) entry point, which accepts a
/// [`ShaderSource`] to distinguish file-based templates from inline sources.
///
/// Custom shader templates registered via [`register_template`](Self::register_template)
/// are stored alongside the built-in environment. When a template name is
/// resolved, custom templates take priority over built-in ones.
///
/// Owned by `RendererState`; references are handed out via `PrepareContext`.
pub struct ShaderManager {
    /// xxh3-128 of final WGSL → compiled module.
    module_cache: FxHashMap<u128, wgpu::ShaderModule>,
    /// User-registered custom shader templates (name → WGSL source).
    custom_templates: FxHashMap<String, String>,
}

impl Default for ShaderManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            module_cache: FxHashMap::default(),
            custom_templates: FxHashMap::default(),
        }
    }

    /// Registers a custom shader template source under the given name.
    ///
    /// Once registered, any material that references this `name` via
    /// `#[myth_material(shader = "name")]` will use the provided WGSL
    /// source instead of looking up a built-in template.
    ///
    /// The source string goes through the minijinja template engine, so
    /// `{$ include "chunks/..." $}` directives are fully supported.
    pub fn register_template(&mut self, name: impl Into<String>, source: impl Into<String>) {
        let name = name.into();
        log::info!("Registered custom shader template: {name}");
        self.custom_templates.insert(name, source.into());
    }

    /// Compile a shader (or return a cached module).
    ///
    /// For [`ShaderSource::File`], if a custom template was registered under
    /// the same name, its source is rendered instead of the built-in template.
    /// For [`ShaderSource::Inline`], the provided WGSL string is rendered as a
    /// one-off minijinja template, enabling `{$ include $}` directives even in
    /// inline sources.
    ///
    /// Returns `(module_ref, source_hash)`.
    pub fn get_or_compile(
        &mut self,
        device: &wgpu::Device,
        source: ShaderSource,
        options: &ShaderCompilationOptions,
    ) -> (&wgpu::ShaderModule, u128) {
        let final_wgsl = match source {
            ShaderSource::File(path) => {
                if let Some(custom_src) = self.custom_templates.get(path) {
                    ShaderGenerator::generate_custom_shader(path, custom_src, options)
                } else {
                    ShaderGenerator::generate_shader(path, options)
                }
            }
            ShaderSource::Inline {
                name,
                source: inline_src,
            } => {
                let src = self
                    .custom_templates
                    .get(name)
                    .map_or(inline_src, String::as_str);
                ShaderGenerator::generate_custom_shader(name, src, options)
            }
        };

        let hash = xxh3_128(final_wgsl.as_bytes());

        let label = match source {
            ShaderSource::File(path) => path,
            ShaderSource::Inline { name, .. } => name,
        };

        let module = self.module_cache.entry(hash).or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Shader Module {label}")),
                source: wgpu::ShaderSource::Wgsl(final_wgsl.into()),
            })
        });

        (module, hash)
    }

    /// Returns the number of cached shader modules.
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.module_cache.len()
    }
}
