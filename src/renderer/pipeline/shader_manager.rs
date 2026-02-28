//! Shader Template Manager
//!
//! Manages WGSL shaders using the minijinja template engine and provides
//! a centralized `ShaderModule` cache shared by all pipeline-creation paths.
//!
//! ## Two compilation modes
//!
//! | Method | Use case | Source |
//! |--------|----------|--------|
//! | [`ShaderManager::get_or_compile_template`] | Material / post-process shaders | minijinja template |
//! | [`ShaderManager::get_or_compile_raw`]      | Compute / utility shaders        | raw WGSL string    |

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
#[folder = "src/renderer/pipeline/shaders"]
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

        env.set_path_join_callback(|name, _parent| format!("chunks/{name}").into());

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

// ─── ShaderManager ────────────────────────────────────────────────────────────

/// Centralized shader module cache.
///
/// Deduplicates compiled `wgpu::ShaderModule`s by hashing the **final** WGSL
/// source with xxh3-128. Two entry points are provided — one for template-based
/// shaders (materials, post-processing) and one for raw WGSL strings (compute /
/// utility).
///
/// Owned by `RendererState`; references are handed out via `PrepareContext`.
pub struct ShaderManager {
    /// xxh3-128 of final WGSL → compiled module.
    module_cache: FxHashMap<u128, wgpu::ShaderModule>,
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
        }
    }

    /// Compile a shader from a **minijinja template** (or return a cached module).
    ///
    /// Internally calls [`ShaderGenerator::generate_shader`], hashes the output,
    /// and caches the resulting `ShaderModule`.
    ///
    /// Returns `(module_ref, source_hash)`.
    pub fn get_or_compile_template(
        &mut self,
        device: &wgpu::Device,
        template_name: &str,
        options: &ShaderCompilationOptions,
        vertex_input_code: &str,
        binding_code: &str,
    ) -> (&wgpu::ShaderModule, u128) {
        let source = ShaderGenerator::generate_shader(
            vertex_input_code,
            binding_code,
            template_name,
            options,
        );

        if cfg!(debug_assertions) {
            Self::debug_print_shader(template_name, &source);
        }

        let hash = xxh3_128(source.as_bytes());

        let module = self.module_cache.entry(hash).or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Shader Module {template_name}")),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        });

        (module, hash)
    }

    /// Compile a **raw WGSL** string (or return a cached module).
    ///
    /// Used by compute / utility passes that embed their WGSL via `include_str!`
    /// rather than going through the template engine.
    ///
    /// Returns `(module_ref, source_hash)`.
    pub fn get_or_compile_raw(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        source: &str,
    ) -> (&wgpu::ShaderModule, u128) {
        let hash = xxh3_128(source.as_bytes());

        let module = self.module_cache.entry(hash).or_insert_with(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        });

        (module, hash)
    }

    /// Returns the number of cached shader modules.
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.module_cache.len()
    }

    fn debug_print_shader(template_name: &str, source: &str) {
        fn normalize_newlines(s: &str) -> String {
            let mut result = String::with_capacity(s.len());
            let mut last_was_newline = false;
            for c in s.chars() {
                if c == '\n' {
                    if !last_was_newline {
                        result.push('\n');
                        last_was_newline = true;
                    }
                } else {
                    result.push(c);
                    last_was_newline = false;
                }
            }
            result
        }

        println!(
            "================= Generated Shader Code {} ==================\n {}",
            template_name,
            normalize_newlines(source)
        );
    }
}
