//! Shader Template Manager
//!
//! Manages WGSL shaders using the minijinja template engine

use minijinja::value::{Object, Value};
use minijinja::{Environment, Error, ErrorKind, syntax::SyntaxConfig};
use rust_embed::RustEmbed;
use serde::Serialize;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

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
