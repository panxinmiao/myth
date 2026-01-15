use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;
use minijinja::{Environment, Error, ErrorKind, syntax::SyntaxConfig};
use rust_embed::RustEmbed;
use std::sync::atomic::{AtomicU32, Ordering};
use serde::Serialize;
use minijinja::value::{Value, Object};

// 全局单例 Environment
pub static SHADER_ENV: OnceLock<Environment<'static>> = OnceLock::new();

// 定义要嵌入的文件夹
#[derive(RustEmbed)]
#[folder = "src/render/pipeline/shaders"]
struct ShaderAssets;

pub fn get_env() -> &'static Environment<'static> {
    SHADER_ENV.get_or_init(|| {
        let mut env = Environment::new();

        let syntax = SyntaxConfig::builder()
            .block_delimiters("{$", "$}")
            .variable_delimiters("{{", "}}")
            .line_statement_prefix("$$") // 允许使用 $$ if use_map 这种写法
            .build()
            .expect("Failed to configure Jinja2 syntax");

        env.set_syntax(syntax);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::SemiStrict);

        env.set_loader(shader_loader);

        env.set_path_join_callback(|name, _parent| {
            format!("chunks/{}", name).into()
        });

        env.add_function("next_loc", next_location);

        env
    })
}

/// 核心加载逻辑
fn shader_loader(name: &str) -> Result<Option<String>, Error> {
    // 1. 处理扩展名
    // 自动补全 "xx.wgsl"
    let filename = if name.ends_with(".wgsl") {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{}.wgsl", name))
    };

    // 2. 尝试加载内容    
    // 策略 A: 优先尝试物理文件 (Debug Friendly)
    #[cfg(debug_assertions)] 
    {
        let path = std::path::Path::new("src/render/pipeline/shaders").join(filename.as_ref());
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(source) => return Ok(Some(source)),
                Err(e) => return Err(Error::new(
                    ErrorKind::TemplateNotFound, 
                    format!("Failed to read file: {}", e)
                )),
            }
        }
    }

    // 策略 B: 尝试从 RustEmbed 加载 (Release Friendly)
    if let Some(file) = ShaderAssets::get(filename.as_ref())
        && let Ok(source) = std::str::from_utf8(file.data.as_ref()) {
            return Ok(Some(source.to_string()));
        }

    // 没找到
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
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }

    pub fn next(&self) -> u32 {
        // 每次调用自动 +1
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl Object for LocationAllocator {
    fn call_method(
        self: &Arc<Self>, 
        _state: &minijinja::State, 
        name: &str, 
        _args: &[Value]
    ) -> Result<Value, Error> {
        if name == "next" {
            Ok(Value::from(self.next()))
        } else {
            Err(Error::new(ErrorKind::UnknownMethod, format!("method {} not found", name)))
        }
    }
}