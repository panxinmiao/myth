use std::borrow::Cow;
use std::path::{Path};
use std::sync::OnceLock;
use minijinja::{Environment, Error, ErrorKind, syntax::SyntaxConfig};
use rust_embed::RustEmbed; 

// 全局单例 Environment
pub static SHADER_ENV: OnceLock<Environment<'static>> = OnceLock::new();


// 定义要嵌入的文件夹
#[derive(RustEmbed)]
#[folder = "src/render/pipeline/shaders"]
struct ShaderAssets;

pub fn get_env() -> &'static Environment<'static> {
    SHADER_ENV.get_or_init(|| {
        let mut env = Environment::new();
        
        // 1. 配置语法：完全对齐 pygfx / Jinja2 自定义配置
        let syntax = SyntaxConfig::builder()
            .block_delimiters("{$", "$}")
            .variable_delimiters("{{", "}}")
            .line_statement_prefix("$$") // 允许使用 $$ if use_map 这种写法
            .build()
            .expect("Failed to configure Jinja2 syntax");

        env.set_syntax(syntax);
        env.set_undefined_behavior(minijinja::UndefinedBehavior::SemiStrict);

        env.set_loader(shader_loader);

        env
    })
}

/// 核心加载逻辑
/// 当模板中写 {% include "common" %} 时，name 参数就是 "common"
fn shader_loader(name: &str) -> Result<Option<String>, Error> {
    // 1. 处理扩展名
    // 自动补全 "xx.wgsl"
    let filename = if name.ends_with(".wgsl") {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{}.wgsl", name))
    };

    // 2. 尝试加载内容
    // 这里我们使用一种“混合策略”：
    // 开发时：如果我们能直接访问物理文件系统，就读物理文件（方便热重载/调试）
    // 发布时：或者物理文件不存在时，尝试从嵌入资源中读取
    
    // 策略 A: 优先尝试物理文件 (Debug Friendly)
    #[cfg(debug_assertions)] 
    {
        let path = Path::new("src/render/pipeline/shaders").join(filename.as_ref());
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
    // 这里的 filename 需要适配 rust-embed 的路径格式（通常是相对路径，如 "bsdfs/pbr.wgsl"）
    // 注意：RustEmbed 的路径分隔符通常是 "/"
    if let Some(file) = ShaderAssets::get(filename.as_ref()) {
        if let Ok(source) = std::str::from_utf8(file.data.as_ref()) {
            return Ok(Some(source.to_string()));
        }
    }

    // 没找到
    Ok(None)
}