#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 资产读取器 Trait
/// 支持本地文件和网络资源的异步读取
#[cfg(not(target_arch = "wasm32"))]
pub trait AssetReader: Send + Sync {
    /// 异步读取资源字节流
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = anyhow::Result<Vec<u8>>> + Send;
}

/// WASM 版本的 AssetReader - 不要求 Send
#[cfg(target_arch = "wasm32")]
pub trait AssetReader: Send + Sync {
    /// 异步读取资源字节流
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = anyhow::Result<Vec<u8>>>;
}

/// 本地文件读取器 (仅在非 WASM 平台可用)
#[cfg(not(target_arch = "wasm32"))]
pub struct FileAssetReader {
    root_path: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileAssetReader {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let root_path = if path.is_file() {
            path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            path.to_path_buf()
        };
        Self { root_path }
    }

    #[inline]
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetReader for FileAssetReader {
    async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        let path = self.root_path.join(uri);
        let data = tokio::fs::read(&path).await?;
        Ok(data)
    }
}

/// HTTP 网络读取器 (条件编译 - 仅 Native)
#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
pub struct HttpAssetReader {
    root_url: reqwest::Url,
    client: reqwest::Client,
}

#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
impl HttpAssetReader {
    pub fn new(url_str: &str) -> anyhow::Result<Self> {
        let url = reqwest::Url::parse(url_str)?;
        let root_url = if !url.path().ends_with('/') {
            let mut u = url.clone();
            if let Ok(mut segments) = u.path_segments_mut() {
                segments.pop();
                segments.push("");
            }
            u
        } else {
            url
        };

        
        Ok(Self {
            root_url,
            client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?,
        })
    }

    #[inline]
    pub fn root_url(&self) -> &reqwest::Url {
        &self.root_url
    }
}

#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
impl AssetReader for HttpAssetReader {
    async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        let url = self.root_url.join(uri)?;
        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", resp.status()));
        }
        let bytes = resp.bytes().await?;
        Ok(bytes.to_vec())
    }
}

/// WASM HTTP 读取器 (使用 web-sys fetch API)
#[cfg(target_arch = "wasm32")]
pub struct WasmHttpReader {
    root_url: String,
}

#[cfg(target_arch = "wasm32")]
impl WasmHttpReader {
    pub fn new(url_str: &str) -> anyhow::Result<Self> {
        // 计算根 URL（移除文件名部分）
        let root_url = if !url_str.ends_with('/') {
            url_str.rsplit_once('/').map(|(base, _)| format!("{}/", base)).unwrap_or_else(|| url_str.to_string())
        } else {
            url_str.to_string()
        };
        
        Ok(Self { root_url })
    }

    #[inline]
    pub fn root_url(&self) -> &str {
        &self.root_url
    }
}

#[cfg(target_arch = "wasm32")]
impl AssetReader for WasmHttpReader {
    async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};
        
        let url = format!("{}{}", self.root_url, uri);
        
        let window = web_sys::window().ok_or_else(|| anyhow::anyhow!("No window found"))?;
        
        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);
        
        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| anyhow::anyhow!("Failed to create request: {:?}", e))?;
        
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow::anyhow!("Fetch failed: {:?}", e))?;
        
        let resp: Response = resp_value.dyn_into()
            .map_err(|_| anyhow::anyhow!("Response is not a Response object"))?;
        
        if !resp.ok() {
            return Err(anyhow::anyhow!("HTTP error: {}", resp.status()));
        }
        
        let array_buffer = JsFuture::from(
            resp.array_buffer().map_err(|e| anyhow::anyhow!("Failed to get array buffer: {:?}", e))?
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read response: {:?}", e))?;
        
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        Ok(uint8_array.to_vec())
    }
}

/// 资产读取器变体枚举
/// 避免 trait object 的运行时开销
#[derive(Clone)]
pub enum AssetReaderVariant {
    #[cfg(not(target_arch = "wasm32"))]
    File(Arc<FileAssetReader>),
    #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
    Http(Arc<HttpAssetReader>),
    #[cfg(target_arch = "wasm32")]
    WasmHttp(Arc<WasmHttpReader>),
}

impl AssetReaderVariant {
    /// 从路径或 URL 自动创建合适的读取器
    pub fn from_source(source: &str) -> anyhow::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if source.starts_with("http://") || source.starts_with("https://") {
                #[cfg(feature = "http")]
                {
                    Ok(Self::Http(Arc::new(HttpAssetReader::new(source)?)))
                }
                #[cfg(not(feature = "http"))]
                {
                    Err(anyhow::anyhow!(
                        "HTTP feature is not enabled. Enable it with `features = [\"http\"]`"
                    ))
                }
            } else {
                Ok(Self::File(Arc::new(FileAssetReader::new(source))))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, always use HTTP reader (fetch API)
            Ok(Self::WasmHttp(Arc::new(WasmHttpReader::new(source)?)))
        }
    }

    /// 异步读取字节数据
    pub async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::File(r) => r.read_bytes(uri).await,
            #[cfg(all(feature = "http", not(target_arch = "wasm32")))]
            Self::Http(r) => r.read_bytes(uri).await,
            #[cfg(target_arch = "wasm32")]
            Self::WasmHttp(r) => r.read_bytes(uri).await,
        }
    }

    /// 获取基础路径的文件名部分
    pub fn source_filename(source: &str) -> &str {
        if source.starts_with("http://") || source.starts_with("https://") {
            source.rsplit('/').next().unwrap_or(source)
        } else {
            std::path::Path::new(source)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(source)
        }
    }
}
