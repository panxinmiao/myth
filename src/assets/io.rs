#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::{borrow::Cow, sync::Arc};

/// 资产读取器 Trait
/// 支持本地文件和网络资源的异步读取
pub trait AssetReader: Send + Sync {
    /// 异步读取资源字节流
    #[cfg(not(target_arch = "wasm32"))]
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = anyhow::Result<Vec<u8>>> + Send;

    /// WASM 下 Future 不需要 Send
    #[cfg(target_arch = "wasm32")]
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

/// HTTP 网络读取器
/// reqwest 跨平台特性 (Native 使用 tokio, WASM 使用 fetch)
#[cfg(all(feature = "http"))]
pub struct HttpAssetReader {
    root_url: reqwest::Url,
    client: reqwest::Client,
}

#[cfg(all(feature = "http"))]
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

        let client = reqwest::Client::builder();

        // Native 平台特定的超时设置等
        #[cfg(not(target_arch = "wasm32"))]
        let client = client.timeout(std::time::Duration::from_secs(30));

        
        Ok(Self {
            root_url,
            client: client.build()?,
        })
    }

    #[inline]
    pub fn root_url(&self) -> &reqwest::Url {
        &self.root_url
    }
}

#[cfg(all(feature = "http"))]
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

/// 资产读取器变体枚举
#[derive(Clone)]
pub enum AssetReaderVariant {
    #[cfg(not(target_arch = "wasm32"))]
    File(Arc<FileAssetReader>),
    #[cfg(all(feature = "http"))]
    Http(Arc<HttpAssetReader>),
}

impl AssetReaderVariant {
    /// 从路径或 URL 自动创建合适的读取器
    pub fn new(source: &impl AssetSource) -> anyhow::Result<Self> {

        let uri = source.uri();

        #[cfg(not(target_arch = "wasm32"))]
        {
            if source.is_http() {
                #[cfg(feature = "http")]
                {
                    Ok(Self::Http(Arc::new(HttpAssetReader::new(&uri)?)))
                }
                #[cfg(not(feature = "http"))]
                {
                    Err(anyhow::anyhow!("HTTP feature is not enabled"))
                }
            } else {
                // 如果不是 HTTP，默认走文件系统
                Ok(Self::File(Arc::new(FileAssetReader::new(uri.as_ref()))))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            // WASM 统一走 HTTP 读取
            #[cfg(not(feature = "http"))]
            {
                return Err(anyhow::anyhow!("HTTP feature is not enabled"));
            }

            let full_uri = if !source.is_http() {
                let window = web_sys::window()
                    .ok_or_else(|| anyhow::anyhow!("No window found"))?;
                
                let location = window.location();
                let href = location.href()
                    .map_err(|e| anyhow::anyhow!("Failed to get href: {:?}", e))?;

                let base = reqwest::Url::parse(&href)
                    .map_err(|e| anyhow::anyhow!("Invalid base URL: {}", e))?;
                
                base.join(&uri)
                    .map_err(|e| anyhow::anyhow!("Failed to join URL: {}", e))?
                    .to_string()
            } else {
                uri.to_string()
            };

            Ok(Self::Http(Arc::new(HttpAssetReader::new(&full_uri)?)))
        }
    }

    /// 异步读取字节数据
    pub async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::File(r) => r.read_bytes(uri).await,
            #[cfg(all(feature = "http"))]
            Self::Http(r) => r.read_bytes(uri).await,
        }
    }

}


pub trait AssetSource: std::fmt::Debug + Send + Sync {
    fn uri(&self) -> Cow<'_, str>;

    fn filename(&self) -> Option<Cow<'_, str>>;

    fn is_http(&self) -> bool;
}

impl AssetSource for str {
    fn uri(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        // 简单的 URL/路径 分割逻辑
        // 如果是 URL "http://example.com/foo/bar.png" -> "bar.png"
        // 如果是 路径 "assets/bar.png" -> "bar.png"
        let name = self.rsplit('/').next().unwrap_or(self);
        // 处理可能存在的 URL query 参数 "bar.png?v=1" -> "bar.png"
        let name = name.split('?').next().unwrap_or(name);
        if name.is_empty() { None } else { Some(Cow::Borrowed(name)) }
    }

    fn is_http(&self) -> bool {
        self.starts_with("http://") || self.starts_with("https://")
    }
}

impl AssetSource for String {
    fn uri(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        self.as_str().filename()
    }

    fn is_http(&self) -> bool {
        self.as_str().is_http()
    }
}

impl AssetSource for &str {
    fn uri(&self) -> Cow<'_, str> {
        Cow::Borrowed(*self)
    }
    fn filename(&self) -> Option<Cow<'_, str>> {
        (**self).filename()
    }
    fn is_http(&self) -> bool {
        (**self).is_http()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetSource for Path {
    fn uri(&self) -> Cow<'_, str> {
        // 将系统路径转换为通用 URI 格式 (正斜杠)
        Cow::Owned(self.to_string_lossy().replace('\\', "/"))
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        self.file_name().map(|s| s.to_string_lossy())
    }

    fn is_http(&self) -> bool {
        false 
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetSource for &Path {
    fn uri(&self) -> Cow<'_, str> {
        // 将系统路径转换为通用 URI 格式 (正斜杠)
        Cow::Owned(self.to_string_lossy().replace('\\', "/"))
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        self.file_name().map(|s| s.to_string_lossy())
    }

    fn is_http(&self) -> bool {
        false 
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetSource for PathBuf {
    fn uri(&self) -> Cow<'_, str> {
        Cow::Owned(self.to_string_lossy().replace('\\', "/"))
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        self.file_name().map(|s| s.to_string_lossy())
    }

    fn is_http(&self) -> bool {
        false
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetSource for &PathBuf {
    fn uri(&self) -> Cow<'_, str> {
        Cow::Owned(self.to_string_lossy().replace('\\', "/"))
    }

    fn filename(&self) -> Option<Cow<'_, str>> {
        self.file_name().map(|s| s.to_string_lossy())
    }

    fn is_http(&self) -> bool {
        false
    }
}