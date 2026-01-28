use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 资产读取器 Trait
/// 支持本地文件和网络资源的异步读取
pub trait AssetReader: Send + Sync {
    /// 异步读取资源字节流
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = anyhow::Result<Vec<u8>>> + Send;
}

/// 本地文件读取器
pub struct FileAssetReader {
    root_path: PathBuf,
}

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

impl AssetReader for FileAssetReader {
    async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        let path = self.root_path.join(uri);
        let data = tokio::fs::read(&path).await?;
        Ok(data)
    }
}

/// HTTP 网络读取器 (条件编译)
#[cfg(feature = "http")]
pub struct HttpAssetReader {
    root_url: reqwest::Url,
    client: reqwest::Client,
}

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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
/// 避免 trait object 的运行时开销
#[derive(Clone)]
pub enum AssetReaderVariant {
    File(Arc<FileAssetReader>),
    #[cfg(feature = "http")]
    Http(Arc<HttpAssetReader>),
}

impl AssetReaderVariant {
    /// 从路径或 URL 自动创建合适的读取器
    pub fn from_source(source: &str) -> anyhow::Result<Self> {
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

    /// 异步读取字节数据
    pub async fn read_bytes(&self, uri: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::File(r) => r.read_bytes(uri).await,
            #[cfg(feature = "http")]
            Self::Http(r) => r.read_bytes(uri).await,
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
