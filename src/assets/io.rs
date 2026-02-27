#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::{borrow::Cow, sync::Arc};

use crate::errors::{AssetError, Error, Result};

/// Asset reader trait.
/// Supports asynchronous reading of local files and network resources.
pub trait AssetReader: Send + Sync {
    /// Asynchronously read resource bytes.
    #[cfg(not(target_arch = "wasm32"))]
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;

    /// On WASM, Futures do not need to be `Send`.
    #[cfg(target_arch = "wasm32")]
    fn read_bytes(&self, uri: &str) -> impl std::future::Future<Output = Result<Vec<u8>>>;
}

/// Local file reader (only available on non-WASM platforms).
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
    #[must_use]
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetReader for FileAssetReader {
    async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>> {
        let path = self.root_path.join(uri);
        let data = tokio::fs::read(&path).await?;
        Ok(data)
    }
}

/// HTTP network reader.
/// Uses reqwest's cross-platform capability (tokio on native, fetch on WASM).
#[cfg(feature = "http")]
pub struct HttpAssetReader {
    root_url: reqwest::Url,
    client: reqwest::Client,
}

#[cfg(feature = "http")]
impl HttpAssetReader {
    pub fn new(url_str: &str) -> Result<Self> {
        let url = reqwest::Url::parse(url_str)?;
        let root_url = if url.path().ends_with('/') {
            url
        } else {
            let mut u = url.clone();
            if let Ok(mut segments) = u.path_segments_mut() {
                segments.pop();
                segments.push("");
            }
            u
        };

        let client = reqwest::Client::builder();

        // Native-specific timeout settings, etc.
        #[cfg(not(target_arch = "wasm32"))]
        let client = client.timeout(std::time::Duration::from_secs(30));

        Ok(Self {
            root_url,
            client: client.build()?,
        })
    }

    #[inline]
    #[must_use]
    pub fn root_url(&self) -> &reqwest::Url {
        &self.root_url
    }
}

#[cfg(feature = "http")]
impl AssetReader for HttpAssetReader {
    async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>> {
        let url = self.root_url.join(uri)?;
        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Err(Error::Asset(AssetError::HttpResponse {
                status: resp.status().as_u16(),
            }));
        }
        let bytes = resp.bytes().await?;
        Ok(bytes.to_vec())
    }
}

/// Asset reader variant enum.
#[derive(Clone)]
pub enum AssetReaderVariant {
    #[cfg(not(target_arch = "wasm32"))]
    File(Arc<FileAssetReader>),
    #[cfg(feature = "http")]
    Http(Arc<HttpAssetReader>),
}

impl AssetReaderVariant {
    /// Automatically creates the appropriate reader from a path or URL.
    pub fn new(source: &impl AssetSource) -> Result<Self> {
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
                    Err(Error::Platform(
                        crate::errors::PlatformError::FeatureNotEnabled("http".to_string()),
                    ))
                }
            } else {
                // If not HTTP, default to filesystem reader
                Ok(Self::File(Arc::new(FileAssetReader::new(uri.as_ref()))))
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM always uses HTTP reader
            #[cfg(not(feature = "http"))]
            {
                return Err(Error::Platform(
                    crate::errors::PlatformError::FeatureNotEnabled("http".to_string()),
                ));
            }

            let full_uri = if !source.is_http() {
                let window = web_sys::window().ok_or_else(|| {
                    Error::Platform(crate::errors::PlatformError::Wasm(
                        "No window found".to_string(),
                    ))
                })?;

                let location = window.location();
                let href = location.href().map_err(|e| {
                    Error::Platform(crate::errors::PlatformError::Wasm(format!(
                        "Failed to get href: {:?}",
                        e
                    )))
                })?;

                let base = reqwest::Url::parse(&href)?;

                base.join(&uri)?.to_string()
            } else {
                uri.to_string()
            };

            Ok(Self::Http(Arc::new(HttpAssetReader::new(&full_uri)?)))
        }
    }

    /// Asynchronously reads byte data.
    pub async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::File(r) => r.read_bytes(uri).await,
            #[cfg(feature = "http")]
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
        // Simple URL/path splitting logic
        // For URL "http://example.com/foo/bar.png" -> "bar.png"
        // For path "assets/bar.png" -> "bar.png"
        let name = self.rsplit('/').next().unwrap_or(self);
        // Handle possible URL query parameters "bar.png?v=1" -> "bar.png"
        let name = name.split('?').next().unwrap_or(name);
        if name.is_empty() {
            None
        } else {
            Some(Cow::Borrowed(name))
        }
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
        // Convert system path to generic URI format (forward slashes)
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
        // Convert system path to generic URI format (forward slashes)
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
