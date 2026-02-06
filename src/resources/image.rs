#[cfg(debug_assertions)]
use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// 全局 Image ID 生成器 (为了高性能 Map 查找，使用 u64)
static NEXT_IMAGE_ID: AtomicU64 = AtomicU64::new(1);

// 把所有可能触发 Re-creation 的元数据打包
#[derive(Debug, Clone, Copy)]
pub struct ImageDescriptor {
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
}

#[derive(Debug)]
pub struct ImageInner {
    pub id: u64,
    pub uuid: Uuid,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,

    pub width: AtomicU32,
    pub height: AtomicU32,
    pub depth: AtomicU32,
    pub mip_level_count: AtomicU32, // 源数据的 mip 层级

    // 格式信息 (Complex types, RwLock)
    pub description: RwLock<ImageDescriptor>,

    // 数据内容 (像素)
    pub data: RwLock<Option<Vec<u8>>>,

    // 版本控制
    pub version: AtomicU64,       // 数据版本 (Data 内容变更时改变)
    pub generation_id: AtomicU64, // 结构版本 (尺寸/格式变更时改变)
}

impl ImageInner {
    pub fn label(&self) -> Option<&str> {
        #[cfg(debug_assertions)]
        {
            Some(&self.label)
        }
        #[cfg(not(debug_assertions))]
        {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image(Arc<ImageInner>);

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}
impl Eq for Image {}
impl std::hash::Hash for Image {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

impl Image {
    #[allow(unused_variables)]
    pub fn new(
        label: Option<&str>,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        data: Option<Vec<u8>>,
    ) -> Self {
        let image_descriptor = ImageDescriptor { dimension, format };
        Self(Arc::new(ImageInner {
            id: NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed),
            uuid: Uuid::new_v4(),
            #[cfg(debug_assertions)]
            label: label.map_or(Cow::Borrowed("Unnamed Image"), |s| {
                Cow::Owned(s.to_string())
            }),

            width: AtomicU32::new(width),
            height: AtomicU32::new(height),
            depth: AtomicU32::new(depth_or_array_layers),
            mip_level_count: AtomicU32::new(1),

            description: RwLock::new(image_descriptor),

            data: RwLock::new(data),
            version: AtomicU64::new(1),
            generation_id: AtomicU64::new(1),
        }))
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.0.id
    }
    #[must_use]
    pub fn uuid(&self) -> Uuid {
        self.0.uuid
    }
    #[must_use]
    pub fn version(&self) -> u64 {
        self.0.version.load(Ordering::Relaxed)
    }
    #[must_use]
    pub fn generation_id(&self) -> u64 {
        self.0.generation_id.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.0.width.load(Ordering::Relaxed)
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        self.0.height.load(Ordering::Relaxed)
    }
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.0.depth.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.0
            .description
            .read()
            .expect("Image descriptor lock poisoned")
            .format
    }
    #[must_use]
    pub fn dimension(&self) -> wgpu::TextureDimension {
        self.0
            .description
            .read()
            .expect("Image descriptor lock poisoned")
            .dimension
    }

    /// 更新数据
    pub fn update_data(&self, data: Vec<u8>) {
        let mut lock = self.0.data.write().expect("Image data lock poisoned");
        *lock = Some(data);
        self.0.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn resize(&self, width: u32, height: u32) {
        let old_w = self.width();
        let old_h = self.height();

        if old_w != width || old_h != height {
            self.0.width.store(width, Ordering::Relaxed);
            self.0.height.store(height, Ordering::Relaxed);
            self.0.generation_id.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn set_format(&self, format: wgpu::TextureFormat) {
        let mut desc = self
            .0
            .description
            .write()
            .expect("Image descriptor lock poisoned");
        if desc.format != format {
            desc.format = format;
            self.0.generation_id.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Deref 方便直接访问内部数据 (只读)
impl std::ops::Deref for Image {
    type Target = ImageInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
