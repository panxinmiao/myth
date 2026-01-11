use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

// 全局 Image ID 生成器 (为了高性能 Map 查找，使用 u64)
static NEXT_IMAGE_ID: AtomicU64 = AtomicU64::new(0);


// 把所有可能触发 Re-creation 的元数据打包
#[derive(Debug, Clone, Copy)]
pub struct ImageDescriptor {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
}

#[derive(Debug)]
pub struct ImageInner {
    pub id: u64, // 使用 u64 替代 Uuid 以提升渲染层性能
    pub uuid: Uuid, // 保留 UUID 用于资产序列化/反序列化
    pub label: String,

    // 元数据
    pub descriptor: RwLock<ImageDescriptor>,
    // 数据内容 (像素)
    pub data: RwLock<Option<Vec<u8>>>,
    
    // 版本控制
    pub version: AtomicU64,  // 数据版本 (Data 内容变更时改变)
    pub generation_id: AtomicU64, // 结构版本 (尺寸/格式变更时改变)
}

#[derive(Debug, Clone)]
pub struct Image(Arc<ImageInner>);

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool { self.0.id == other.0.id }
}
impl Eq for Image {}
impl std::hash::Hash for Image {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.0.id.hash(state); }
}

impl Image {
    pub fn new(
        label: &str,
        width: u32, 
        height: u32, 
        depth_or_array_layers: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat, 
        data: Option<Vec<u8>>
    ) -> Self {
        let image_descriptor = ImageDescriptor {
            width,
            height,
            depth_or_array_layers,
            dimension,
            format,
            mip_level_count: 1, // 暂简化
        };
        Self(Arc::new(ImageInner {
            id: NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed),
            uuid: Uuid::new_v4(),
            label: label.to_string(),
            descriptor: RwLock::new(image_descriptor),
            data: RwLock::new(data),
            version: AtomicU64::new(1),
            generation_id: AtomicU64::new(1),
        }))
    }

    pub fn id(&self) -> u64 { self.0.id }
    pub fn uuid(&self) -> Uuid { self.0.uuid }
    pub fn version(&self) -> u64 { self.0.version.load(Ordering::Relaxed) }
    pub fn generation_id(&self) -> u64 { self.0.generation_id.load(Ordering::Relaxed) }

    /// 更新数据
    pub fn update_data(&self, data: Vec<u8>) {
        let mut lock = self.0.data.write().unwrap();
        *lock = Some(data);
        self.0.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn resize(&self, width: u32, height: u32) {
        let mut desc = self.0.descriptor.write().unwrap();
        
        // 1. 检查是否真的变了 (避免无谓的 GPU 重建)
        if desc.width == width && desc.height == height {
            return;
        }

        // 2. 更新 CPU 端数据
        desc.width = width;
        desc.height = height;

        // 3. 标记结构脏数据 -> 触发 GpuImage 重建 -> 触发 GpuTexture 重建
        self.0.generation_id.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_format(&self, format: wgpu::TextureFormat) {
        let mut desc = self.0.descriptor.write().unwrap();
        if desc.format != format {
            desc.format = format;
            self.0.generation_id.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Deref 方便直接访问内部数据 (只读)
impl std::ops::Deref for Image {
    type Target = ImageInner;
    fn deref(&self) -> &Self::Target { &self.0 }
}