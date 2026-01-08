use uuid::Uuid;
use std::sync::{Arc, RwLock};
use bytemuck::Pod;

/// 通用 CPU 端数据缓冲
/// 统一管理 Vertex, Index, Uniform, Storage 等数据的 CPU 副本
#[derive(Debug)]
pub struct DataBuffer {
    pub id: Uuid,
    pub label: String,
    pub data: Vec<u8>,
    pub version: u64,
    pub usage: wgpu::BufferUsages,
}

impl DataBuffer {
    pub fn new<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: label.unwrap_or("Buffer").to_string(),
            data: bytemuck::cast_slice(data).to_vec(),
            version: 0,
            usage,
        }
    }

    /// 更新数据并增加版本号
    pub fn update<T: Pod>(&mut self, data: &[T]) {
        self.data = bytemuck::cast_slice(data).to_vec();
        self.version = self.version.wrapping_add(1);
    }
}

/// 线程安全的 Buffer 引用
// pub type BufferRef = Arc<RwLock<DataBuffer>>;

// === 新增：高性能引用包装器 ===
#[derive(Debug, Clone)]
pub struct BufferRef {
    pub id: Uuid,
    inner: Arc<RwLock<DataBuffer>>,
}

impl PartialEq for BufferRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for BufferRef {}

impl std::hash::Hash for BufferRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // 只 Hash ID。
        // 如果 ID 相同，意味着这是同一个 Buffer 引用，
        // BindGroup 就不需要重建。
        self.id.hash(state);
    }
}

impl BufferRef {
    pub fn new(buffer: DataBuffer) -> Self {
        Self {
            id: buffer.id,
            inner: Arc::new(RwLock::new(buffer)),
        }
    }

    // 转发 read/write
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, DataBuffer> {
        self.inner.read().unwrap()
    }

    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, DataBuffer> {
        self.inner.write().unwrap()
    }
}