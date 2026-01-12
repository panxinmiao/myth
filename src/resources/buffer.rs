use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use bytemuck::Pod;

// 全局 Buffer ID 生成器
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

/// 通用 CPU 端数据缓冲
/// 统一管理 Vertex, Index, Uniform, Storage 等数据的 CPU 副本
#[derive(Debug)]
pub struct DataBuffer {
    pub id: u64,
    pub label: String,
    version: AtomicU64,
    data: RwLock<Vec<u8>>,
    pub usage: wgpu::BufferUsages,
}

// === 新增：高性能引用包装器 ===
#[derive(Debug, Clone)]
pub struct BufferRef(Arc<DataBuffer>);

impl PartialEq for BufferRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for BufferRef {}

impl std::hash::Hash for BufferRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

impl BufferRef {
    pub fn new<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        Self(Arc::new(DataBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            version: AtomicU64::new(0), 
            data: RwLock::new(raw_data),
            usage,
        }))
    }

    pub fn from_bytes(data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self(Arc::new(DataBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            version: AtomicU64::new(0), 
            data: RwLock::new(data.to_vec()),
            usage,
        }))
    }

    pub fn empty(usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self(Arc::new(DataBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            version: AtomicU64::new(0), 
            data: RwLock::new(Vec::new()),
            usage,
        }))
    }

    pub fn new_with_capacity(capacity: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let data = vec![0u8; capacity];
        Self(Arc::new(DataBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            version: AtomicU64::new(0), 
            data: RwLock::new(data),
            usage,
        }))
    }

    // === 核心优化：无锁获取版本号 ===
    pub fn version(&self) -> u64 {
        self.0.version.load(Ordering::Relaxed)
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }
    
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.0.usage
    }
    
    pub fn label(&self) -> &str {
        &self.0.label
    }

    // === 数据更新 ===
    pub fn update<T: Pod>(&self, data: &[T]) {
        // 1. 获取写锁
        let mut inner_data = self.0.data.write().unwrap();
        // 2. 写入数据
        *inner_data = bytemuck::cast_slice(data).to_vec();
        // 3. 释放锁 (自动)
        
        // 4. 更新版本 (无锁)
        self.0.version.fetch_add(1, Ordering::Relaxed);
    }
    
    //读取数据的接口
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        self.0.data.read().unwrap()
    }
}

impl std::ops::Deref for BufferRef {
    type Target = DataBuffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}