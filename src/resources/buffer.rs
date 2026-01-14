use std::sync::atomic::{AtomicU64, Ordering};
use bytemuck::Pod;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

/// BufferRef: lightweight handle (id, label, usage, size). No CPU data owned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub id: u64,
    pub label: String,
    pub usage: wgpu::BufferUsages,
    pub size: usize, // 记录大小，用于创建时的参考
}

impl BufferRef {
    /// 只是生成 ID 和元数据，不再持有数据
    pub fn new(size: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            usage,
            size,
        }
    }
    
    /// 辅助构造：根据数据长度创建句柄
    pub fn from_data<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = std::mem::size_of_val(data);
        Self::new(size, usage, label)
    }

    /// 辅助构造：从字节数组创建句柄
    pub fn from_bytes(data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(data.len(), usage, label)
    }

    /// 创建空句柄
    pub fn empty(usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(0, usage, label)
    }

    /// 创建指定容量的句柄
    pub fn with_capacity(capacity: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(capacity, usage, label)
    }

    pub fn id(&self) -> u64 { self.id }
    pub fn usage(&self) -> wgpu::BufferUsages { self.usage }
    pub fn label(&self) -> &str { &self.label }
    pub fn size(&self) -> usize { self.size }
}