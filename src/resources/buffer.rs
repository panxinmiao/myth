use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(debug_assertions)]
use std::borrow::Cow;
use bytemuck::Pod;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

/// BufferRef: lightweight handle (id, label, usage, size). No CPU data owned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub id: u64,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,
    pub usage: wgpu::BufferUsages,
    pub size: usize, // 记录大小，用于创建时的参考
}

impl BufferRef {
    pub fn new(size: usize, usage: wgpu::BufferUsages, _label: Option<&str>) -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            #[cfg(debug_assertions)]
            label: _label
                .map(|s| Cow::Owned(s.to_string()))
                .unwrap_or(Cow::Borrowed("Unnamed Buffer")),
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
    pub fn size(&self) -> usize { self.size }
}