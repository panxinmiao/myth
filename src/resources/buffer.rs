use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(debug_assertions)]
use std::borrow::Cow;
use bytemuck::Pod;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

pub trait GpuData {
    fn as_bytes(&self) -> &[u8];
    fn byte_size(&self) -> usize;
}

impl<T: Pod> GpuData for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
    
    fn byte_size(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }
}

/// BufferRef: lightweight handle (id, label, usage, size). No CPU data owned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub id: u64,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,
    pub usage: wgpu::BufferUsages,
    pub size: usize, // 记录大小，用于创建时的参考
    pub version: u64, // 版本号，用于追踪内容变更
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
            version: 0,
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

pub struct BufferGuard<'a, T: GpuData> {
    buffer: &'a mut CpuBuffer<T>,
}

impl<'a, T: GpuData> std::ops::Deref for BufferGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.buffer.data
    }
}

impl<'a, T: GpuData> std::ops::DerefMut for BufferGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer.data
    }
}

impl<'a, T: GpuData> Drop for BufferGuard<'a, T> {
    fn drop(&mut self) {
        self.buffer.buffer.version = self.buffer.buffer.version.wrapping_add(1);
        
        let new_size = self.buffer.data.byte_size();
        if new_size != self.buffer.buffer.size {
            self.buffer.buffer.size = new_size;
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuBuffer<T: GpuData> {
    data: T,
    pub buffer: BufferRef,
}

impl<T: GpuData> CpuBuffer<T> {
    pub fn new(data: T, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = data.byte_size();
        let mut buffer = BufferRef::new(size, usage, label);
        buffer.version = 0;
        Self { data, buffer }
    }

    pub fn default() -> Self 
    where T: Default 
    {
        Self::new(T::default(), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, None)
    }

    pub fn new_uniform(label: Option<&str>) -> Self 
    where T: Default 
    {
        Self::new(T::default(), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, label)
    }

    pub fn new_storage(label: Option<&str>) -> Self 
    where T: Default 
    {
        Self::new(T::default(), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, label)
    }

    pub fn read(&self) -> &T {
        &self.data
    }

    pub fn write(&mut self) -> BufferGuard<'_, T> {
        BufferGuard { buffer: self }
    }
    
    pub fn handle(&self) -> &BufferRef {
        &self.buffer
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

impl<T: GpuData> std::ops::Deref for CpuBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: GpuData> GpuData for CpuBuffer<T> {
    fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
    
    fn byte_size(&self) -> usize {
        self.data.byte_size()
    }
}