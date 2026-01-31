use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(debug_assertions)]
use std::borrow::Cow;
use bytemuck::Pod;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

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

    /// 创建具有指定 ID 的句柄（用于全局资源管理）
    /// 
    /// 注意：此方法允许指定固定的 ID，用于需要稳定引用的全局资源。
    /// 普通使用场景应使用 `new()` 方法让系统自动分配 ID。
    pub fn with_fixed_id(id: u64, size: usize, usage: wgpu::BufferUsages, version: u64, _label: Option<&str>) -> Self {
        Self {
            id,
            #[cfg(debug_assertions)]
            label: _label
                .map(|s| Cow::Owned(s.to_string()))
                .unwrap_or(Cow::Borrowed("Unnamed Buffer")),
            usage,
            size,
            version,
        }
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
    guard: RwLockWriteGuard<'a, CpuBufferState<T>>,
    pub(crate) changed: bool,
}

impl<'a, T: GpuData> BufferGuard<'a, T> {
    // 允许用户手动取消版本更新
    pub fn skip_sync(&mut self) {
        self.changed = false;
    }
}

impl<'a, T: GpuData> std::ops::Deref for BufferGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard.data
    }
}

impl<'a, T: GpuData> std::ops::DerefMut for BufferGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.data
    }
}

impl<'a, T: GpuData> Drop for BufferGuard<'a, T> {
    fn drop(&mut self) {
        if self.changed {
            self.guard.version = self.guard.version.wrapping_add(1);
            self.guard.size = self.guard.data.byte_size();
        }
    }
}


/// 内部可变状态：只有这些数据需要被锁保护
#[derive(Debug)]
struct CpuBufferState<T: GpuData> {
    data: T,
    version: u64,
    size: usize,
}

#[derive(Debug)]
pub struct CpuBuffer<T: GpuData> {
    // 1. 不可变元数据 (Immutable Metadata) - 放在锁外，支持无锁访问
    id: u64,
    usage: wgpu::BufferUsages,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,

    // 2. 可变状态 (Mutable State) - 放在锁内
    inner: RwLock<CpuBufferState<T>>,
}

impl<T: GpuData + Clone> Clone for CpuBuffer<T> {
    fn clone(&self) -> Self {
        // 1. 获取读锁，拿到当前数据的引用
        let guard = self.inner.read();
        // 2. 构造新的 CpuBuffer，克隆数据
        Self::new(
            guard.data.clone(), 
            self.usage, 
            self.label() // 复用 Label
        )
    }
}

impl<T: GpuData> CpuBuffer<T> {
    pub fn new(data: T, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = data.byte_size();

        // 先创建 BufferRef 主要是为了复用它的 ID 生成逻辑
        let base_ref = BufferRef::new(size, usage, label);

        Self {
            // 提取不可变元数据到外层
            id: base_ref.id,
            usage: base_ref.usage,
            #[cfg(debug_assertions)]
            label: base_ref.label,

            // 初始化内部可变状态
            inner: RwLock::new(CpuBufferState {
                data,
                version: 0,
                size,
            }),
        }

        // let mut buffer = BufferRef::new(size, usage, label);
        // buffer.version = 0;
        // Self { data, buffer }
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

    // pub fn read(&self) -> &T {
    //     &self.data
    // }

    // pub fn write(&mut self) -> BufferGuard<'_, T> {
    //     BufferGuard { buffer: self }
    // }
    
    // pub fn handle(&self) -> &BufferRef {
    //     &self.buffer
    // }

    // === 不需要锁的操作 ===
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    pub fn label(&self) -> Option<&str> {
        #[cfg(debug_assertions)]
        { Some(&self.label) }
        #[cfg(not(debug_assertions))]
        { None }
    }

    // === 需要锁的操作 ===

    pub fn size(&self) -> usize {
        self.inner.read().size
    }

    pub fn version(&self) -> u64 {
        self.inner.read().version
    }

    /// 获取数据读锁
    /// 注意：返回的 Guard 只能访问 data
    pub fn read(&self) -> BufferReadGuard<'_, T> {
        BufferReadGuard {
            guard: self.inner.read()
        }
    }

    // 为了让 deref 能够工作，我们需要一个小技巧或者让用户手动访问 .data
    // 鉴于 CpuBufferState 对外不可见，建议让用户通过 read().data 访问，或者自定义 ReadGuard

    pub fn write(&self) -> BufferGuard<'_, T> {
        BufferGuard {
            guard: self.inner.write(),
            changed: true,
        }
    }

    /// 获取当前的 BufferRef 快照
    /// 注意：这里由返回 `&BufferRef` 改为了返回 `BufferRef` (值类型)
    /// 因为版本号在锁里，我们必须现场构造一个新的 BufferRef
    pub fn handle(&self) -> BufferRef {
        let state = self.inner.read();
        BufferRef {
            id: self.id,
            #[cfg(debug_assertions)]
            label: self.label.clone(),
            usage: self.usage,
            size: state.size,
            version: state.version,
        }
    }

    // pub fn as_bytes(&self) -> &[u8] {
    //     self.data.as_bytes()
    // }
}

pub struct BufferReadGuard<'a, T: GpuData> {
    guard: RwLockReadGuard<'a, CpuBufferState<T>>,
}

impl<'a, T: GpuData> std::ops::Deref for BufferReadGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard.data
    }
}

// impl<T: GpuData> std::ops::Deref for CpuBuffer<T> {
//     type Target = T;
//     fn deref(&self) -> &Self::Target {
//         &self.data
//     }
// }

// impl<T: GpuData> GpuData for CpuBuffer<T> {
//     fn as_bytes(&self) -> &[u8] {
//         self.data.as_bytes()
//     }
    
//     fn byte_size(&self) -> usize {
//         self.data.byte_size()
//     }
// }