// src/resources/uniform_slot.rs
//! 轻量级 Uniform 数据容器
//! 
//! 专为小型 Pod 类型的 Uniform 数据设计，直接持有数据而不需要 Vec<u8> 中间层。
//! 相比 BufferRef，UniformSlot 具有：
//! - 零拷贝（数据内联存储）
//! - 无锁访问（仅版本号使用原子操作）
//! - 类型安全（泛型约束）

use bytemuck::Pod;
use std::sync::atomic::{AtomicU64, Ordering};
use std::ops::{Deref, DerefMut};

static NEXT_UNIFORM_SLOT_ID: AtomicU64 = AtomicU64::new(1 << 60); // 从大数字开始，避免和BufferID冲突

/// 轻量级 Uniform 数据容器
/// 
/// # 示例
/// ```
/// use three::resources::UniformSlot;
/// use three::resources::uniforms::MeshBasicUniforms;
/// 
/// let mut slot = UniformSlot::new(
///     MeshBasicUniforms::default(),
///     "BasicMaterial"
/// );
/// 
/// // 读取
/// let color = slot.get().color;
/// 
/// // 修改（自动标记dirty）
/// slot.get_mut().color = Vec4::new(1.0, 0.0, 0.0, 1.0);
/// slot.mark_dirty();
/// ```
#[derive(Debug)]
pub struct UniformSlot<T: Pod> {
    id: u64,
    data: T,
    version: AtomicU64,
    label: String,
}

impl<T: Pod> UniformSlot<T> {
    /// 创建新的 Uniform Slot
    pub fn new(data: T, label: &str) -> Self {
        Self {
            id: NEXT_UNIFORM_SLOT_ID.fetch_add(1, Ordering::Relaxed),
            data,
            version: AtomicU64::new(0),
            label: label.to_string(),
        }
    }
    
    /// 获取数据的不可变引用
    #[inline]
    pub fn get(&self) -> &T {
        &self.data
    }
    
    /// 替换整个数据（自动递增版本号）
    pub fn set(&mut self, data: T) {
        self.data = data;
        self.version.fetch_add(1, Ordering::Relaxed);
    }
    
    /// 获取数据的可变引用
    /// 
    /// 注意：修改后需要手动调用 `mark_dirty()` 来递增版本号
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
    
    /// 手动标记为脏数据（递增版本号）
    /// 
    /// 当使用 `get_mut()` 修改数据后，需要调用此方法
    #[inline]
    pub fn mark_dirty(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
    
    /// 获取当前版本号
    #[inline]
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
    
    /// 获取唯一 ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }
    
    /// 获取 label
    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }
    
    /// 获取数据的字节表示
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(&self.data)
    }
}

impl<T: Pod + Default> Default for UniformSlot<T> {
    fn default() -> Self {
        Self::new(T::default(), "UniformSlot")
    }
}

impl<T: Pod + Clone> Clone for UniformSlot<T> {
    fn clone(&self) -> Self {
        Self {
            id: NEXT_UNIFORM_SLOT_ID.fetch_add(1, Ordering::Relaxed),
            data: self.data.clone(),
            version: AtomicU64::new(self.version.load(Ordering::Relaxed)),
            label: self.label.clone(),
        }
    }
}

impl<T: Pod> Deref for UniformSlot<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// 2. 实现 DerefMut：写入时自动标记 dirty
impl<T: Pod> DerefMut for UniformSlot<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mark_dirty();
        &mut self.data
    }
}

