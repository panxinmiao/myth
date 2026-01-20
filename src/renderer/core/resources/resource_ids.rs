//! 资源 ID 追踪模块
//!
//! 提供轻量级的资源 ID 追踪机制，用于检测 GPU 资源变化。
//! 
//! # 设计思路
//! 
//! 1. **EnsureResult**: ensure 操作的返回值，包含物理资源 ID
//! 2. **ResourceIdSet**: 一组资源 ID 的集合，支持高效比较
//! 3. **BindGroupFingerprint**: BindGroup 的完整指纹，包含所有依赖资源的 ID

use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;
use smallvec::SmallVec;

/// GPU 资源的唯一标识符
/// 
/// 当 GPU 资源被重建（如 Buffer 扩容、Texture 重新创建）时，ID 会变化
pub type ResourceId = u64;

/// Ensure 操作的结果
/// 
/// 包含资源的物理 ID，调用者可以用来判断资源是否发生变化
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnsureResult {
    /// GPU 资源的物理 ID
    pub resource_id: ResourceId,
    /// 资源是否刚刚被创建或重建
    pub was_recreated: bool,
}

impl EnsureResult {
    #[inline]
    pub fn new(resource_id: ResourceId, was_recreated: bool) -> Self {
        Self { resource_id, was_recreated }
    }

    #[inline]
    pub fn existing(resource_id: ResourceId) -> Self {
        Self { resource_id, was_recreated: false }
    }

    #[inline]
    pub fn created(resource_id: ResourceId) -> Self {
        Self { resource_id, was_recreated: true }
    }
}

/// 资源 ID 集合
/// 
/// 用于追踪一组资源的物理 ID，支持快速比较是否发生变化
#[derive(Debug, Clone, Default)]
pub struct ResourceIdSet {
    /// 按添加顺序存储的资源 ID
    ids: SmallVec<[ResourceId; 16]>,
    /// 预计算的哈希值（用于快速比较）
    cached_hash: u64,
    /// 标记哈希是否需要重新计算
    hash_dirty: bool,
}

impl ResourceIdSet {
    pub fn new() -> Self {
        Self {
            ids: SmallVec::new(),
            cached_hash: 0,
            hash_dirty: true,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: SmallVec::with_capacity(capacity),
            cached_hash: 0,
            hash_dirty: true,
        }
    }

    /// 添加一个资源 ID
    #[inline]
    pub fn push(&mut self, id: ResourceId) {
        self.ids.push(id);
        self.hash_dirty = true;
    }

    /// 添加一个可选的资源 ID
    #[inline]
    pub fn push_optional(&mut self, id: Option<ResourceId>) {
        // 使用特殊值表示 None
        self.ids.push(id.unwrap_or(u64::MAX));
        self.hash_dirty = true;
    }

    /// 清空集合
    #[inline]
    pub fn clear(&mut self) {
        self.ids.clear();
        self.hash_dirty = true;
    }

    /// 获取资源 ID 数量
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 获取所有 ID 的切片
    #[inline]
    pub fn as_slice(&self) -> &[ResourceId] {
        &self.ids
    }

    /// 计算并缓存哈希值
    fn compute_hash(&mut self) {
        if !self.hash_dirty {
            return;
        }

        let mut hasher = FxHasher::default();
        self.ids.len().hash(&mut hasher);
        for id in &self.ids {
            id.hash(&mut hasher);
        }
        self.cached_hash = hasher.finish();
        self.hash_dirty = false;
    }

    /// 获取哈希值（用于快速比较）
    #[inline]
    pub fn hash_value(&mut self) -> u64 {
        self.compute_hash();
        self.cached_hash
    }

    /// 比较两个集合是否相同
    pub fn matches(&mut self, other: &mut ResourceIdSet) -> bool {
        if self.ids.len() != other.ids.len() {
            return false;
        }
        
        // 先比较哈希（快速路径）
        if self.hash_value() != other.hash_value() {
            return false;
        }
        
        // 哈希相同时再逐个比较（处理碰撞）
        self.ids == other.ids
    }
}

impl PartialEq for ResourceIdSet {
    fn eq(&self, other: &Self) -> bool {
        self.ids == other.ids
    }
}

impl Eq for ResourceIdSet {}

impl Hash for ResourceIdSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ids.hash(state);
    }
}

/// BindGroup 的完整指纹
/// 
/// 包含 BindGroup 依赖的所有物理资源 ID，用于判断是否需要重建
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupFingerprint {
    /// 所有依赖资源的 ID（按绑定顺序）
    pub resource_ids: Vec<ResourceId>,
    /// Layout entries 的哈希值
    pub layout_hash: u64,
}

impl BindGroupFingerprint {
    pub fn new(resource_ids: Vec<ResourceId>, layout_hash: u64) -> Self {
        Self { resource_ids, layout_hash }
    }

    /// 检查资源 ID 是否发生变化
    pub fn resources_changed(&self, new_ids: &[ResourceId]) -> bool {
        self.resource_ids != new_ids
    }

    /// 检查 Layout 是否发生变化
    pub fn layout_changed(&self, new_layout_hash: u64) -> bool {
        self.layout_hash != new_layout_hash
    }
}

/// 计算 BindGroupLayoutEntry 列表的哈希值
/// 
/// 利用 wgpu::BindGroupLayoutEntry 实现的 Hash trait
pub fn hash_layout_entries(entries: &[wgpu::BindGroupLayoutEntry]) -> u64 {
    let mut hasher = FxHasher::default();
    entries.len().hash(&mut hasher);
    for entry in entries {
        entry.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_id_set_equality() {
        let mut set1 = ResourceIdSet::new();
        set1.push(1);
        set1.push(2);
        set1.push(3);

        let mut set2 = ResourceIdSet::new();
        set2.push(1);
        set2.push(2);
        set2.push(3);

        assert!(set1.matches(&mut set2));
    }

    #[test]
    fn test_resource_id_set_difference() {
        let mut set1 = ResourceIdSet::new();
        set1.push(1);
        set1.push(2);

        let mut set2 = ResourceIdSet::new();
        set2.push(1);
        set2.push(3); // 不同

        assert!(!set1.matches(&mut set2));
    }
}
