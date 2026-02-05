use std::sync::Arc;
use slotmap::{Key, SlotMap};
use rustc_hash::FxHashMap;
use uuid::Uuid;
use parking_lot::{RwLock, RwLockReadGuard};

// 内部数据结构，被锁保护
pub struct StorageInner<H: Key, T> {
    pub map: SlotMap<H, Arc<T>>, 
    pub lookup: FxHashMap<Uuid, H>,
}

impl<H: Key, T> Default for StorageInner<H, T> {
    fn default() -> Self {
        Self {
            map: SlotMap::default(),
            lookup: FxHashMap::default(),
        }
    }
}

// 对外暴露的线程安全容器
pub struct AssetStorage<H: Key, T> {
    inner: RwLock<StorageInner<H, T>>,
}

impl<H: Key, T> AssetStorage<H, T> {
    pub fn new() -> Self {
        Self {
            inner: RwLock::default(),
        }
    }

    /// [写操作] 添加资源，返回 Handle
    /// 注意：不再需要 &mut self
    pub fn add(&self, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        guard.map.insert(Arc::new(asset.into()))
    }

    /// [写操作] 带 UUID 的添加 (用于文件加载去重)
    pub fn add_with_uuid(&self, uuid: Uuid, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            return handle;
        }
        let handle = guard.map.insert(Arc::new(asset.into()));
        guard.lookup.insert(uuid, handle);
        handle
    }

    /// [读操作] 获取单个资源
    /// 返回 Arc<T>，开销极小
    pub fn get(&self, handle: H) -> Option<Arc<T>> {
        let guard = self.inner.read();
        guard.map.get(handle).cloned()
    }

    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<Arc<T>> {
        let guard = self.inner.read();
        let handle = guard.lookup.get(uuid)?;
        guard.map.get(*handle).cloned()
    }
    
    // 获取 Handle (如果只知道 UUID)
    pub fn get_handle_by_uuid(&self, uuid: &Uuid) -> Option<H> {
        let guard = self.inner.read();
        guard.lookup.get(uuid).cloned()
    }
    
    /// [读操作 - 高级] 获取读锁 Guard
    /// 用于渲染循环中的批量访问，避免多次获取锁
    pub fn read_lock(&self) -> RwLockReadGuard<'_, StorageInner<H, T>> {
        self.inner.read()
    }
}