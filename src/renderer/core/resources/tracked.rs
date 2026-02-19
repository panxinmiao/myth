use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

/// 全局唯一 ID 生成器
static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// 带有唯一 ID 的资源包装器
/// 用于给 `FrameResources` 或临时资源赋予身份标识
#[derive(Debug, Clone)]
pub struct Tracked<T> {
    inner: T,
    id: u64,
}

impl<T> Tracked<T> {
    /// 包装一个资源并分配新 ID
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            id: next_id(),
        }
    }

    /// 获取唯一 ID (作为 `BindGroup` Cache 的 Key)
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 解包获取内部资源
    pub fn into_inner(self) -> T {
        self.inner
    }
}

// 方便直接访问内部方法 (如 texture_view.format())
impl<T> Deref for Tracked<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
