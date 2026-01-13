/// 版本追踪器 - 用于标记资源变化
#[derive(Debug, Clone, Copy, Default)]
pub struct ChangeTracker {
    version: u64,
}

impl ChangeTracker {
    pub fn new() -> Self {
        Self { version: 0 }
    }

    /// 标记为已修改，版本号+1
    pub fn changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
    
    /// 获取当前版本号
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// 可变守卫 - 当作用域结束时自动更新版本号
pub struct MutGuard<'a, T> {
    data: &'a mut T,
    version: &'a mut u64,
}

impl<'a, T> MutGuard<'a, T> {
    pub fn new(data: &'a mut T, version: &'a mut u64) -> Self {
        Self { data, version }
    }
}

impl<'a, T> std::ops::Deref for MutGuard<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> std::ops::DerefMut for MutGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

// 关键：当 Guard 销毁时，自动增加版本号
impl<'a, T> Drop for MutGuard<'a, T> {
    fn drop(&mut self) {
        *self.version = self.version.wrapping_add(1);
    }
}
