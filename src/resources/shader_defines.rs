//! Shader 宏定义系统
//!
//! 提供统一的、高性能的 Shader 宏定义管理。
//! 基于字符串驻留 (Interning) + 哈希缓存实现 O(1) 的宏列表比较。

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::utils::interner::{self, Symbol};

/// Shader 宏定义集合
/// 
/// 内部使用有序的 `Vec<(Symbol, Symbol)>` 存储宏定义，
/// 确保相同的宏集合产生相同的哈希值。
#[derive(Debug, Clone, Default)]
pub struct ShaderDefines {
    defines: Vec<(Symbol, Symbol)>,
}

impl ShaderDefines {
    /// 创建空的宏定义集合
    #[inline]
    pub fn new() -> Self {
        Self {
            defines: Vec::new(),
        }
    }

    /// 创建具有预分配容量的宏定义集合
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            defines: Vec::with_capacity(capacity),
        }
    }

    /// 设置宏定义（自动保持排序）
    /// 
    /// 如果 key 已存在，更新其 value；否则插入新条目。
    pub fn set(&mut self, key: &str, value: &str) {
        let key_sym = interner::intern(key);
        let value_sym = interner::intern(value);
        self.set_symbol(key_sym, value_sym);
    }

    /// 使用 Symbol 设置宏定义（内部方法，更高效）
    #[inline]
    pub fn set_symbol(&mut self, key: Symbol, value: Symbol) {
        match self.defines.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(idx) => {
                self.defines[idx].1 = value;
            }
            Err(idx) => {
                self.defines.insert(idx, (key, value));
            }
        }
    }

    /// 移除宏定义
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(key_sym) = interner::get(key) {
            self.remove_symbol(key_sym)
        } else {
            false
        }
    }

    /// 使用 Symbol 移除宏定义
    #[inline]
    pub fn remove_symbol(&mut self, key: Symbol) -> bool {
        if let Ok(idx) = self.defines.binary_search_by_key(&key, |&(k, _)| k) {
            self.defines.remove(idx);
            true
        } else {
            false
        }
    }

    /// 检查是否包含某个宏定义
    pub fn contains(&self, key: &str) -> bool {
        interner::get(key).map_or(false, |key_sym| self.contains_symbol(key_sym))
    }

    /// 使用 Symbol 检查是否包含某个宏定义
    #[inline]
    pub fn contains_symbol(&self, key: Symbol) -> bool {
        self.defines.binary_search_by_key(&key, |&(k, _)| k).is_ok()
    }

    /// 获取宏定义的值
    pub fn get(&self, key: &str) -> Option<&str> {
        interner::get(key).and_then(|key_sym| self.get_symbol(key_sym))
    }

    /// 使用 Symbol 获取宏定义的值
    #[inline]
    pub fn get_symbol(&self, key: Symbol) -> Option<&'static str> {
        self.defines
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|idx| interner::resolve(self.defines[idx].1))
    }

    /// 清空所有宏定义
    #[inline]
    pub fn clear(&mut self) {
        self.defines.clear();
    }

    /// 获取宏定义数量
    #[inline]
    pub fn len(&self) -> usize {
        self.defines.len()
    }

    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.defines.is_empty()
    }

    /// 迭代所有宏定义 (以 Symbol 形式)
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(Symbol, Symbol)> {
        self.defines.iter()
    }

    /// 迭代所有宏定义 (以字符串形式)
    #[inline]
    pub fn iter_strings(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.defines.iter().map(|&(k, v)| (interner::resolve(k), interner::resolve(v)))
    }

    /// 转换为 BTreeMap (用于模板渲染)
    pub fn to_map(&self) -> BTreeMap<String, String> {
        self.defines
            .iter()
            .map(|&(k, v)| (interner::resolve(k).to_string(), interner::resolve(v).to_string()))
            .collect()
    }

    /// 合并另一个 ShaderDefines 的宏定义
    /// 
    /// 如果有冲突，other 中的值会覆盖 self 中的值。
    pub fn merge(&mut self, other: &ShaderDefines) {
        for &(key, value) in &other.defines {
            self.set_symbol(key, value);
        }
    }

    /// 创建合并后的新 ShaderDefines
    pub fn merged_with(&self, other: &ShaderDefines) -> ShaderDefines {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// 计算内容哈希（用于缓存）
    pub fn compute_hash(&self) -> u64 {
        use std::hash::BuildHasher;
        let mut hasher = rustc_hash::FxBuildHasher.build_hasher();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// 获取内部的 defines 引用（用于直接访问）
    #[inline]
    pub fn as_slice(&self) -> &[(Symbol, Symbol)] {
        &self.defines
    }
}

impl Hash for ShaderDefines {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.defines.hash(state);
    }
}

impl PartialEq for ShaderDefines {
    fn eq(&self, other: &Self) -> bool {
        self.defines == other.defines
    }
}

impl Eq for ShaderDefines {}

/// 从宏定义列表创建 ShaderDefines
impl From<&[(&str, &str)]> for ShaderDefines {
    fn from(defines: &[(&str, &str)]) -> Self {
        let mut result = Self::with_capacity(defines.len());
        for (k, v) in defines {
            result.set(k, v);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut defines = ShaderDefines::new();
        defines.set("USE_MAP", "1");
        defines.set("USE_NORMAL_MAP", "1");
        
        assert!(defines.contains("USE_MAP"));
        assert!(defines.contains("USE_NORMAL_MAP"));
        assert!(!defines.contains("USE_AO_MAP"));
        
        assert_eq!(defines.get("USE_MAP"), Some("1"));
    }

    #[test]
    fn test_ordering() {
        let mut defines = ShaderDefines::new();
        defines.set("B", "1");
        defines.set("A", "1");
        defines.set("C", "1");
        
        // 验证内部按 Symbol (整数ID) 排序，保证哈希一致性
        // 注意：Symbol 顺序取决于 intern 顺序，不是字符串字典序
        let symbols: Vec<_> = defines.iter().map(|(k, _)| k).collect();
        assert!(symbols.windows(2).all(|w| w[0] < w[1]), "Symbols should be sorted by numeric value");
        
        // 验证所有 key 都存在
        assert!(defines.contains("A"));
        assert!(defines.contains("B"));
        assert!(defines.contains("C"));
    }

    #[test]
    fn test_merge() {
        let mut d1 = ShaderDefines::new();
        d1.set("A", "1");
        d1.set("B", "2");
        
        let mut d2 = ShaderDefines::new();
        d2.set("B", "3");
        d2.set("C", "4");
        
        d1.merge(&d2);
        
        assert_eq!(d1.get("A"), Some("1"));
        assert_eq!(d1.get("B"), Some("3")); // 被覆盖
        assert_eq!(d1.get("C"), Some("4"));
    }

    #[test]
    fn test_hash_consistency() {
        let mut d1 = ShaderDefines::new();
        d1.set("A", "1");
        d1.set("B", "2");
        
        let mut d2 = ShaderDefines::new();
        d2.set("B", "2");
        d2.set("A", "1");
        
        assert_eq!(d1.compute_hash(), d2.compute_hash());
    }
}
