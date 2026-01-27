//! 全局字符串驻留器 (String Interner)
//!
//! 提供高性能的字符串驻留服务，将字符串转换为整数 Symbol 进行比较和哈希。
//! 这是动态宏系统的基础设施。

use lasso::{ThreadedRodeo, Spur};
use once_cell::sync::Lazy;

/// 全局字符串驻留器实例
static INTERNER: Lazy<ThreadedRodeo> = Lazy::new(ThreadedRodeo::new);

/// Symbol 类型别名
/// 
/// Symbol 是一个紧凑的整数标识符，可以高效地进行比较和哈希操作。
pub type Symbol = Spur;

/// 驻留一个字符串，返回其 Symbol
/// 
/// 如果字符串已存在于驻留池中，返回已有的 Symbol。
/// 如果不存在，将其添加到驻留池并返回新的 Symbol。
#[inline]
pub fn intern(s: &str) -> Symbol {
    INTERNER.get_or_intern(s)
}

/// 尝试获取已存在字符串的 Symbol
/// 
/// 如果字符串不存在于驻留池中，返回 None。
/// 这个方法不会分配新内存。
#[inline]
pub fn get(s: &str) -> Option<Symbol> {
    INTERNER.get(s)
}

/// 将 Symbol 解析回字符串
/// 
/// 返回驻留池中对应的字符串引用。
/// 
/// # Panics
/// 如果 Symbol 无效（通常不会发生），会 panic。
#[inline]
pub fn resolve(sym: Symbol) -> &'static str {
    INTERNER.resolve(&sym)
}

/// 预驻留常用宏名称
/// 
/// 在渲染引擎初始化时调用，确保常用的宏名称已被驻留，
/// 减少热路径上的驻留操作。
pub fn preload_common_macros() {
    let common = [
        // 材质相关
        "USE_IBL",
        "USE_IOR",
        "USE_SPECULAR",
        "USE_CLEARCOAT",
        "HAS_MAP",
        "HAS_NORMAL_MAP",
        "HAS_ROUGHNESS_MAP",
        "HAS_METALNESS_MAP",
        "HAS_EMISSIVE_MAP",
        "HAS_AO_MAP",
        "HAS_SPECULAR_MAP",
        "HAS_SPECULAR_INTENSITY_MAP",
        "HAS_CLEARCOAT_MAP",
        "HAS_CLEARCOAT_ROUGHNESS_MAP",
        "HAS_CLEARCOAT_NORMAL_MAP",
        // 几何体相关
        "HAS_UV",
        "HAS_NORMAL",
        "HAS_VERTEX_COLOR",
        "HAS_TANGENT",
        "HAS_SKINNING",
        "HAS_MORPH_TARGETS",
        "HAS_MORPH_NORMALS",
        "HAS_MORPH_TANGENTS",
        // 场景相关
        "HAS_ENV_MAP",
        // 管线相关
        "ALPHA_MODE",
        "OPAQUE",
        "MASK",
        "BLEND",
        // 常用值
        "1",
        "true",
    ];
    
    for name in common {
        intern(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_and_resolve() {
        let s1 = intern("hello");
        let s2 = intern("hello");
        let s3 = intern("world");
        
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        
        assert_eq!(resolve(s1), "hello");
        assert_eq!(resolve(s3), "world");
    }

    #[test]
    fn test_get() {
        let _ = intern("existing");
        
        assert!(get("existing").is_some());
        assert!(get("non_existing").is_none());
    }
}
