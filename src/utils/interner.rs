//! Global String Interner
//!
//! Provides high-performance string interning service that converts strings into
//! integer [`Symbol`]s for efficient comparison and hashing. This is the foundational
//! infrastructure for the dynamic shader macro system.

use std::borrow::Cow;
use lasso::Spur;

#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::Lazy;
#[cfg(not(target_arch = "wasm32"))]
use lasso::ThreadedRodeo;

#[cfg(target_arch = "wasm32")]
use std::cell::UnsafeCell;
#[cfg(target_arch = "wasm32")]
use lasso::Rodeo;

/// Global string interner instance (Native - thread-safe)
#[cfg(not(target_arch = "wasm32"))]
static INTERNER: Lazy<ThreadedRodeo> = Lazy::new(ThreadedRodeo::new);


#[cfg(target_arch = "wasm32")]
// Global string interner instance (WASM - single-threaded)
// 
// We use UnsafeCell here because WASM is single-threaded and RefCell can cause
// double-borrow panics when async callbacks interleave with event handlers.
// Since WASM guarantees single-threaded execution, this is safe.
thread_local! {
    static INTERNER: UnsafeCell<Rodeo> = UnsafeCell::new(Rodeo::new());
}

/// Symbol type alias
///
/// A Symbol is a compact integer identifier that can be efficiently compared
/// and hashed, providing O(1) equality checks for interned strings.
pub type Symbol = Spur;

/// Interns a string and returns its Symbol.
///
/// If the string already exists in the intern pool, returns the existing Symbol.
/// If it doesn't exist, adds it to the pool and returns a new Symbol.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn intern(s: &str) -> Symbol {
    INTERNER.get_or_intern(s)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn intern(s: &str) -> Symbol {
    // SAFETY: WASM is single-threaded, so there's no concurrent access.
    // We use UnsafeCell to avoid RefCell double-borrow panics that occur
    // when async callbacks (spawn_local) interleave with event handlers.
    INTERNER.with(|interner| unsafe { (*interner.get()).get_or_intern(s) })
}

/// Attempts to get the Symbol for an existing string.
///
/// Returns `None` if the string doesn't exist in the intern pool.
/// This method does not allocate new memory.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn get(s: &str) -> Option<Symbol> {
    INTERNER.get(s)
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn get(s: &str) -> Option<Symbol> {
    // SAFETY: WASM is single-threaded
    INTERNER.with(|interner| unsafe { (*interner.get()).get(s) })
}

/// Resolves a Symbol back to its string.
///
/// Returns a String containing the resolved value.
///
/// # Panics
///
/// Panics if the Symbol is invalid (this typically shouldn't happen).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn resolve(sym: Symbol) -> Cow<'static, str> {
    Cow::Borrowed(INTERNER.resolve(&sym))
}

/// Resolves a Symbol back to its string (WASM version).
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn resolve(sym: Symbol) -> Cow<'static, str> {
    INTERNER.with(|interner| {
        unsafe { 
            Cow::Owned((*interner.get()).resolve(&sym).to_string()) 
        }
    })
}

/// Pre-interns commonly used macro names.
///
/// Called during rendering engine initialization to ensure common macro names
/// are already interned, reducing interning operations on hot paths.
pub fn preload_common_macros() {
    let common = [
        // Material-related
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
        // Geometry-related
        "HAS_UV",
        "HAS_NORMAL",
        "HAS_COLOR",
        "HAS_TANGENT",
        "HAS_SKINNING",
        "HAS_MORPH_TARGETS",
        "HAS_MORPH_NORMALS",
        "HAS_MORPH_TANGENTS",
        "SUPPORT_SKINNING",
        // Scene-related
        "HAS_ENV_MAP",
        // Pipeline-related
        "ALPHA_MODE",
        "OPAQUE",
        "MASK",
        "BLEND",
        // Common values
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
