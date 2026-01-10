pub mod buffer;
pub mod uniforms;
pub mod material;
pub mod geometry;
pub mod node;
pub mod mesh;
pub mod texture;
pub mod camera;
pub mod scene;
pub mod light;
pub mod world;
pub mod assets;

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::ops::{Deref, DerefMut};

pub fn uuid_to_u64(uuid: &uuid::Uuid) -> u64 {
    let mut hasher = DefaultHasher::new();
    uuid.hash(&mut hasher);
    hasher.finish()
}

/// 修改守卫：在 Drop 时自动增加版本号
pub struct Mut<'a, T: ?Sized> {
    data: &'a mut T,
    version: &'a mut u64,
}

impl<'a, T: ?Sized> Deref for Mut<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T: ?Sized> DerefMut for Mut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<'a, T: ?Sized> Drop for Mut<'a, T> {
    fn drop(&mut self) {
        // 使用 wrapping_add 防止溢出 panic
        *self.version = self.version.wrapping_add(1);
    }
}