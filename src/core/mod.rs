pub mod binding;
pub mod buffer;
pub mod uniforms;
pub mod material;
pub mod geometry;
pub mod node;
pub mod mesh;
pub mod texture;
pub mod camera;
pub mod scene;

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub fn uuid_to_u64(uuid: &uuid::Uuid) -> u64 {
    let mut hasher = DefaultHasher::new();
    uuid.hash(&mut hasher);
    hasher.finish()
}