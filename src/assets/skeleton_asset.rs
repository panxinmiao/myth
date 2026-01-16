use glam::{Affine3A};
use uuid::Uuid;

pub struct SkeletonAsset {
    pub id: Uuid,
    pub name: String,
    pub bone_names: Vec<String>,
    pub parent_indices: Vec<i16>,
    pub inverse_bind_matrices: Vec<Affine3A>,
}