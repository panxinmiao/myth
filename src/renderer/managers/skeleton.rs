//! 骨骼动画管理器
//!
//! 管理骨骼动画的 GPU 资源

use rustc_hash::FxHashMap;
use glam::Mat4;

use crate::renderer::core::resources::ResourceManager;
use crate::resources::buffer::CpuBuffer;
use crate::scene::SkeletonKey;

pub struct SkeletonManager {
    buffers: FxHashMap<SkeletonKey, CpuBuffer<Vec<Mat4>>>,
}

impl Default for SkeletonManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SkeletonManager {
    pub fn new() -> Self {
        Self { buffers: FxHashMap::default() }
    }

    pub fn update(&mut self, resource_manager: &mut ResourceManager, skeleton_id: SkeletonKey, matrices: &[Mat4]) {
        let buffer = self.buffers.entry(skeleton_id).or_insert_with(|| {
            CpuBuffer::new(
                matrices.to_vec(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Skeleton")
            )
        });

        *buffer.write() = matrices.to_vec();
        resource_manager.write_buffer(buffer.handle(), buffer.as_bytes());
    }

    pub fn get_buffer(&self, skeleton_id: SkeletonKey) -> Option<&CpuBuffer<Vec<Mat4>>> {
        self.buffers.get(&skeleton_id)
    }
}
