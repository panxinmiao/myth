use rustc_hash::FxHashMap;
use glam::Mat4;
use crate::render::resources::manager::ResourceManager;
use crate::resources::buffer::CpuBuffer;
use crate::scene::SkeletonKey;

pub struct SkeletonManager {
    // 简单起见，目前每个 Skeleton 对应一个独立的 UniformBuffer
    // 进阶优化可以使用一个巨大的 Buffer + Dynamic Offset (类似 ModelBufferManager)
    buffers: FxHashMap<SkeletonKey, CpuBuffer<Vec<Mat4>>>,
}

impl SkeletonManager {
    pub fn new() -> Self {
        Self { buffers: FxHashMap::default() }
    }

    /// 每帧调用：更新 GPU 数据
    pub fn update(&mut self, resource_manager: &mut ResourceManager, skeleton_id: SkeletonKey, matrices: &[Mat4]) {
        // 1. 查找或创建 Buffer
        let buffer = self.buffers.entry(skeleton_id).or_insert_with(|| {
            CpuBuffer::new(
                vec![Mat4::IDENTITY; 64],
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some(&format!("Skeleton"))
            )
        });

        // 2. 更新数据 (如果发生变化)
        // 确保始终保持64个矩阵的大小，未使用的部分保持单位矩阵
        let mut data = buffer.write();
        let len = matrices.len().min(64);
        data[..len].copy_from_slice(&matrices[..len]);
        // 剩余部分保持单位矩阵 (已经初始化为IDENTITY)
        drop(data);

        // 3. 上传到 GPU
        resource_manager.write_buffer(buffer.handle(), buffer.as_bytes());
    }

    /// 获取 GPU Buffer Handle 用于绑定
    pub fn get_buffer(&self, skeleton_id: SkeletonKey) -> Option<&CpuBuffer<Vec<Mat4>>> {
        self.buffers.get(&skeleton_id)
    }
}