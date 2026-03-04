use crate::renderer::core::resources::Tracked;
use wgpu::{Device, TextureView};

use super::types::RdgTextureDesc;

pub(crate) struct PhysicalTexture {
    // texture: wgpu::Texture,
    pub(crate) view: Tracked<wgpu::TextureView>,
    desc: RdgTextureDesc,
}

pub struct RdgTransientPool {
    pub(crate) resources: Vec<PhysicalTexture>,
    // 记录每个物理纹理在当前帧被占用到的 `last_use` 次序
    active_allocations: Vec<usize>,
}

impl RdgTransientPool {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            active_allocations: Vec::new(),
        }
    }

    /// 每帧编译图前调用，重置占用状态，但不销毁实际纹理
    pub fn begin_frame(&mut self) {
        // 将所有物理纹理标记为“在时间 0 之前已结束占用” (也就是全空闲)
        self.active_allocations.fill(0);
    }

    /// 贪心算法分配物理显存
    pub fn acquire(
        &mut self,
        device: &Device,
        desc: &RdgTextureDesc,
        first_use: usize,
        last_use: usize,
    ) -> usize {
        // 1. 尝试寻找可复用的纹理 (Memory Aliasing 魔法就在这里)
        for (i, res) in self.resources.iter().enumerate() {
            // 条件：它的上一次占用结束时间 < 新需求的开始时间，并且规格完全一致
            if self.active_allocations[i] < first_use && res.desc == *desc {
                self.active_allocations[i] = last_use; // 续租
                return i;
            }
        }

        // 2. 如果找不到，只能向 wgpu 申请一张新的
        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("RDG Transient Texture"), // 可以考虑使用传递过来的 name
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        };

        let texture = device.create_texture(&wgpu_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let tracked_view = Tracked::new(view);

        let index = self.resources.len();
        self.resources.push(PhysicalTexture {
            // texture,
            view: tracked_view,
            desc: desc.clone(),
        });

        // 扩容占用表
        self.active_allocations.push(last_use);

        index
    }

    pub fn get_view(&self, index: usize) -> &TextureView {
        &self.resources[index].view
    }
}
