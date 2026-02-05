//! 带状态追踪的渲染通道
//!
//! 避免冗余的状态切换调用
//!

// 用于替代 Vec<u32> 的结构体，避免堆内存分配
#[derive(Clone, Copy, PartialEq)]
struct BindGroupState {
    id: u64,
    // wgpu 的动态偏移量通常很少（限制一般是 8 或 4），用固定数组足够
    offsets: [u32; 8],
    offset_count: u8,
}

pub struct TrackedRenderPass<'a> {
    pass: wgpu::RenderPass<'a>,
    current_pipeline_id: Option<u16>,
    current_bind_groups: [Option<BindGroupState>; 4],
    current_vertex_buffers: [Option<u64>; 8],
    current_index_buffer: Option<u64>,
}

impl<'a> TrackedRenderPass<'a> {
    #[must_use]
    pub fn new(pass: wgpu::RenderPass<'a>) -> Self {
        Self {
            pass,
            current_pipeline_id: None,
            current_bind_groups: [None; 4],
            current_vertex_buffers: [None; 8],
            current_index_buffer: None,
        }
    }

    pub fn set_pipeline(&mut self, pipeline_resource_id: u16, pipeline: &'a wgpu::RenderPipeline) {
        if self.current_pipeline_id != Some(pipeline_resource_id) {
            self.pass.set_pipeline(pipeline);
            self.current_pipeline_id = Some(pipeline_resource_id);
        }
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group_resource_id: u64,
        bind_group: &'a wgpu::BindGroup,
        offsets: &[u32],
    ) {
        let slot = index as usize;
        let needs_update = if let Some(state) = &self.current_bind_groups[slot] {
            if state.id != bind_group_resource_id {
                true
            } else if state.offset_count as usize != offsets.len() {
                true
            } else {
                &state.offsets[..offsets.len()] != offsets
            }
        } else {
            true
        };

        if needs_update {
            self.pass.set_bind_group(index, bind_group, offsets);

            let mut state = BindGroupState {
                id: bind_group_resource_id,
                offsets: [0; 8],
                offset_count: offsets.len() as u8,
            };
            let len = offsets.len().min(8);
            state.offsets[..len].copy_from_slice(&offsets[..len]);

            self.current_bind_groups[slot] = Some(state);
        }
    }

    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer_resource_id: u64,
        buffer_slice: wgpu::BufferSlice<'a>,
    ) {
        let index = slot as usize;
        if self.current_vertex_buffers[index] != Some(buffer_resource_id) {
            self.pass.set_vertex_buffer(slot, buffer_slice);
            self.current_vertex_buffers[index] = Some(buffer_resource_id);
        }
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_resource_id: u64,
        buffer_slice: wgpu::BufferSlice<'a>,
        format: wgpu::IndexFormat,
    ) {
        if self.current_index_buffer != Some(buffer_resource_id) {
            self.pass.set_index_buffer(buffer_slice, format);
            self.current_index_buffer = Some(buffer_resource_id);
        }
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.pass.draw(vertices, instances);
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        self.pass.draw_indexed(indices, base_vertex, instances);
    }
}
