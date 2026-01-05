use crate::renderer::uniforms::DynamicModelUniforms;

/// 管理动态 Uniform Buffer 的自动扩容和上传
pub struct DynamicBuffer {
    label: String,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
    
    capacity: usize, // 当前 Buffer 能容纳的物体数量
}

impl DynamicBuffer {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        // 1. 创建 Layout (has_dynamic_offset = true)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} Layout", label)),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true, // <--- 关键：开启动态偏移
                    min_binding_size: wgpu::BufferSize::new(256), // 单次绑定的视窗大小
                },
                count: None,
            }],
        });

        // 2. 初始容量 (例如 128 个物体)
        let initial_capacity = 128;
        let (buffer, bind_group) = Self::create_resources(device, label, &layout, initial_capacity);

        Self {
            label: label.to_string(),
            buffer,
            bind_group,
            layout,
            capacity: initial_capacity,
        }
    }

    /// 每一帧调用：上传数据，如果容量不足则自动扩容
    pub fn write_and_expand(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[DynamicModelUniforms]) {
        if data.is_empty() {
            return;
        }

        let required_count = data.len();

        // 1. 检查是否需要扩容
        if required_count > self.capacity {
            // 扩容策略：2 倍增长，避免频繁重分配
            let new_capacity = (self.capacity * 2).max(required_count);
            log::info!("Expanding DynamicBuffer '{}' capacity: {} -> {}", self.label, self.capacity, new_capacity);

            let (new_buffer, new_bg) = Self::create_resources(device, &self.label, &self.layout, new_capacity);
            
            self.buffer = new_buffer;
            self.bind_group = new_bg;
            self.capacity = new_capacity;
        }

        // 2. 写入数据 (直接覆盖 Buffer)
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    // 内部辅助：创建 Buffer 和 BindGroup
    fn create_resources(
        device: &wgpu::Device, 
        label: &str, 
        layout: &wgpu::BindGroupLayout, 
        count: usize
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let alignment = 256;
        let size = (count * alignment) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Buffer", label)),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} BindGroup", label)),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(256),
                }),
            }],
        });

        (buffer, bind_group)
    }
}