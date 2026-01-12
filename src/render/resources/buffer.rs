// src/render/resources/buffer.rs
use wgpu::util::DeviceExt;
use crate::render::resources::manager::generate_resource_id;

/// GPU Buffer 抽象
/// 统一管理 Vertex, Index, Uniform, Storage Buffer 的生命周期和数据更新
pub struct GpuBuffer {
    pub id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,           // 当前 GPU 分配的大小
    pub usage: wgpu::BufferUsages,
    pub label: String,
    
    pub last_used_frame: u64, // 用于 GC

    // --- 状态追踪 ---
    pub version: u64,        // 数据版本 (用于基于版本的更新，如 Geometry)
    
    // --- 影子副本 (用于 Diff 更新) ---
    // 仅对于 Uniform Buffer 或小型动态 Buffer 开启，用于避免重复上传相同数据
    shadow_data: Option<Vec<u8>>,
}

impl GpuBuffer {
    pub fn new(
        device: &wgpu::Device, 
        data: &[u8], 
        usage: wgpu::BufferUsages, 
        label: Option<&str>
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });

        // 自动生成 ID
        let id = generate_resource_id();

        Self {
            id,
            buffer,
            size: data.len() as u64,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            shadow_data: None,
        }
    }

    /// 启用影子副本 (开启 Diff 模式)
    /// 通常用于 Material Uniforms 等频繁更新但数据量小的 Buffer
    pub fn enable_shadow_copy(&mut self) {
        if self.shadow_data.is_none() {
            self.shadow_data = Some(Vec::new());
        }
    }

    /// 【Diff 模式更新】
    /// 比较数据内容，仅当数据变化时上传
    /// 适合: UniformBuffer
    pub fn update_with_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool{
        // 1. Diff 检查
        if let Some(prev) = &mut self.shadow_data {
            if prev == data {
                return false; // 数据完全一致，跳过
            }
            // 更新影子
            // 如果容量不够，重新分配
            if prev.len() != data.len() {
                *prev = vec![0u8; data.len()];
            }
            prev.copy_from_slice(data);
        }

        // 2. 写入 GPU
        self.write_to_gpu(device, queue, data)
    }

    /// 【Version 模式更新】
    /// 比较外部传入的版本号
    /// 适合: VertexBuffer, IndexBuffer (数据量大，memcmp 开销大)
    /// 返回值: bool, 表示是否发生了 Resize (需要重建 BindGroup 或 更新引用)
    pub fn update_with_version(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], new_version: u64) -> bool {
        if new_version <= self.version {
            return false;
        }
        
        self.version = new_version;
        self.write_to_gpu(device, queue, data)
    }

    // 内部写入逻辑：包含自动 Resize
    // 返回 true 如果发生了 Resize
    fn write_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let new_size = data.len() as u64;
        
        // 自动扩容 (Resize)
        if new_size > self.size {
            self.resize(device, new_size);
            queue.write_buffer(&self.buffer, 0, data);
            return true;
        }

        queue.write_buffer(&self.buffer, 0, data);
        false
    }

    fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        // log::info!("Resizing buffer {} from {} to {}", self.label, self.size, new_size);
        self.buffer.destroy(); // 显式销毁旧资源
        
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.size = new_size;

        // 关键：重建 Buffer 后，物理资源变了，必须更新 ID！
        self.id = generate_resource_id();
    }
}