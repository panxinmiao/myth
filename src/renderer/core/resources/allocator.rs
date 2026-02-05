//! Model Buffer Allocator
//!
//! 纯逻辑结构，不持有 wgpu 资源，只管理字节和索引
//! 每帧动态分配 Model Uniform 的 Offset

use std::num::NonZero;

use crate::resources::buffer::{BufferRef, CpuBuffer};
use crate::resources::uniforms::DynamicModelUniforms;

/// Model Buffer 分配器
///
/// 负责管理 `DynamicModelUniforms` 的 CPU 端缓存和分配
pub struct ModelBufferAllocator {
    /// CPU 端数据缓存
    host_data: Vec<DynamicModelUniforms>,
    /// 当前帧写到的位置
    cursor: usize,
    /// Buffer 容量
    capacity: usize,
    /// CPU Buffer 句柄
    buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    /// 标记是否需要重建 GPU Buffer
    needs_recreate: bool,

    pub(crate) last_ensure_frame: u64,
}

impl ModelBufferAllocator {
    /// 创建新的分配器
    #[must_use]
    pub fn new() -> Self {
        let initial_capacity = 4096;
        let initial_data = vec![DynamicModelUniforms::default(); initial_capacity];
        let buffer = CpuBuffer::new(
            initial_data.clone(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );

        Self {
            host_data: Vec::with_capacity(initial_capacity),
            cursor: 0,
            capacity: initial_capacity,
            buffer,
            needs_recreate: false,
            last_ensure_frame: 0,
        }
    }

    /// 每帧开始时重置
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.host_data.clear();
        self.needs_recreate = false;
    }

    /// 分配一个 Model Uniform，返回字节偏移量
    pub fn allocate(&mut self, data: DynamicModelUniforms) -> u32 {
        let index = self.cursor;
        self.cursor += 1;

        // 检查是否需要扩容
        if self.cursor > self.capacity {
            self.expand_capacity();
        }

        // self.host_data.push(data);
        // self.buffer.write()[index] = data;
        self.host_data.push(data);

        // 返回字节偏移量
        (index * std::mem::size_of::<DynamicModelUniforms>()) as u32
    }

    /// 将 `host_data` 同步到 `CpuBuffer`
    pub fn flush_to_buffer(&mut self) {
        if self.host_data.is_empty() {
            return;
        }

        // 这一帧只获取一次锁/借用，进行批量拷贝
        let mut buffer_write = self.buffer.write();
        let len = self.host_data.len();
        // 确保 buffer 足够大 (expand_capacity 应该已经处理了，但为了安全)
        if buffer_write.len() < len {
            // 理论上不应发生，因为 allocate 会扩容，但 CpuBuffer 内部可能需要 resize
            // 这里因为我们重建了 CpuBuffer，所以它是同步的
        }
        buffer_write[..len].copy_from_slice(&self.host_data);
    }

    /// 扩容
    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!(
            "Model Buffer expanding capacity: {} -> {}",
            self.capacity,
            new_cap
        );

        self.capacity = new_cap;
        self.needs_recreate = true;

        // 重建 CpuBuffer
        let new_data = vec![DynamicModelUniforms::default(); new_cap];
        self.buffer = CpuBuffer::new(
            new_data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );
    }

    pub fn need_recreate_buffer(&self) -> bool {
        self.needs_recreate
    }

    // 预先确保容量
    pub fn ensure_capacity(&mut self, required_count: usize) {
        if required_count > self.capacity {
            let mut new_cap = self.capacity;
            while new_cap < required_count {
                new_cap *= 2;
            }
            new_cap = new_cap.max(128);

            log::info!(
                "Model Buffer resizing to fit {} items: {} -> {}",
                required_count,
                self.capacity,
                new_cap
            );

            self.capacity = new_cap;
            self.needs_recreate = true;

            // 重建 CpuBuffer
            let new_data = vec![DynamicModelUniforms::default(); new_cap];
            self.buffer = CpuBuffer::new(
                new_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer"),
            );
        }
    }

    /// 获取 Buffer 句柄
    pub fn buffer_handle(&self) -> BufferRef {
        self.buffer.handle()
    }

    /// 获取 Buffer ID
    pub fn buffer_id(&self) -> u64 {
        self.buffer.handle().id
    }

    /// 获取当前帧的数据量
    pub fn len(&self) -> usize {
        self.cursor
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    /// 获取 `CpuBuffer` 引用（用于构建 `BindGroup`）
    pub fn cpu_buffer(&self) -> &CpuBuffer<Vec<DynamicModelUniforms>> {
        &self.buffer
    }

    /// 获取动态 uniform 的字节大小
    pub fn uniform_stride() -> NonZero<u64> {
        std::mem::size_of::<DynamicModelUniforms>()
            .try_into()
            .ok()
            .and_then(NonZero::new)
            .expect("DynamicModelUniforms size should be non-zero")
    }
}

impl Default for ModelBufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}
