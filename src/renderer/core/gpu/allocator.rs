//! Model Buffer Allocator
//!
//! Manages per-frame dynamic model uniform allocation. Maintains a CPU-side
//! staging vector and a [`GpuBufferHandle`] pointing directly into the
//! `ResourceManager`'s SlotMap arena, bypassing the `CpuBuffer` + `RwLock`
//! intermediary for zero-overhead GPU uploads.

use std::num::NonZero;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::renderer::core::gpu::GpuBuffer;
use crate::resources::buffer::BufferRef;
use crate::resources::uniforms::DynamicModelUniforms;

use super::buffer::GpuBufferHandle;

/// Model Buffer allocator.
///
/// Each frame the sequence is:
/// 1. [`reset()`]  — clear the staging vector
/// 2. [`allocate()`] — push uniforms and return byte offsets
/// 3. Caller uploads via `ResourceManager::upload_model_buffer()`
pub struct ModelBufferAllocator {
    /// CPU-side staging data for the current frame.
    host_data: Vec<DynamicModelUniforms>,
    // /// Current write position (number of allocated slots this frame).
    // cursor: usize,
    /// Allocated entry count (may be larger than cursor).
    // capacity: usize,
    // /// Handle into the SlotMap GPU buffer arena.
    // gpu_handle: Option<GpuBufferHandle>,
    // /// Logical buffer id used for BufferRef compatibility.
    // buffer_id: u64,
    // /// Whether a capacity expansion happened this frame.
    // needs_recreate: bool,

    // pub(crate) last_ensure_frame: u64,

    /// 容量（元素个数）
    capacity: usize,
    /// 稳定的逻辑 ID，用于向 RenderGraph 提供绑定的 Key
    logical_ref: BufferRef,
    /// 缓存的 GPU 句柄
    cached_gpu_handle: AtomicU64,
    
    // pub(crate) last_ensure_frame: u64,
}

impl ModelBufferAllocator {
    #[must_use]
    pub fn new() -> Self {
        let initial_capacity = 4096;
        let logical_ref = BufferRef::empty(
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            Some("GlobalModelBuffer")
        );

        Self {
            host_data: Vec::with_capacity(initial_capacity),
            // cursor: 0,
            capacity: initial_capacity,
            // gpu_handle: None,
            // buffer_id: buffer_ref.id,
            // needs_recreate: false,
            logical_ref,
            cached_gpu_handle: AtomicU64::new(0),
            // last_ensure_frame: 0,
        }
    }

    /// Reset at the beginning of each frame.
    pub fn reset(&mut self) {
        // self.cursor = 0;
        // self.host_data.clear();
        // self.needs_recreate = false;

        self.host_data.clear();
    }

    pub fn buffer_handle(&self) -> &BufferRef {
        &self.logical_ref
    }

    // pub fn uniform_stride() -> u64 {
    //     std::mem::size_of::<DynamicModelUniforms>() as u64
    // }

    pub fn uniform_stride() -> NonZero<u64> {
        std::mem::size_of::<DynamicModelUniforms>()
            .try_into()
            .ok()
            .and_then(NonZero::new)
            .expect("DynamicModelUniforms size should be non-zero")
    }

    /// Allocate a model uniform slot, returning the byte offset.
    pub fn allocate(&mut self, data: DynamicModelUniforms) -> u32 {
        // let index = self.cursor;
        // self.cursor += 1;

        // if self.cursor > self.capacity {
        //     self.expand_capacity();
        // }

        // self.host_data.push(data);

        // (index * std::mem::size_of::<DynamicModelUniforms>()) as u32

        let index = self.host_data.len();
        
        if index >= self.capacity {
            self.expand_capacity();
        }

        self.host_data.push(data);
        (index * std::mem::size_of::<DynamicModelUniforms>()) as u32
    }


    pub fn flush_to_buffer(
        &self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_buffers: &mut slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: &mut rustc_hash::FxHashMap<u64, GpuBufferHandle>,
        frame_index: u64,
    ) {
        // 目标物理大小由 Capacity 决定，而不是当前帧的元素个数！
        let target_size = (self.capacity * std::mem::size_of::<DynamicModelUniforms>()) as u64;
        let active_data = bytemuck::cast_slice(&self.host_data); // 仅有用的数据

        let handle_bits = self.cached_gpu_handle.load(Ordering::Acquire);
        
        let handle = if handle_bits == 0 {
            // 第一次：按 capacity 分配物理缓冲
            let mut gpu_buf = GpuBuffer::with_capacity(
                &device,
                target_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer"),
            );
            
            // 如果第一帧就有数据，执行上传
            if !active_data.is_empty() {
                queue.write_buffer(&gpu_buf.buffer, 0, active_data);
            }
            
            gpu_buf.last_used_frame = frame_index;
            let h = gpu_buffers.insert(gpu_buf);
            buffer_index.insert(self.logical_ref.id(), h);
            self.cached_gpu_handle.store(h.to_bits(), Ordering::Release);
            h
        } else {
            GpuBufferHandle::from_bits(handle_bits).unwrap()
        };

        // 如果资源已经存在，检查是否需要 Resize，然后只上传有用的 active_data
        if handle_bits != 0 {
            if let Some(gpu_buf) = gpu_buffers.get_mut(handle) {
                // 如果容量扩张了，重建底层的 wgpu::Buffer
                if gpu_buf.size < target_size {
                    gpu_buf.resize(device, target_size);
                }
                
                // 极致性能点：只通过 PCIe 发送实际 push 的那部分字节
                if !active_data.is_empty() {
                    queue.write_buffer(&gpu_buf.buffer, 0, active_data);
                }
                gpu_buf.last_used_frame = frame_index;
            }
        }
    }

    pub fn host_data(&self) -> &[DynamicModelUniforms] {
        &self.host_data
    }

    /// Pre-ensure capacity for `required_count` items.
    // pub fn ensure_capacity(&mut self, required_count: usize) {
    //     if required_count > self.capacity {
    //         let mut new_cap = self.capacity;
    //         while new_cap < required_count {
    //             new_cap *= 2;
    //         }
    //         new_cap = new_cap.max(128);

    //         log::info!(
    //             "Model Buffer resizing to fit {} items: {} -> {}",
    //             required_count,
    //             self.capacity,
    //             new_cap
    //         );

    //         self.capacity = new_cap;
    //         self.needs_recreate = true;
    //         self.regenerate_buffer_id();
    //     }
    // }

    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!("Model Buffer expanding capacity: {} -> {}", self.capacity, new_cap);
        self.capacity = new_cap;
    }

    pub fn len(&self) -> usize {
        self.host_data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.host_data.is_empty()
    }

}

impl Default for ModelBufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}
