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
    capacity: usize,
    logical_ref: BufferRef,
    cached_gpu_handle: AtomicU64,
}

impl ModelBufferAllocator {
    #[must_use]
    pub fn new() -> Self {
        let initial_capacity = 4096;
        let logical_ref = BufferRef::empty(
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );

        Self {
            host_data: Vec::with_capacity(initial_capacity),
            capacity: initial_capacity,
            logical_ref,
            cached_gpu_handle: AtomicU64::new(0),
        }
    }

    /// Reset at the beginning of each frame.
    pub fn reset(&mut self) {
        self.host_data.clear();
    }

    pub fn buffer_handle(&self) -> &BufferRef {
        &self.logical_ref
    }

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
    ) -> bool {
        let mut was_resized = false;

        let target_size = (self.capacity * std::mem::size_of::<DynamicModelUniforms>()) as u64;
        let active_data = bytemuck::cast_slice(&self.host_data);

        let handle_bits = self.cached_gpu_handle.load(Ordering::Acquire);

        let handle = if handle_bits == 0 {
            was_resized = true;
            // First: allocate a new GPU buffer and cache the handle
            let mut gpu_buf = GpuBuffer::with_capacity(
                device,
                target_size,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer"),
            );

            // If there is data in the first frame, upload it immediately
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

        // Check if the existing buffer needs to be resized (capacity expansion)
        if handle_bits != 0
            && let Some(gpu_buf) = gpu_buffers.get_mut(handle) {
                if gpu_buf.size < target_size {
                    gpu_buf.resize(device, target_size);
                    was_resized = true;
                }

                if !active_data.is_empty() {
                    queue.write_buffer(&gpu_buf.buffer, 0, active_data);
                }
                gpu_buf.last_used_frame = frame_index;
            }
        was_resized
    }

    pub fn host_data(&self) -> &[DynamicModelUniforms] {
        &self.host_data
    }

    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!(
            "Model Buffer expanding capacity: {} -> {}",
            self.capacity,
            new_cap
        );
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
