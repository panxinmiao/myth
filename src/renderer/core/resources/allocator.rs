//! Model Buffer Allocator
//!
//! A pure logical structure that does not hold wgpu resources; it only manages bytes and indices.
//! Dynamically allocates Model Uniform offsets each frame.

use std::num::NonZero;

use crate::resources::buffer::{BufferRef, CpuBuffer};
use crate::resources::uniforms::DynamicModelUniforms;

/// Model Buffer allocator
///
/// Manages the CPU-side cache and allocation of `DynamicModelUniforms`
pub struct ModelBufferAllocator {
    /// CPU-side data cache
    host_data: Vec<DynamicModelUniforms>,
    /// Current write position for this frame
    cursor: usize,
    /// Buffer capacity
    capacity: usize,
    /// CPU Buffer handle
    buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    /// Flag indicating whether the GPU Buffer needs recreation
    needs_recreate: bool,

    pub(crate) last_ensure_frame: u64,
}

impl ModelBufferAllocator {
    /// Create a new allocator
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

    /// Reset at the beginning of each frame
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.host_data.clear();
        self.needs_recreate = false;
    }

    /// Allocate a Model Uniform slot, returning the byte offset
    pub fn allocate(&mut self, data: DynamicModelUniforms) -> u32 {
        let index = self.cursor;
        self.cursor += 1;

        // Check if expansion is needed
        if self.cursor > self.capacity {
            self.expand_capacity();
        }

        self.host_data.push(data);

        // Return byte offset
        (index * std::mem::size_of::<DynamicModelUniforms>()) as u32
    }

    /// Flush `host_data` to the `CpuBuffer`
    pub fn flush_to_buffer(&mut self) {
        if self.host_data.is_empty() {
            return;
        }

        // Only acquire the lock/borrow once this frame for batch copy
        let mut buffer_write = self.buffer.write();
        let len = self.host_data.len();
        // Ensure buffer is large enough (expand_capacity should have handled this, but for safety)
        if buffer_write.len() < len {
            // Theoretically should not happen since allocate expands capacity,
            // but CpuBuffer internals may need resizing.
            // Since we've rebuilt CpuBuffer, this is synchronized.
        }
        buffer_write[..len].copy_from_slice(&self.host_data);
    }

    /// Expand capacity
    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!(
            "Model Buffer expanding capacity: {} -> {}",
            self.capacity,
            new_cap
        );

        self.capacity = new_cap;
        self.needs_recreate = true;

        // Rebuild CpuBuffer
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

    // Pre-ensure capacity
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

            // Rebuild CpuBuffer
            let new_data = vec![DynamicModelUniforms::default(); new_cap];
            self.buffer = CpuBuffer::new(
                new_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer"),
            );
        }
    }

    /// Get the Buffer handle
    pub fn buffer_handle(&self) -> BufferRef {
        self.buffer.handle()
    }

    /// Get the Buffer ID
    pub fn buffer_id(&self) -> u64 {
        self.buffer.handle().id
    }

    /// Get the data count for the current frame
    pub fn len(&self) -> usize {
        self.cursor
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    /// Get a reference to the `CpuBuffer` (used for building `BindGroup`)
    pub fn cpu_buffer(&self) -> &CpuBuffer<Vec<DynamicModelUniforms>> {
        &self.buffer
    }

    /// Get the byte size of a dynamic uniform
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
