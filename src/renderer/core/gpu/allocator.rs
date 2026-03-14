//! Model Buffer Allocator
//!
//! Manages per-frame dynamic model uniform allocation. Maintains a CPU-side
//! staging vector and a [`GpuBufferHandle`] pointing directly into the
//! `ResourceManager`'s SlotMap arena, bypassing the `CpuBuffer` + `RwLock`
//! intermediary for zero-overhead GPU uploads.

use std::num::NonZero;

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
    /// Current write position (number of allocated slots this frame).
    cursor: usize,
    /// Allocated entry count (may be larger than cursor).
    capacity: usize,
    /// Handle into the SlotMap GPU buffer arena.
    gpu_handle: Option<GpuBufferHandle>,
    /// Logical buffer id used for BufferRef compatibility.
    buffer_id: u64,
    /// Whether a capacity expansion happened this frame.
    needs_recreate: bool,

    pub(crate) last_ensure_frame: u64,
}

impl ModelBufferAllocator {
    #[must_use]
    pub fn new() -> Self {
        let initial_capacity = 4096;
        let buffer_ref = BufferRef::new(
            initial_capacity * std::mem::size_of::<DynamicModelUniforms>(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );

        Self {
            host_data: Vec::with_capacity(initial_capacity),
            cursor: 0,
            capacity: initial_capacity,
            gpu_handle: None,
            buffer_id: buffer_ref.id,
            needs_recreate: false,
            last_ensure_frame: 0,
        }
    }

    /// Reset at the beginning of each frame.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.host_data.clear();
        self.needs_recreate = false;
    }

    /// Allocate a model uniform slot, returning the byte offset.
    pub fn allocate(&mut self, data: DynamicModelUniforms) -> u32 {
        let index = self.cursor;
        self.cursor += 1;

        if self.cursor > self.capacity {
            self.expand_capacity();
        }

        self.host_data.push(data);

        (index * std::mem::size_of::<DynamicModelUniforms>()) as u32
    }

    /// Pre-ensure capacity for `required_count` items.
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
            self.regenerate_buffer_id();
        }
    }

    fn expand_capacity(&mut self) {
        let new_cap = (self.capacity * 2).max(128);
        log::info!(
            "Model Buffer expanding capacity: {} -> {}",
            self.capacity,
            new_cap
        );

        self.capacity = new_cap;
        self.needs_recreate = true;
        self.regenerate_buffer_id();
    }

    /// Assign a new BufferRef id — called on capacity changes so callers
    /// detect the stale buffer identity.
    fn regenerate_buffer_id(&mut self) {
        let new_ref = BufferRef::new(
            self.capacity * std::mem::size_of::<DynamicModelUniforms>(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer"),
        );
        self.buffer_id = new_ref.id;
        self.gpu_handle = None; // Force re-lookup in ResourceManager
    }

    pub fn need_recreate_buffer(&self) -> bool {
        self.needs_recreate
    }

    /// Build a [`BufferRef`] snapshot for `write_buffer_internal` compatibility.
    pub fn buffer_handle(&self) -> BufferRef {
        BufferRef::with_fixed_id(
            self.buffer_id,
            self.capacity * std::mem::size_of::<DynamicModelUniforms>(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            0,
            Some("GlobalModelBuffer"),
        )
    }

    /// CPU-side buffer id used as the key in the buffer index.
    pub fn buffer_id(&self) -> u64 {
        self.buffer_id
    }

    /// Cached [`GpuBufferHandle`] in the SlotMap arena.
    pub fn gpu_handle(&self) -> Option<GpuBufferHandle> {
        self.gpu_handle
    }

    /// Store the handle after the first GPU allocation.
    pub fn set_gpu_handle(&mut self, handle: GpuBufferHandle) {
        self.gpu_handle = Some(handle);
    }

    /// Raw host data bytes for the used portion of this frame.
    pub fn host_bytes(&self) -> &[u8] {
        let stride = std::mem::size_of::<DynamicModelUniforms>();
        let used_bytes = self.cursor * stride;
        &bytemuck::cast_slice::<DynamicModelUniforms, u8>(&self.host_data)[..used_bytes]
    }

    /// Number of slots written this frame.
    pub fn len(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    /// Byte size of a single dynamic model uniform.
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
