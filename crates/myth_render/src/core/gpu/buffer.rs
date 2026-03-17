//! GPU buffer storage and lifecycle management.
//!
//! # Architecture
//!
//! GPU buffers are stored in a [`SlotMap`] keyed by [`GpuBufferHandle`], providing
//! O(1) insertion, removal, and lookup by handle. A secondary index
//! (`buffer_index: FxHashMap<u64, GpuBufferHandle>`) maps CPU-side buffer IDs to
//! their corresponding slot, bridging the gap for code paths that only have a
//! `BufferRef` or raw `u64` id.
//!
//! [`CpuBuffer`] caches its assigned [`GpuBufferHandle`] in an `AtomicU64` field,
//! enabling a lock-free O(1) fast path in [`ResourceManager::ensure_buffer`].
//!
//! # Resize Strategy
//!
//! When incoming data exceeds the current `wgpu::Buffer` capacity, the old buffer
//! is destroyed and a new, larger one is created **in the same SlotMap slot**. The
//! handle remains stable, but `GpuBuffer::id` is regenerated to signal downstream
//! consumers (e.g. `ResourceIdSet` fingerprints) that the physical resource has
//! changed and dependent `BindGroup`s must be rebuilt.

use slotmap::SlotMap;

use super::{EnsureResult, ResourceManager, generate_gpu_resource_id};
use myth_resources::buffer::BufferRef;

// ────────────────────────────────────────────────────────────────────────────
// Handle — imported from `myth_resources` to ensure CpuBuffer←→SlotMap
// identity agreement.
// ────────────────────────────────────────────────────────────────────────────

pub use myth_resources::GpuBufferHandle;

// ────────────────────────────────────────────────────────────────────────────
// GpuBuffer
// ────────────────────────────────────────────────────────────────────────────

/// A GPU-resident buffer managed by [`ResourceManager`].
///
/// The `id` field is a *physical* resource identity: it changes whenever the
/// underlying `wgpu::Buffer` is recreated (e.g. on resize), allowing
/// `ResourceIdSet` fingerprints to detect the change and trigger `BindGroup`
/// rebuilds.
pub struct GpuBuffer {
    pub id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub label: String,
    pub last_used_frame: u64,
    pub version: u64,
    pub last_uploaded_version: u64,
}

impl GpuBuffer {
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });

        Self {
            id: generate_gpu_resource_id(),
            buffer,
            size: data.len() as u64,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
        }
    }

    pub fn with_capacity(
        device: &wgpu::Device,
        capacity_bytes: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        let min_size = if usage.contains(wgpu::BufferUsages::UNIFORM) {
            256
        } else {
            16
        };
        let size = capacity_bytes.max(min_size);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            id: generate_gpu_resource_id(),
            buffer,
            size,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
        }
    }

    /// Write `data` to the GPU, resizing the buffer in-place if necessary.
    ///
    /// Returns `true` when the physical `wgpu::Buffer` was recreated (callers
    /// must rebuild any `BindGroup`s that reference this buffer).
    pub(crate) fn write_to_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> bool {
        let new_size = data.len() as u64;
        if new_size > self.size {
            self.resize(device, new_size);
            queue.write_buffer(&self.buffer, 0, data);
            return true;
        }
        queue.write_buffer(&self.buffer, 0, data);
        false
    }

    /// Destroy the current `wgpu::Buffer` and allocate a larger one.
    ///
    /// The handle in the SlotMap is unchanged; only `id` is regenerated to
    /// propagate the physical-resource change through the fingerprint system.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        self.buffer.destroy();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.size = new_size;
        self.id = generate_gpu_resource_id();
    }
}

impl ResourceManager {
    // ────────────────────────────────────────────────────────────────────────
    // Internal write helper (borrows split fields to satisfy borrow-checker)
    // ────────────────────────────────────────────────────────────────────────

    /// Upload `data` for the buffer identified by `buffer_ref`, creating or
    /// resizing the GPU-side buffer as needed.
    ///
    /// This is a **static method** that borrows only the fields it touches,
    /// allowing callers to hold references to other `ResourceManager` members
    /// concurrently (e.g. `model_allocator`).
    pub fn write_buffer_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_buffers: &mut SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: &mut rustc_hash::FxHashMap<u64, GpuBufferHandle>,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        let cpu_id = buffer_ref.id();

        if let Some(&handle) = buffer_index.get(&cpu_id) {
            if let Some(gpu_buf) = gpu_buffers.get_mut(handle) {
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let old_id = gpu_buf.id;
                    was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                    if !was_recreated && gpu_buf.id != old_id {
                        was_recreated = true;
                    }
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            // Stale handle — slot was freed. Remove from index and fall through.
            buffer_index.remove(&cpu_id);
        }

        // First encounter: create a new GPU buffer.
        let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
        buf.last_uploaded_version = buffer_ref.version;
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = gpu_buffers.insert(buf);
        buffer_index.insert(cpu_id, handle);
        (handle, EnsureResult::created(phys_id))
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public ensure_buffer family
    // ────────────────────────────────────────────────────────────────────────

    /// Ensure the GPU buffer for a [`CpuBuffer`] exists and contains the
    /// latest data.
    ///
    /// Uses the `CpuBuffer`'s internal atomic handle cache for an O(1) fast
    /// path on subsequent calls (no hash lookup required).
    pub fn ensure_buffer<T: super::GpuData>(
        &mut self,
        cpu_buffer: &super::CpuBuffer<T>,
    ) -> (GpuBufferHandle, EnsureResult) {
        // ── Fast path: CpuBuffer already knows its slot ────────────
        if let Some(handle) = cpu_buffer.gpu_handle() {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let buffer_ref = cpu_buffer.handle();
                let mut was_recreated = false;

                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    let guard = cpu_buffer.read();
                    let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());
                    was_recreated = gpu_buf.write_to_gpu(&self.device, &self.queue, data);
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = self.frame_index;
                return (handle, EnsureResult::new(gpu_buf.id, was_recreated));
            }
            // Handle went stale (shouldn't happen under normal operation).
            // Clear the cache and fall through to the slow path.
            cpu_buffer.clear_gpu_handle();
        }

        // ── Slow path: first call or stale handle ──────────────────
        let buffer_ref = cpu_buffer.handle();
        let guard = cpu_buffer.read();
        let data: &[u8] = bytemuck::cast_slice(guard.as_bytes());

        let (handle, result) = Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            &mut self.buffer_index,
            self.frame_index,
            &buffer_ref,
            data,
        );
        cpu_buffer.set_gpu_handle(handle);
        (handle, result)
    }

    /// Ensure a GPU buffer from a [`BufferRef`] and raw byte data.
    ///
    /// Used by generic interfaces (e.g. `MaterialTrait`) that don't hold a
    /// `CpuBuffer`.
    pub fn ensure_buffer_ref(
        &mut self,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            &mut self.buffer_index,
            self.frame_index,
            buffer_ref,
            data,
        )
    }

    /// Convenience wrapper returning only the physical resource ID.
    #[inline]
    pub fn ensure_buffer_id<T: super::GpuData>(&mut self, cpu_buffer: &super::CpuBuffer<T>) -> u64 {
        self.ensure_buffer(cpu_buffer).1.resource_id
    }

    // ────────────────────────────────────────────────────────────────────────
    // Attribute / slot-based helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Ensure the GPU buffer for a geometry attribute is created and current.
    pub fn prepare_attribute_buffer(
        &mut self,
        attr: &myth_resources::geometry::Attribute,
    ) -> EnsureResult {
        let cpu_id = attr.buffer.id();

        // ── Existing buffer ────────────────────────────────────────
        if let Some(&handle) = self.buffer_index.get(&cpu_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let mut was_recreated = false;

                if attr.version > gpu_buf.last_uploaded_version
                    && let Some(data) = &attr.data
                {
                    let bytes: &[u8] = data.as_ref();
                    was_recreated = gpu_buf.write_to_gpu(&self.device, &self.queue, bytes);
                    gpu_buf.last_uploaded_version = attr.version;
                }
                gpu_buf.last_used_frame = self.frame_index;
                return EnsureResult::new(gpu_buf.id, was_recreated);
            }
            // Stale handle
            self.buffer_index.remove(&cpu_id);
        }

        // ── New buffer ─────────────────────────────────────────────
        if let Some(data) = &attr.data {
            let bytes: &[u8] = data.as_ref();
            let mut gpu_buf = GpuBuffer::new(
                &self.device,
                bytes,
                attr.buffer.usage(),
                attr.buffer.label(),
            );
            gpu_buf.last_uploaded_version = attr.version;
            gpu_buf.last_used_frame = self.frame_index;
            let phys_id = gpu_buf.id;
            let handle = self.gpu_buffers.insert(gpu_buf);
            self.buffer_index.insert(cpu_id, handle);
            EnsureResult::created(phys_id)
        } else {
            log::error!(
                "Geometry attribute buffer {:?} missing CPU data!",
                attr.buffer.label()
            );
            // Re-check after logging (fallback for race with late uploads)
            if let Some(&h) = self.buffer_index.get(&cpu_id)
                && let Some(g) = self.gpu_buffers.get(h)
            {
                return EnsureResult::existing(g.id);
            }
            let dummy_data = [0u8; 1];
            let gpu_buf = GpuBuffer::new(
                &self.device,
                &dummy_data,
                attr.buffer.usage(),
                Some("Dummy Fallback Buffer"),
            );
            let phys_id = gpu_buf.id;
            let handle = self.gpu_buffers.insert(gpu_buf);
            self.buffer_index.insert(cpu_id, handle);
            EnsureResult::created(phys_id)
        }
    }

    /// Ensure a uniform slot buffer exists, creating it on first access and
    /// uploading new data on subsequent calls when content differs.
    pub fn prepare_uniform_slot_data(
        &mut self,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> EnsureResult {
        if let Some(&handle) = self.buffer_index.get(&slot_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let was_recreated = gpu_buf.write_to_gpu(&self.device, &self.queue, data);
                gpu_buf.last_used_frame = self.frame_index;
                return EnsureResult::new(gpu_buf.id, was_recreated);
            }
            self.buffer_index.remove(&slot_id);
        }

        let mut buf = GpuBuffer::new(
            &self.device,
            data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(label),
        );
        buf.last_used_frame = self.frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(slot_id, handle);
        EnsureResult::created(phys_id)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Lookup helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Look up a [`GpuBuffer`] by the CPU-side buffer ID.
    ///
    /// This goes through the `buffer_index` reverse map and is slightly
    /// slower than a direct `gpu_buffers.get(handle)`.
    #[inline]
    pub fn get_gpu_buffer_by_cpu_id(&self, cpu_id: u64) -> Option<&GpuBuffer> {
        self.buffer_index
            .get(&cpu_id)
            .and_then(|&h| self.gpu_buffers.get(h))
    }
}
