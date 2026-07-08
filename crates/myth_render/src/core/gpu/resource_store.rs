//! GPU-side asset arenas and reverse indices.

use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};
use myth_resources::buffer::BufferRef;

use super::{
    EnsureResult, GpuBuffer, GpuBufferHandle, GpuGeometry, GpuImage, GpuMaterial, MipmapRequest,
    TextureBinding,
};

/// GPU-side asset arenas and reverse indices.
///
/// This is the part of `ResourceManager` that mirrors CPU asset handles to
/// persistent GPU allocations. Keeping it separate lets callers borrow asset
/// storage independently from bind-group and environment state.
pub(crate) struct GpuResourceStore {
    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    gpu_images: SecondaryMap<ImageHandle, GpuImage>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID -> SlotMap handle.
    buffer_index: FxHashMap<u64, GpuBufferHandle>,

    pending_mipmap_requests: Vec<MipmapRequest>,
    pending_mipmap_keys: FxHashSet<(u64, u64)>,
}

impl GpuResourceStore {
    #[must_use]
    pub(super) fn new(
        gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: FxHashMap<u64, GpuBufferHandle>,
    ) -> Self {
        Self {
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_images: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            gpu_buffers,
            buffer_index,
            pending_mipmap_requests: Vec::new(),
            pending_mipmap_keys: FxHashSet::default(),
        }
    }

    pub(super) fn queue_mipmap_request(&mut self, request: MipmapRequest) {
        if self.pending_mipmap_keys.insert(request.key()) {
            self.pending_mipmap_requests.push(request);
        }
    }

    #[must_use]
    pub(super) fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.pending_mipmap_keys.clear();
        std::mem::take(&mut self.pending_mipmap_requests)
    }

    pub(super) fn buffer_storage_mut(
        &mut self,
    ) -> (
        &mut slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        &mut FxHashMap<u64, GpuBufferHandle>,
    ) {
        (&mut self.gpu_buffers, &mut self.buffer_index)
    }

    #[inline]
    #[must_use]
    pub(super) fn geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.gpu_geometries.get(handle)
    }

    #[inline]
    pub(super) fn geometry_mut(&mut self, handle: GeometryHandle) -> Option<&mut GpuGeometry> {
        self.gpu_geometries.get_mut(handle)
    }

    #[inline]
    pub(super) fn insert_geometry(&mut self, handle: GeometryHandle, geometry: GpuGeometry) {
        self.gpu_geometries.insert(handle, geometry);
    }

    #[inline]
    #[must_use]
    pub(super) fn material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }

    #[inline]
    pub(super) fn material_mut(&mut self, handle: MaterialHandle) -> Option<&mut GpuMaterial> {
        self.gpu_materials.get_mut(handle)
    }

    #[inline]
    #[must_use]
    pub(super) fn contains_material(&self, handle: MaterialHandle) -> bool {
        self.gpu_materials.contains_key(handle)
    }

    #[inline]
    pub(super) fn insert_material(&mut self, handle: MaterialHandle, material: GpuMaterial) {
        self.gpu_materials.insert(handle, material);
    }

    #[inline]
    #[must_use]
    pub(super) fn image(&self, handle: ImageHandle) -> Option<&GpuImage> {
        self.gpu_images.get(handle)
    }

    #[inline]
    pub(super) fn image_mut(&mut self, handle: ImageHandle) -> Option<&mut GpuImage> {
        self.gpu_images.get_mut(handle)
    }

    #[inline]
    pub(super) fn insert_image(&mut self, handle: ImageHandle, image: GpuImage) {
        self.gpu_images.insert(handle, image);
    }

    #[inline]
    pub(super) fn remove_image(&mut self, handle: ImageHandle) {
        self.gpu_images.remove(handle);
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        self.texture_bindings.get(handle)
    }

    #[inline]
    pub(super) fn insert_texture_binding(
        &mut self,
        handle: TextureHandle,
        binding: TextureBinding,
    ) {
        self.texture_bindings.insert(handle, binding);
    }

    #[inline]
    #[must_use]
    pub(super) fn texture_image(&self, handle: TextureHandle) -> Option<&GpuImage> {
        self.texture_binding(handle)
            .and_then(|binding| self.image(binding.image_handle))
    }

    pub(super) fn prune(&mut self, cutoff: u64) {
        self.gpu_geometries
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.buffer_index
            .retain(|_, h| self.gpu_buffers.contains_key(*h));
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.texture_bindings
            .retain(|_, b| self.gpu_images.contains_key(b.image_handle));
    }

    /// Upload `data` for the buffer identified by `buffer_ref`, creating or
    /// resizing the GPU-side buffer as needed.
    pub(super) fn write_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> (GpuBufferHandle, EnsureResult) {
        let cpu_id = buffer_ref.id();

        if let Some(&handle) = self.buffer_index.get(&cpu_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
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
            self.buffer_index.remove(&cpu_id);
        }

        let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
        buf.last_uploaded_version = buffer_ref.version;
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(cpu_id, handle);
        (handle, EnsureResult::created(phys_id))
    }

    pub(super) fn write_uniform_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_index: u64,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> EnsureResult {
        if let Some(&handle) = self.buffer_index.get(&slot_id) {
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                let was_recreated = gpu_buf.write_to_gpu(device, queue, data);
                gpu_buf.last_used_frame = frame_index;
                return EnsureResult::new(gpu_buf.id, was_recreated);
            }
            self.buffer_index.remove(&slot_id);
        }

        let mut buf = GpuBuffer::new(
            device,
            data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some(label),
        );
        buf.last_used_frame = frame_index;
        let phys_id = buf.id;
        let handle = self.gpu_buffers.insert(buf);
        self.buffer_index.insert(slot_id, handle);
        EnsureResult::created(phys_id)
    }

    pub(super) fn touch_buffer_by_cpu_id(&mut self, cpu_id: u64, frame_index: u64) -> bool {
        let Some(&handle) = self.buffer_index.get(&cpu_id) else {
            return false;
        };

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
            gpu_buf.last_used_frame = frame_index;
            return true;
        }

        self.buffer_index.remove(&cpu_id);
        false
    }

    #[inline]
    pub(super) fn buffer_mut(&mut self, handle: GpuBufferHandle) -> Option<&mut GpuBuffer> {
        self.gpu_buffers.get_mut(handle)
    }

    #[inline]
    pub(super) fn buffer(&self, handle: GpuBufferHandle) -> Option<&GpuBuffer> {
        self.gpu_buffers.get(handle)
    }

    #[inline]
    pub(super) fn buffer_by_cpu_id(&self, cpu_id: u64) -> Option<&GpuBuffer> {
        self.buffer_index
            .get(&cpu_id)
            .and_then(|&handle| self.gpu_buffers.get(handle))
    }
}
