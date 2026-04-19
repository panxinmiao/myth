//! RDG Transient Resource Pool
//!
//! Manages GPU texture and buffer allocation for RDG transient resources with
//! three key performance features:
//!
//! 1. **Bucketed O(1) lookup** — textures are indexed by `(Dimension, Format)`
//!    for fast acquisition instead of linear scanning.
//! 2. **Frame-based eviction** — textures unused for [`EVICTION_THRESHOLD`]
//!    consecutive frames are automatically released, preventing VRAM leaks
//!    after resize or dynamic resolution changes.
//! 3. **Lazy sub-view caching** — per-texture sub-views (mip slices, array
//!    layers, depth-only aspects) are created on demand and cached for reuse.

use crate::core::gpu::Tracked;
use rustc_hash::FxHashMap;
use wgpu::{Device, TextureView};

use super::types::{BufferDesc, TextureDesc};

/// Number of consecutive idle frames before a texture is evicted from the pool.
const EVICTION_THRESHOLD: u32 = 3;

// --- Sub-View Key --------------------------------------------------------------

/// Discriminator for lazily-cached sub-views of a physical texture.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SubViewKey {
    pub base_mip: u32,
    pub mip_count: Option<u32>,
    pub base_layer: u32,
    pub layer_count: Option<u32>,
    pub aspect: wgpu::TextureAspect,
    pub dimension: Option<wgpu::TextureViewDimension>,
}

impl Default for SubViewKey {
    fn default() -> Self {
        Self {
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
            aspect: wgpu::TextureAspect::All,
            dimension: None,
        }
    }
}

// --- Bucket Key ----------------------------------------------------------------

/// Coarse classification key for the bucketed free-list.
///
/// Textures that share the same `(Dimension, Format)` are grouped together.
/// Within a bucket, exact descriptor matching is still performed so that
/// size, mip count, sample count, and usage flags must all agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BucketKey {
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
}

impl BucketKey {
    #[inline]
    fn from_desc(desc: &TextureDesc) -> Self {
        Self {
            dimension: desc.dimension,
            format: desc.format,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferBucketKey {
    physical_size: u64,
}

impl BufferBucketKey {
    #[inline]
    fn from_desc(desc: &BufferDesc) -> Self {
        Self {
            physical_size: desc.physical_allocation_size(),
        }
    }
}

// --- Physical Texture ----------------------------------------------------------

pub(crate) struct PhysicalTexture {
    /// Monotonically-increasing allocation identifier.
    pub(crate) uid: u64,
    pub(crate) desc: TextureDesc,
    /// Raw wgpu texture handle — kept alive for sub-view creation.
    pub(crate) texture: wgpu::Texture,
    /// Full-texture default view (all mips, all layers, aspect = All).
    pub(crate) default_view: Tracked<wgpu::TextureView>,
    /// Optional cached view for the first mip level.
    pub(crate) base_mip_view: Option<Tracked<wgpu::TextureView>>,
    /// Lazily-populated sub-view cache.
    pub(crate) sub_views: FxHashMap<SubViewKey, Tracked<wgpu::TextureView>>,
    /// Frame index of the last access (acquire or sub-view retrieval). Used for eviction.
    pub(crate) last_accessed_frame: u64,
}

pub(crate) struct PhysicalBuffer {
    pub(crate) uid: u64,
    pub(crate) physical_size: u64,
    pub(crate) usage: wgpu::BufferUsages,
    pub(crate) buffer: Tracked<wgpu::Buffer>,
    pub(crate) last_accessed_frame: u64,
}

// --- RDG Transient Pool --------------------------------------------------------

/// GPU texture pool for RDG transient resources.
///
/// Textures are bucketed by `(Dimension, Format)` for O(1) lookup, with
/// automatic eviction of stale textures that have been idle for more than
/// [`EVICTION_THRESHOLD`] frames.
pub struct TransientPool {
    /// All physical textures, indexed by dense slot index.
    pub(crate) resources: Vec<Option<PhysicalTexture>>,
    /// Free slot indices within `resources` (due to eviction).
    free_slots: Vec<usize>,
    /// Per-frame timeline occupancy: `active_allocations[i]` holds the
    /// exclusive upper bound of the last timeline slot that acquired texture `i`.
    active_allocations: Vec<usize>,
    /// Monotonically-increasing allocation UID counter.
    uid_counter: u64,
    /// Bucketed free-list: maps `(Dimension, Format)` → indices into `resources`.
    buckets: FxHashMap<BucketKey, Vec<usize>>,
    /// All physical buffers, indexed by dense slot index.
    pub(crate) buffers: Vec<Option<PhysicalBuffer>>,
    /// Free slot indices within `buffers`.
    buffer_free_slots: Vec<usize>,
    /// Per-frame timeline occupancy for transient buffers.
    buffer_active_allocations: Vec<usize>,
    /// Bucketed free-list for POT-sized buffers.
    buffer_buckets: FxHashMap<BufferBucketKey, Vec<usize>>,
    /// Current frame index (for eviction tracking).
    current_frame_index: u64,
}

impl Default for TransientPool {
    fn default() -> Self {
        Self::new()
    }
}

impl TransientPool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            free_slots: Vec::new(),
            active_allocations: Vec::new(),
            uid_counter: 0,
            buckets: FxHashMap::default(),
            buffers: Vec::new(),
            buffer_free_slots: Vec::new(),
            buffer_active_allocations: Vec::new(),
            buffer_buckets: FxHashMap::default(),
            current_frame_index: 0,
        }
    }

    /// Resets per-frame occupancy and evicts textures that have been idle for
    /// too many consecutive frames.
    ///
    /// Must be called once at the start of each frame, before any `acquire`.
    pub fn begin_frame(&mut self) {
        // 1. Advance frame index for eviction tracking.
        self.current_frame_index += 1;

        #[cfg(debug_assertions)]
        let mut evicted = 0u32;
        #[cfg(debug_assertions)]
        let mut evicted_buffers = 0u32;

        // Iterate over all live textures
        for i in 0..self.resources.len() {
            // 2. Perform a read-only check first (no borrow conflicts or memory writes)
            let should_evict = if let Some(tex) = &self.resources[i] {
                // If the current frame index minus the last accessed frame index >= threshold, it is considered expired
                self.current_frame_index - tex.last_accessed_frame >= u64::from(EVICTION_THRESHOLD)
            } else {
                false
            };

            // 3. Only borrow mutably and perform eviction logic if necessary
            if should_evict {
                // 1. Take the physical texture, leaving the slot empty (None)
                let removed = self.resources[i].take().unwrap();

                // 2. Add the index to the free slots list
                self.free_slots.push(i);

                // 3. O(k) clean up the bucket (k is usually very small, and no need to fix other element indices!)
                let key = BucketKey::from_desc(&removed.desc);
                if let Some(bucket) = self.buckets.get_mut(&key) {
                    bucket.retain(|&idx| idx != i);
                }
                #[cfg(debug_assertions)]
                {
                    evicted += 1;
                }
            }
        }

        #[cfg(debug_assertions)]
        if evicted > 0 {
            log::debug!(
                "RDG pool: evicted {evicted} stale texture(s), {} remaining slots",
                self.resources.len() - self.free_slots.len()
            );
        }

        for i in 0..self.buffers.len() {
            let should_evict = if let Some(buffer) = &self.buffers[i] {
                self.current_frame_index - buffer.last_accessed_frame
                    >= u64::from(EVICTION_THRESHOLD)
            } else {
                false
            };

            if should_evict {
                let removed = self.buffers[i].take().unwrap();
                self.buffer_free_slots.push(i);

                let key = BufferBucketKey {
                    physical_size: removed.physical_size,
                };
                if let Some(bucket) = self.buffer_buckets.get_mut(&key) {
                    bucket.retain(|&idx| idx != i);
                }

                #[cfg(debug_assertions)]
                {
                    evicted_buffers += 1;
                }
            }
        }

        #[cfg(debug_assertions)]
        if evicted_buffers > 0 {
            log::debug!(
                "RDG pool: evicted {evicted_buffers} stale buffer(s), {} remaining slots",
                self.buffers.len() - self.buffer_free_slots.len()
            );
        }

        // Reset per-frame timeline occupancy.
        self.active_allocations.fill(0);
        self.buffer_active_allocations.fill(0);
    }

    /// Acquires a physical texture matching `desc`, reusing a pooled texture
    /// when possible.
    ///
    /// The `first_use` / `last_use` parameters are timeline indices from the
    /// compiled execution queue, enabling within-frame aliasing: a texture
    /// whose last use is ≤ another resource's first use can be reused.
    ///
    /// # Lookup strategy
    ///
    /// 1. Look up the `(Dimension, Format)` bucket.
    /// 2. Scan **only** that bucket for a texture whose descriptor matches
    ///    exactly and whose timeline occupancy has expired.
    /// 3. On miss, allocate a new GPU texture and insert it into the bucket.
    pub fn acquire(
        &mut self,
        device: &Device,
        desc: &TextureDesc,
        first_use: usize,
        last_use: usize,
    ) -> usize {
        let bucket_key = BucketKey::from_desc(desc);

        // Search within the bucket for a compatible, idle texture.
        if let Some(bucket) = self.buckets.get(&bucket_key) {
            for &idx in bucket {
                if self.active_allocations[idx] <= first_use
                    && let Some(tex) = &mut self.resources[idx]
                    && tex.desc == *desc
                {
                    self.active_allocations[idx] = last_use + 1;
                    tex.last_accessed_frame = self.current_frame_index;
                    return idx;
                }
            }
        }

        // No reusable texture found — allocate a new one.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Transient"),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let default_view = Tracked::new(view);

        self.uid_counter += 1;

        // Optional cached view for the first mip level (common case for render targets).
        let base_mip = if texture.mip_level_count() > 1 {
            Some(Tracked::new(texture.create_view(
                &wgpu::TextureViewDescriptor {
                    label: Some("Mip0 View"),
                    format: None,
                    dimension: None,
                    usage: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                },
            )))
        } else {
            None
        };

        let physical_tex = PhysicalTexture {
            uid: self.uid_counter,
            desc: *desc,
            texture,
            default_view,
            base_mip_view: base_mip,
            sub_views: FxHashMap::default(),
            last_accessed_frame: self.current_frame_index,
        };

        let index = if let Some(free_idx) = self.free_slots.pop() {
            self.resources[free_idx] = Some(physical_tex);
            self.active_allocations[free_idx] = last_use + 1;
            free_idx
        } else {
            let new_idx = self.resources.len();
            self.resources.push(Some(physical_tex));
            self.active_allocations.push(last_use + 1);
            new_idx
        };

        // Insert into bucket index.
        self.buckets
            .entry(bucket_key)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(index);

        index
    }

    /// Acquires a transient buffer matching the requested descriptor,
    /// reusing a POT-sized allocation when both the lifetime and usage allow it.
    pub fn acquire_buffer(
        &mut self,
        device: &Device,
        desc: &BufferDesc,
        first_use: usize,
        last_use: usize,
    ) -> usize {
        let bucket_key = BufferBucketKey::from_desc(desc);

        if let Some(bucket) = self.buffer_buckets.get(&bucket_key) {
            for &idx in bucket {
                if self.buffer_active_allocations[idx] <= first_use
                    && let Some(buffer) = &mut self.buffers[idx]
                    && buffer.usage.contains(desc.usage)
                {
                    self.buffer_active_allocations[idx] = last_use + 1;
                    buffer.last_accessed_frame = self.current_frame_index;
                    return idx;
                }
            }
        }

        let physical_size = desc.physical_allocation_size();
        let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transient Buffer (POT)"),
            size: physical_size,
            usage: desc.usage,
            mapped_at_creation: false,
        }));

        self.uid_counter += 1;

        let physical_buffer = PhysicalBuffer {
            uid: self.uid_counter,
            physical_size,
            usage: desc.usage,
            buffer,
            last_accessed_frame: self.current_frame_index,
        };

        let index = if let Some(free_idx) = self.buffer_free_slots.pop() {
            self.buffers[free_idx] = Some(physical_buffer);
            self.buffer_active_allocations[free_idx] = last_use + 1;
            free_idx
        } else {
            let new_idx = self.buffers.len();
            self.buffers.push(Some(physical_buffer));
            self.buffer_active_allocations.push(last_use + 1);
            new_idx
        };

        self.buffer_buckets
            .entry(bucket_key)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(index);

        index
    }

    #[inline]
    fn get_tex(&self, index: usize) -> &PhysicalTexture {
        debug_assert!(
            self.resources[index].is_some(),
            "Fatal: try to access evicted transient texture index {index}"
        );
        // Todo: Use unwrap_unchecked here to avoid the overhead of Option's safety checks,
        // since we have a debug_assert ensuring the texture is present.
        self.resources[index].as_ref().unwrap()
    }

    #[inline]
    fn get_tex_mut(&mut self, index: usize) -> &mut PhysicalTexture {
        debug_assert!(
            self.resources[index].is_some(),
            "Fatal: try to access evicted transient texture index {index}"
        );
        self.resources[index].as_mut().unwrap()
    }

    #[inline]
    fn get_buffer_ref(&self, index: usize) -> &PhysicalBuffer {
        debug_assert!(
            self.buffers[index].is_some(),
            "Fatal: try to access evicted transient buffer index {index}"
        );
        self.buffers[index].as_ref().unwrap()
    }

    /// Returns the default full-texture view for the given pool index.
    #[inline]
    #[must_use]
    pub fn get_view(&self, index: usize) -> &TextureView {
        &self.get_tex(index).default_view
    }

    #[inline]
    #[must_use]
    pub fn get_base_mip_view(&self, index: usize) -> &TextureView {
        let tex = &self.get_tex(index);
        tex.base_mip_view.as_ref().unwrap_or(&tex.default_view)
    }

    /// Returns the tracked default view (carries a unique ID for state dedup).
    #[inline]
    #[must_use]
    pub fn get_tracked_view(&self, index: usize) -> &Tracked<wgpu::TextureView> {
        &self.get_tex(index).default_view
    }

    /// Returns the raw `wgpu::Texture` handle.
    #[inline]
    #[must_use]
    pub fn get_texture(&self, index: usize) -> &wgpu::Texture {
        &self.get_tex(index).texture
    }

    /// Returns the allocation UID (monotonically increasing, unique per texture).
    #[inline]
    #[must_use]
    pub fn get_uid(&self, index: usize) -> u64 {
        self.get_tex(index).uid
    }

    /// Returns the raw `wgpu::Buffer` handle.
    #[inline]
    #[must_use]
    pub fn get_buffer(&self, index: usize) -> &wgpu::Buffer {
        &self.get_buffer_ref(index).buffer
    }

    /// Returns the tracked buffer handle (carries a unique ID for cache keys).
    #[inline]
    #[must_use]
    pub fn get_tracked_buffer(&self, index: usize) -> &Tracked<wgpu::Buffer> {
        &self.get_buffer_ref(index).buffer
    }

    /// Returns the allocation UID for the given transient buffer.
    #[inline]
    #[must_use]
    pub fn get_buffer_uid(&self, index: usize) -> u64 {
        self.get_buffer_ref(index).uid
    }

    /// Lazily creates and caches a sub-view for the given physical texture.
    ///
    /// Sub-views are used for mip-level slices (Bloom), array-layer slices
    /// (shadow cascades), or depth-only aspect views (SSAO sampling).
    pub fn get_or_create_sub_view(
        &mut self,
        physical_index: usize,
        key: &SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let res = self.get_tex_mut(physical_index);
        res.sub_views.entry(*key).or_insert_with(|| {
            let view = res.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Sub-View"),
                format: None,
                dimension: key.dimension,
                usage: None,
                aspect: key.aspect,
                base_mip_level: key.base_mip,
                mip_level_count: key.mip_count,
                base_array_layer: key.base_layer,
                array_layer_count: key.layer_count,
            });
            Tracked::new(view)
        })
    }

    #[must_use]
    pub fn get_sub_view(
        &self,
        physical_index: usize,
        key: &SubViewKey,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        self.get_tex(physical_index).sub_views.get(key)
    }
}
