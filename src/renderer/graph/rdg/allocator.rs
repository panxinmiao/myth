//! RDG Transient Resource Pool
//!
//! Manages GPU texture allocation for RDG transient resources with three
//! key performance features:
//!
//! 1. **Bucketed O(1) lookup** — textures are indexed by `(Dimension, Format)`
//!    for fast acquisition instead of linear scanning.
//! 2. **Frame-based eviction** — textures unused for [`EVICTION_THRESHOLD`]
//!    consecutive frames are automatically released, preventing VRAM leaks
//!    after resize or dynamic resolution changes.
//! 3. **Lazy sub-view caching** — per-texture sub-views (mip slices, array
//!    layers, depth-only aspects) are created on demand and cached for reuse.

use crate::renderer::core::resources::Tracked;
use rustc_hash::FxHashMap;
use wgpu::{Device, TextureView};

use super::types::RdgTextureDesc;

/// Number of consecutive idle frames before a texture is evicted from the pool.
const EVICTION_THRESHOLD: u32 = 3;

// --- Sub-View Key --------------------------------------------------------------

/// Discriminator for lazily-cached sub-views of a physical texture.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SubViewKey {
    pub base_mip: u32,
    pub mip_count: Option<u32>,
    pub base_layer: u32,
    pub layer_count: Option<u32>,
    pub aspect: wgpu::TextureAspect,
}

impl Default for SubViewKey {
    fn default() -> Self {
        Self {
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
            aspect: wgpu::TextureAspect::All,
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
    fn from_desc(desc: &RdgTextureDesc) -> Self {
        Self {
            dimension: desc.dimension,
            format: desc.format,
        }
    }
}

// --- Physical Texture ----------------------------------------------------------

pub(crate) struct PhysicalTexture {
    /// Monotonically-increasing allocation identifier.
    pub(crate) uid: u64,
    pub(crate) desc: RdgTextureDesc,
    /// Raw wgpu texture handle — kept alive for sub-view creation.
    pub(crate) texture: wgpu::Texture,
    /// Full-texture default view (all mips, all layers, aspect = All).
    pub(crate) default_view: Tracked<wgpu::TextureView>,
    /// Lazily-populated sub-view cache.
    pub(crate) sub_views: FxHashMap<SubViewKey, Tracked<wgpu::TextureView>>,
    /// Number of consecutive frames this texture has been idle (not acquired).
    idle_frames: u32,
    /// Whether this texture was acquired during the current frame.
    used_this_frame: bool,
}

// --- RDG Transient Pool --------------------------------------------------------

/// GPU texture pool for RDG transient resources.
///
/// Textures are bucketed by `(Dimension, Format)` for O(1) lookup, with
/// automatic eviction of stale textures that have been idle for more than
/// [`EVICTION_THRESHOLD`] frames.
pub struct RdgTransientPool {
    /// All physical textures, indexed by dense slot index.
    pub(crate) resources: Vec<PhysicalTexture>,
    /// Per-frame timeline occupancy: `active_allocations[i]` holds the
    /// exclusive upper bound of the last timeline slot that acquired texture `i`.
    active_allocations: Vec<usize>,
    /// Monotonically-increasing allocation UID counter.
    uid_counter: u64,
    /// Bucketed free-list: maps `(Dimension, Format)` → indices into `resources`.
    buckets: FxHashMap<BucketKey, Vec<usize>>,
}

impl RdgTransientPool {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            active_allocations: Vec::new(),
            uid_counter: 0,
            buckets: FxHashMap::default(),
        }
    }

    /// Resets per-frame occupancy and evicts textures that have been idle for
    /// too many consecutive frames.
    ///
    /// Must be called once at the start of each frame, before any `acquire`.
    pub fn begin_frame(&mut self) {
        // Update idle counters and mark all textures as unused for this frame.
        for tex in &mut self.resources {
            if tex.used_this_frame {
                tex.idle_frames = 0;
            } else {
                tex.idle_frames += 1;
            }
            tex.used_this_frame = false;
        }

        // Evict stale textures (iterate in reverse to preserve indices).
        let mut evicted = 0u32;
        for i in (0..self.resources.len()).rev() {
            if self.resources[i].idle_frames >= EVICTION_THRESHOLD {
                let removed = self.resources.swap_remove(i);
                self.active_allocations.swap_remove(i);

                // Remove old index from its bucket.
                let key = BucketKey::from_desc(&removed.desc);
                if let Some(bucket) = self.buckets.get_mut(&key) {
                    bucket.retain(|&idx| idx != i);
                    // The swap_remove moved the last element into position `i` —
                    // update any bucket entry that pointed to the old last index.
                    let old_last = self.resources.len(); // post-swap_remove, this is the old last
                    if old_last != i {
                        let moved_key = BucketKey::from_desc(&self.resources[i].desc);
                        if let Some(moved_bucket) = self.buckets.get_mut(&moved_key) {
                            for idx in moved_bucket.iter_mut() {
                                if *idx == old_last {
                                    *idx = i;
                                    break;
                                }
                            }
                        }
                    }
                }
                evicted += 1;
            }
        }

        if evicted > 0 {
            log::debug!(
                "RDG pool: evicted {evicted} stale texture(s), {} remaining",
                self.resources.len()
            );
        }

        // Reset per-frame timeline occupancy.
        self.active_allocations.fill(0);
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
        desc: &RdgTextureDesc,
        first_use: usize,
        last_use: usize,
    ) -> usize {
        let bucket_key = BucketKey::from_desc(desc);

        // Search within the bucket for a compatible, idle texture.
        if let Some(bucket) = self.buckets.get(&bucket_key) {
            for &idx in bucket {
                if self.active_allocations[idx] <= first_use && self.resources[idx].desc == *desc {
                    self.active_allocations[idx] = last_use + 1;
                    self.resources[idx].used_this_frame = true;
                    return idx;
                }
            }
        }

        // No reusable texture found — allocate a new one.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RDG Transient"),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.uid_counter += 1;

        let index = self.resources.len();
        self.resources.push(PhysicalTexture {
            uid: self.uid_counter,
            desc: desc.clone(),
            texture,
            default_view: Tracked::new(view),
            sub_views: FxHashMap::default(),
            idle_frames: 0,
            used_this_frame: true,
        });
        self.active_allocations.push(last_use + 1);

        // Insert into bucket index.
        self.buckets
            .entry(bucket_key)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(index);

        index
    }

    /// Returns the default full-texture view for the given pool index.
    #[inline]
    pub fn get_view(&self, index: usize) -> &TextureView {
        &self.resources[index].default_view
    }

    /// Returns the tracked default view (carries a unique ID for state dedup).
    #[inline]
    pub fn get_tracked_view(&self, index: usize) -> &Tracked<wgpu::TextureView> {
        &self.resources[index].default_view
    }

    /// Returns the raw `wgpu::Texture` handle.
    #[inline]
    pub fn get_texture(&self, index: usize) -> &wgpu::Texture {
        &self.resources[index].texture
    }

    /// Returns the allocation UID (monotonically increasing, unique per texture).
    #[inline]
    pub fn get_uid(&self, index: usize) -> u64 {
        self.resources[index].uid
    }

    /// Lazily creates and caches a sub-view for the given physical texture.
    ///
    /// Sub-views are used for mip-level slices (Bloom), array-layer slices
    /// (shadow cascades), or depth-only aspect views (SSAO sampling).
    pub fn get_or_create_sub_view(
        &mut self,
        physical_index: usize,
        key: SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let res = &mut self.resources[physical_index];
        res.sub_views.entry(key.clone()).or_insert_with(|| {
            let view = res.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("RDG Sub-View"),
                format: None,
                dimension: None,
                usage: None,
                aspect: key.aspect,
                base_mip_level: key.base_mip,
                mip_level_count: key.mip_count,
                base_array_layer: key.base_layer,
                array_layer_count: key.layer_count,
                ..Default::default()
            });
            Tracked::new(view)
        })
    }

    pub fn get_sub_view(&self, physical_index: usize, key: &SubViewKey) -> Option<&Tracked<wgpu::TextureView>> {
        self.resources[physical_index].sub_views.get(key)
    }
}
