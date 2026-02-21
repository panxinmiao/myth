//! Transient Texture Pool
//!
//! Provides a GPU texture pool for short-lived, per-frame texture allocations.
//! Passes allocate textures during the **prepare** phase and use them during the
//! **execute** phase. At frame end, all allocations are returned to the free pool
//! for reuse in subsequent frames.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              TransientTexturePool                    │
//! │                                                     │
//! │  active: [PooledTexture]  ←── indexed by Id         │
//! │  free:   HashMap<Key, Vec<PooledTexture>>           │
//! │                                                     │
//! │  allocate() → Id    (prepare phase, &mut self)      │
//! │  get_view(Id)       (execute phase, &self)          │
//! │  get_mip_view(Id,n) (execute phase, &self)          │
//! │  reset()            (end of frame, &mut self)       │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Memory Strategy
//!
//! - Textures are **never** destroyed during normal rendering; they remain
//!   in the free pool for reuse.
//! - The pool grows on-demand: if no compatible free texture exists, a new
//!   one is created.
//! - Call [`TransientTexturePool::trim`] after resolution changes to release
//!   stale textures (those that haven't been used for several frames).

use rustc_hash::FxHashMap;

use crate::renderer::core::resources::Tracked;

// ─── Public Types ─────────────────────────────────────────────────────────────

/// Lightweight handle to a transient texture allocated from the pool.
///
/// Valid only for the current frame. After [`TransientTexturePool::reset`]
/// is called, outstanding IDs become invalid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TransientTextureId(u32);

/// Descriptor for requesting a transient texture.
#[derive(Clone, Debug)]
pub struct TransientTextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub mip_level_count: u32,
    pub label: &'static str,
}

// ─── Internal Types ───────────────────────────────────────────────────────────

/// Key for texture recycling (usage-agnostic matching is intentionally avoided
/// because mis-matched usages would cause GPU validation errors).
#[derive(Clone, PartialEq, Eq, Hash)]
struct PoolKey {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mip_level_count: u32,
}

impl PoolKey {
    fn from_desc(desc: &TransientTextureDesc) -> Self {
        Self {
            width: desc.width,
            height: desc.height,
            format: desc.format,
            usage: desc.usage,
            mip_level_count: desc.mip_level_count,
        }
    }
}

/// A pooled texture with its pre-built views.
struct PooledTexture {
    texture: wgpu::Texture,
    /// Default (full-texture, all mips) view.
    default_view: Tracked<wgpu::TextureView>,
    /// Per-mip single-level views, lazily built on first request via
    /// [`TransientTexturePool::get_mip_view`].
    mip_views: Vec<Tracked<wgpu::TextureView>>,
    /// Number of frames this texture has been sitting in the free pool
    /// without being reused. Used by [`TransientTexturePool::trim`].
    idle_frames: u32,
}

impl PooledTexture {
    fn new(device: &wgpu::Device, desc: &TransientTextureDesc) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(desc.label),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: desc.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });

        let default_view =
            Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // Pre-build per-mip views for all levels
        let mip_views = (0..desc.mip_level_count)
            .map(|mip| {
                Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(desc.label),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                }))
            })
            .collect();

        Self {
            texture,
            default_view,
            mip_views,
            idle_frames: 0,
        }
    }
}

// ─── Pool Implementation ──────────────────────────────────────────────────────

/// GPU texture pool for transient per-frame allocations.
///
/// # Three-Level Resource Architecture
///
/// | Level | Storage | Lifetime | Access |
/// |-------|---------|----------|--------|
/// | Core persistent | `FrameResources` | App lifetime | `GraphResource` enum |
/// | Transient | `TransientTexturePool` | Per-frame | `TransientTextureId` |
/// | Long-lived assets | `ResourceManager` | On-demand | `AssetHandle` |
///
/// # Thread Safety
///
/// The pool itself is **not** `Sync`. During the prepare phase it requires
/// `&mut self`; during execute it only needs `&self`. The borrow splitting
/// between `PrepareContext` and `ExecuteContext` enforces this statically.
pub struct TransientTexturePool {
    /// Textures currently allocated this frame.
    active: Vec<PooledTexture>,
    /// Free textures available for reuse, grouped by pool key.
    free: FxHashMap<PoolKey, Vec<PooledTexture>>,
}

impl TransientTexturePool {
    /// Creates an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: Vec::new(),
            free: FxHashMap::default(),
        }
    }

    // ── Prepare phase (requires &mut self) ─────────────────────────────────

    /// Allocate a transient texture matching the given descriptor.
    ///
    /// If a compatible texture is available in the free pool it will be
    /// reused; otherwise a new GPU texture is created.
    ///
    /// The returned [`TransientTextureId`] is valid until [`reset`](Self::reset)
    /// is called at the end of the frame.
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        desc: &TransientTextureDesc,
    ) -> TransientTextureId {
        let key = PoolKey::from_desc(desc);

        let mut pooled = if let Some(bucket) = self.free.get_mut(&key) {
            if let Some(mut t) = bucket.pop() {
                t.idle_frames = 0;
                t
            } else {
                PooledTexture::new(device, desc)
            }
        } else {
            PooledTexture::new(device, desc)
        };

        // Ensure mip views are present (they may have been built for a
        // previous allocation with the same dimensions but we rebuild them
        // here as a safety net — the common case is a pool hit where they
        // already exist).
        if pooled.mip_views.len() != desc.mip_level_count as usize {
            pooled.mip_views = (0..desc.mip_level_count)
                .map(|mip| {
                    Tracked::new(pooled.texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some(desc.label),
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        ..Default::default()
                    }))
                })
                .collect();
        }

        let id = TransientTextureId(self.active.len() as u32);
        self.active.push(pooled);
        id
    }

    // ── Execute phase (requires &self only) ────────────────────────────────

    /// Get the default (full-texture) view for a transient texture.
    #[must_use]
    #[inline]
    pub fn get_view(&self, id: TransientTextureId) -> &Tracked<wgpu::TextureView> {
        &self.active[id.0 as usize].default_view
    }

    /// Get a single-mip view for a transient texture.
    ///
    /// # Panics
    ///
    /// Panics if `mip_level` is out of range.
    #[must_use]
    #[inline]
    pub fn get_mip_view(
        &self,
        id: TransientTextureId,
        mip_level: u32,
    ) -> &Tracked<wgpu::TextureView> {
        &self.active[id.0 as usize].mip_views[mip_level as usize]
    }

    /// Get the raw `wgpu::Texture` for a transient texture.
    ///
    /// Useful for operations like `copy_texture_to_texture`.
    #[must_use]
    #[inline]
    pub fn get_texture(&self, id: TransientTextureId) -> &wgpu::Texture {
        &self.active[id.0 as usize].texture
    }

    /// Returns the number of mip levels for a transient texture.
    #[must_use]
    #[inline]
    pub fn mip_count(&self, id: TransientTextureId) -> u32 {
        self.active[id.0 as usize].mip_views.len() as u32
    }

    // ── Frame boundary ─────────────────────────────────────────────────────

    /// Return all active textures to the free pool.
    ///
    /// Call this at the end of every frame. After this call, all
    /// previously returned [`TransientTextureId`]s become invalid.
    pub fn reset(&mut self) {
        for t in self.active.drain(..) {
            let key = PoolKey {
                width: t.texture.width(),
                height: t.texture.height(),
                format: t.texture.format(),
                usage: t.texture.usage(),
                mip_level_count: t.texture.mip_level_count(),
            };
            self.free.entry(key).or_default().push(t);
        }
    }

    /// Release textures that have been idle for more than `max_idle_frames`.
    ///
    /// Call this periodically (e.g., after a resolution change) to avoid
    /// holding stale GPU memory.
    pub fn trim(&mut self, max_idle_frames: u32) {
        for bucket in self.free.values_mut() {
            for t in bucket.iter_mut() {
                t.idle_frames += 1;
            }
            bucket.retain(|t| t.idle_frames <= max_idle_frames);
        }
        self.free.retain(|_, bucket| !bucket.is_empty());
    }

    /// Returns the total number of textures managed by the pool
    /// (both active and free).
    #[must_use]
    pub fn total_texture_count(&self) -> usize {
        self.active.len() + self.free.values().map(Vec::len).sum::<usize>()
    }
}

impl Default for TransientTexturePool {
    fn default() -> Self {
        Self::new()
    }
}
