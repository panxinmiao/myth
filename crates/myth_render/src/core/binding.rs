//! Bind-group cache and the [`Bindings`] trait.
//!
//! The [`myth_resources::BindingResource`] and [`ResourceBuilder`] data types live in
//! `myth_resources`; this module provides the **trait** that connects
//! domain objects (Material, Geometry, Mesh, etc.) to the builder, plus
//! a frame-based bind-group cache with TTL eviction.

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::graph::RenderState;
use myth_resources::Mesh;
use myth_resources::ResourceBuilder;
use myth_resources::geometry::Geometry;
use myth_resources::material::{Material, RenderableMaterialTrait};
use myth_resources::uniforms::{MorphUniforms, RenderStateUniforms};
use myth_scene::Scene;

/// Binding resource Trait
pub trait Bindings {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
}

impl Bindings for Material {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }
}

impl Bindings for Geometry {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // Morph Target Storage Buffers
        if self.has_morph_targets() {
            // Position morph storage
            if let (Some(buffer), Some(data)) =
                (&self.morph_position_buffer, self.morph_position_bytes())
            {
                builder.add_storage_buffer(
                    "morph_positions",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }

            // Normal morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_normal_buffer, self.morph_normal_bytes())
            {
                builder.add_storage_buffer(
                    "morph_normals",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }

            // Tangent morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_tangent_buffer, self.morph_tangent_bytes())
            {
                builder.add_storage_buffer(
                    "morph_tangents",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(myth_resources::WgslStructName::Name("f32".into())),
                );
            }
        }
    }
}

impl Bindings for Mesh {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // todo: should we check if geometry features contain USE_MORPHING?
        builder.add_uniform::<MorphUniforms>(
            "morph_targets",
            &self.morph_uniforms,
            wgpu::ShaderStages::VERTEX,
        );
    }
}

impl Bindings for RenderState {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<RenderStateUniforms>(
            "render_state",
            self.uniforms(),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        );
    }
}

impl Bindings for Scene {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // Scene-level global bindings are now built by
        // `ResourceManager::define_global_scene_bindings` which resolves
        // environment textures from the GPU cache instead of from Environment.
        // This impl is kept empty for trait coherence; the actual bindings
        // are constructed in `ResourceManager::create_global_state`.
    }
}

/// Number of idle frames before a cached bind group is evicted.
const BIND_GROUP_EVICTION_THRESHOLD: u64 = 10;

/// A cached bind group entry with TTL (Time-To-Live) tracking.
///
/// Records the last frame the bind group was accessed so that stale
/// entries can be garbage-collected when the associated physical
/// textures are released by the transient pool.
struct CachedBindGroup {
    bg: wgpu::BindGroup,
    last_accessed_frame: AtomicU64,
}

/// Global bind group cache with frame-based TTL eviction.
///
/// Caches `wgpu::BindGroup` instances keyed by their resource composition.
/// Each entry tracks the last frame it was accessed; entries idle for more
/// than [`BIND_GROUP_EVICTION_THRESHOLD`] consecutive frames are evicted
/// by [`garbage_collect`](Self::garbage_collect), ensuring that VRAM held
/// by stale bind groups (e.g. after SSAO is disabled or a resize occurs)
/// is promptly released.
///
/// # Heap-Stable Pointers
///
/// Entries are stored behind [`Box`] to guarantee that `&wgpu::BindGroup`
/// references remain valid even when the backing `HashMap` resizes.  This
/// enables the prepare phase to hand out `&'a wgpu::BindGroup` references
/// that survive across multiple `prepare()` calls without invalidation.
pub struct GlobalBindGroupCache {
    cache: FxHashMap<BindGroupKey, Box<CachedBindGroup>>,
    current_frame: u64,
}

impl Default for GlobalBindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalBindGroupCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            current_frame: 0,
        }
    }

    /// Advances the internal frame counter.
    ///
    /// Must be called once per frame **before** any `get` / `insert` /
    /// `get_or_create` calls so that TTL timestamps are correct.
    #[inline]
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Returns the current frame number.
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Looks up a cached bind group, updating its TTL timestamp on hit.
    #[must_use]
    pub fn get(&self, key: &BindGroupKey) -> Option<&wgpu::BindGroup> {
        if let Some(entry) = self.cache.get(key) {
            entry
                .last_accessed_frame
                .store(self.current_frame, Ordering::Relaxed);
            Some(&entry.bg)
        } else {
            None
        }
    }

    /// Checks whether the cache contains an entry for the given key
    /// without updating its TTL.
    #[must_use]
    pub fn contains(&self, key: &BindGroupKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Inserts a bind group into the cache with the current frame timestamp.
    pub fn insert(&mut self, key: BindGroupKey, bind_group: wgpu::BindGroup) {
        self.cache.insert(
            key,
            Box::new(CachedBindGroup {
                bg: bind_group,
                last_accessed_frame: AtomicU64::new(self.current_frame),
            }),
        );
    }

    /// Returns an existing entry or creates one via `factory`, updating TTL.
    pub fn get_or_create(
        &mut self,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        let frame = self.current_frame;
        let entry = self.cache.entry(key).or_insert_with(|| {
            Box::new(CachedBindGroup {
                bg: factory(),
                last_accessed_frame: AtomicU64::new(frame),
            })
        });
        entry.last_accessed_frame.store(frame, Ordering::Relaxed);
        &entry.bg
    }

    /// Ensure a bind group exists and return a pointer-stable `&'a` reference.
    ///
    /// Because entries are `Box`-allocated, the returned `&wgpu::BindGroup`
    /// resides at a heap address that remains valid even if the underlying
    /// `HashMap` is later resized by subsequent insertions.  This allows
    /// the prepare phase to hand out frame-scoped references that survive
    /// across multiple `prepare()` calls.
    ///
    /// # Safety
    ///
    /// The caller must ensure the cache is **not** garbage-collected or
    /// cleared while the returned reference is alive.  In practice this
    /// is guaranteed by the sequential prepare→execute pipeline: GC runs
    /// only at frame boundaries, after all references have expired.
    pub unsafe fn get_or_create_stable<'a>(
        &mut self,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> &'a wgpu::BindGroup {
        let frame = self.current_frame;
        let entry = self.cache.entry(key).or_insert_with(|| {
            Box::new(CachedBindGroup {
                bg: factory(),
                last_accessed_frame: AtomicU64::new(frame),
            })
        });
        entry.last_accessed_frame.store(frame, Ordering::Relaxed);
        // SAFETY: Box<CachedBindGroup> places the BindGroup on the heap
        // at a stable address.  HashMap resizing moves Box pointers but
        // not the heap-allocated data they reference.  The entry is
        // retained until garbage_collect() at frame boundaries.
        let ptr = &raw const entry.bg;
        unsafe { &*ptr }
    }

    /// Retrieve or create a bind group, returning a pointer-stable `&'a`
    /// reference suitable for storing on a `PassNode<'a>`.
    ///
    /// This is the **safe entry-point** for RDG prepare-phase code.
    /// It wraps [`Self::get_or_create_stable`] and relies on the
    /// architectural invariant that cache entries live for the full frame.
    pub(crate) fn get_or_create_bg<'a>(
        &mut self,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> &'a wgpu::BindGroup {
        // SAFETY: Box'd entries have heap-stable addresses.  The cache is
        // not garbage-collected until frame boundaries, after all `&'a`
        // references have expired in the sequential prepare→execute model.
        unsafe { self.get_or_create_stable(key, factory) }
    }

    /// Forcibly clears all entries.
    ///
    /// Call on window resize to drop every bind group that references
    /// now-stale texture views.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Evicts entries that have not been accessed for
    /// [`BIND_GROUP_EVICTION_THRESHOLD`] frames.
    ///
    /// Should be called once at the **end** of each frame, after all
    /// passes have executed.  This keeps the cache in sync with the
    /// transient pool's own idle-eviction policy, preventing VRAM leaks
    /// from orphaned bind groups.
    pub fn garbage_collect(&mut self) {
        let threshold = self
            .current_frame
            .saturating_sub(BIND_GROUP_EVICTION_THRESHOLD);
        let before = self.cache.len();
        self.cache
            .retain(|_, entry| entry.last_accessed_frame.load(Ordering::Relaxed) >= threshold);
        let evicted = before - self.cache.len();
        if evicted > 0 {
            log::debug!(
                "GlobalBindGroupCache: evicted {evicted} stale entries, {} remaining",
                self.cache.len()
            );
        }
    }
}

/// Global `BindGroup` cache key
/// Contains the unique identifier of the Layout + unique identifiers of all binding resources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupKey {
    layout_id: u64,
    resources: SmallVec<[u64; 8]>,
}

impl BindGroupKey {
    #[must_use]
    pub fn new(layout_id: u64) -> Self {
        Self {
            layout_id,
            resources: SmallVec::with_capacity(8), // Estimated common size
        }
    }

    #[must_use]
    pub fn with_resource(mut self, id: u64) -> Self {
        self.resources.push(id);
        self
    }
}
