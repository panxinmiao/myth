use crate::core::ResourceManager;
use crate::core::WgpuContext;
use crate::core::binding::{BindGroupKey, GlobalBindGroupCache};
use crate::core::gpu::MipmapGenerator;
use crate::core::gpu::{SamplerRegistry, ScreenBindGroupInfo, Tracked};
use crate::graph::frame::{BakedRenderLists, RenderLists};
use crate::graph::{ExtractedScene, RenderState};
use crate::pipeline::{PipelineCache, ShaderManager};
use myth_assets::AssetServer;
use myth_scene::RenderCamera;
use wgpu::{Device, Queue, TextureView};

use super::allocator::{SubViewKey, TransientPool};
use super::types::{RenderTargetOps, ResourceRecord, TextureNodeId};

// ─── Extract Context (Feature Pre-RDG Phase) ────────────────────────────────

/// Rich context available during the **Feature extract-and-prepare** phase.
///
/// Each `Feature::extract_and_prepare(&mut self, ctx: &mut ExtractContext)`
/// is called **before** the render graph is built.  The context provides full
/// access to GPU infrastructure, scene data, and the asset server so that
/// features can:
///
/// - Create / cache `wgpu::BindGroupLayout`s
/// - Compile pipelines via [`PipelineCache`]
/// - Upload non-transient GPU data (uniform buffers, noise textures, etc.)
///
/// After this phase, Features hold only lightweight pipeline IDs.  The
/// per-frame ephemeral `PassNode` created by `Feature::add_to_graph()`
/// carries those IDs into the graph.
pub struct ExtractContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,
    pub sampler_registry: &'a mut SamplerRegistry,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    pub resource_manager: &'a mut ResourceManager,
    pub wgpu_ctx: &'a WgpuContext,
    pub render_lists: &'a mut RenderLists,
    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,
    pub assets: &'a AssetServer,
    pub render_camera: &'a RenderCamera,
}

// ─── Prepare Context (Transient-Only) ─────────────────────────────────────────

/// Minimal context available during the RDG **prepare** phase.
///
/// After the render graph has been compiled and transient resources allocated,
/// each pass's [`PassNode::prepare`] receives this context to assemble
/// `wgpu::BindGroup`s that reference RDG-managed transient textures.
///
/// This context is deliberately kept **pure**: it provides only the GPU
/// device, transient view resolver, sampler registry, and the global bind
/// group cache.  All persistent GPU resources (pipelines, material buffers,
/// geometry buffers) must be resolved during the earlier
/// `Feature::extract_and_prepare()` phase and carried into the PassNode
/// as lightweight cloned handles.
pub struct PrepareContext<'a> {
    /// Transient resource view resolver.
    pub views: ViewResolver<'a>,
    /// GPU device for creating bind groups and sub-views.
    pub device: &'a wgpu::Device,
    /// GPU queue for immediate buffer uploads (rare in prepare).
    pub queue: &'a wgpu::Queue,
    /// Shared sampler registry (persistent, immutable during prepare).
    pub sampler_registry: &'a SamplerRegistry,
    /// Mutable cache for transient bind groups with TTL eviction.
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
}

pub struct ViewResolver<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a mut TransientPool,
}

#[must_use]
pub fn resolve_root_id(resources: &[ResourceRecord], mut id: TextureNodeId) -> TextureNodeId {
    while let Some(parent) = resources[id.0 as usize].alias_of {
        id = parent;
    }
    id
}

impl ViewResolver<'_> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`Tracked<TextureView>`].
    ///
    /// For external resources, the view is looked up in `external_resources`.
    /// For transient resources, the **default** view is obtained from the pool.
    #[must_use]
    pub fn get_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res
                .external_view_ptr
                .expect("External resource missing view pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            self.pool.get_tracked_view(physical_index)
        }
    }

    /// Returns the physical-texture allocation UID for the given node.
    ///
    /// Useful for dirty-checking: if the UID hasn't changed between frames,
    /// the physical texture is the same and derived state can be reused.
    #[must_use]
    pub fn get_physical_texture_uid(&self, id: TextureNodeId) -> u64 {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_uid(physical_index)
    }

    /// Returns the raw `wgpu::Texture` handle for the given node.
    ///
    /// Useful for passes that need to create custom views (e.g. Bloom mip chain).
    #[must_use]
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_texture(physical_index)
    }

    /// Lazily creates and caches a sub-view for a transient resource.
    ///
    /// Typical use: obtaining a `DepthOnly` aspect view from the combined
    /// depth-stencil texture for bind-group sampling.
    pub fn get_or_create_sub_view(
        &mut self,
        id: TextureNodeId,
        key: &SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_or_create_sub_view(physical_index, key)
    }

    #[must_use]
    pub fn get_sub_view(
        &self,
        id: TextureNodeId,
        key: &SubViewKey,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_sub_view(physical_index, key)
    }

    /// Returns `true` if the resource has a physical GPU allocation.
    ///
    /// A resource is considered allocated if it is external (always backed by
    /// caller-provided memory) or if the graph compiler assigned a physical
    /// pool slot.  Dead resources — those written but never read — will
    /// return `false`, allowing passes to skip bind-group creation and
    /// select leaner pipeline variants in [`PassNode::prepare`].
    #[inline]
    #[must_use]
    pub fn is_resource_allocated(&self, id: TextureNodeId) -> bool {
        let res = &self.resources[id.0 as usize];
        res.is_external || res.physical_index.is_some()
    }
}

// ─── PrepareContext Helpers ────────────────────────────────────────────────────

/// Build the screen / transient bind group (Group 3), returning a
/// pointer-stable `&'a` reference.
///
/// Encapsulates key construction and cache lookup for the screen bind group
/// used by Opaque, Transparent, and SimpleForward passes.
///
/// Callers must destructure [`PrepareContext`] to obtain split borrows of
/// `global_bind_group_cache` and `device` before calling this function.
pub fn build_screen_bind_group<'a>(
    cache: &mut GlobalBindGroupCache,
    device: &wgpu::Device,
    screen_info: &ScreenBindGroupInfo,
    transmission_view: &Tracked<wgpu::TextureView>,
    ssao_view: &Tracked<wgpu::TextureView>,
    shadow_view: &Tracked<wgpu::TextureView>,
) -> &'a wgpu::BindGroup {
    let key = BindGroupKey::new(screen_info.layout.id())
        .with_resource(transmission_view.id())
        .with_resource(screen_info.sampler.id())
        .with_resource(ssao_view.id())
        .with_resource(shadow_view.id())
        .with_resource(screen_info.shadow_compare_sampler.id());

    let layout = &*screen_info.layout;
    let sampler = &*screen_info.sampler;
    let tv = &**transmission_view;
    let sv = &**ssao_view;
    let shv = &**shadow_view;
    let shs = &*screen_info.shadow_compare_sampler;

    cache.get_or_create_bg(key, || {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen BindGroup (Group 3)"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(sv),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(shv),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(shs),
                },
            ],
        })
    })
}

// ─── Execute Context ──────────────────────────────────────────────────────────

/// Immutable context available during the RDG **execute** phase.
///
/// Provides read-only access to the compiled render graph, physical resource
/// pool, pipeline cache, render lists, and any external views injected before
/// execution.
pub struct ExecuteContext<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a TransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,

    // ─── Scene Data (Phase 3: full RDG integration) ──────────────────
    /// GPU resource manager (read-only) — used by compute, post-processing,
    /// and skybox passes that have not yet been migrated to baked commands.
    pub mipmap_generator: &'a MipmapGenerator,

    /// Pre-baked draw command lists with all GPU handles resolved.
    ///
    /// Scene-drawing passes (opaque, transparent, shadow, prepass,
    /// simple-forward) consume these instead of performing per-command
    /// handle lookups via `resource_manager`.
    pub baked_lists: &'a BakedRenderLists<'a>,

    /// Full wgpu context — depth format, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,

    /// Index of the currently executing pass within the compiled execution
    /// queue timeline.  Used by [`get_color_attachment`] and
    /// [`get_depth_stencil_attachment`] to auto-deduce `LoadOp` / `StoreOp`.
    pub current_timeline_index: usize,
}

impl ExecuteContext<'_> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// For external resources, the view is looked up in `external_views`.
    /// For transient resources, the view is obtained from the physical pool.
    #[must_use]
    pub fn get_texture_view(&self, id: TextureNodeId) -> &TextureView {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res
                .external_view_ptr
                .expect("External resource missing view pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_view(physical_index)
        }
    }

    /// Returns the [`Tracked<TextureView>`] for cache-key use during execute.
    #[must_use]
    pub fn get_tracked_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res
                .external_view_ptr
                .expect("External resource missing view pointer!");

            (unsafe { &*ptr }) as _
        } else {
            let physical_index = res
                .physical_index
                .expect("Resource has no physical memory!");
            self.pool.get_tracked_view(physical_index)
        }
    }

    /// Returns the raw [`wgpu::Texture`] handle for the given node.
    ///
    /// Useful for passes that need to create custom views at execute time
    /// (e.g. per-layer shadow map views from a 2D-array texture).
    #[must_use]
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res
                .external_view_ptr
                .expect("External resource missing view pointer!");
            let tracked_view = unsafe { &*ptr };
            tracked_view.texture()
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_texture(physical_index)
        }
    }

    /// Safely resolve a [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// Returns `None` if the resource was culled by the graph compiler
    /// (i.e. it has no consumers and no physical allocation).  Passes
    /// should use this for optional MRT targets that may have been
    /// optimized out.
    #[must_use]
    pub fn try_get_texture_view(&self, id: TextureNodeId) -> Option<&TextureView> {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res.external_view_ptr?;
            let tracked = unsafe { &*ptr };
            Some(&**tracked) // Deref Tracked 获得 wgpu::TextureView
        } else {
            res.physical_index.map(|idx| self.pool.get_view(idx))
        }
    }

    #[must_use]
    pub fn try_get_base_mip_view(&self, id: TextureNodeId) -> Option<&TextureView> {
        let root_id = resolve_root_id(self.resources, id);
        let res = &self.resources[root_id.0 as usize];

        if res.is_external {
            let ptr = res.external_view_ptr?;
            let tracked = unsafe { &*ptr };
            Some(&**tracked) // Deref Tracked 获得 wgpu::TextureView
        } else {
            res.physical_index
                .map(|idx| self.pool.get_base_mip_view(idx))
        }
    }

    /// Returns `true` if the resource is backed by physical GPU memory.
    ///
    /// Equivalent to [`RdgViewResolver::is_resource_allocated`] but
    /// available during the execute phase.
    #[inline]
    #[must_use]
    pub fn is_resource_allocated(&self, id: TextureNodeId) -> bool {
        let res = &self.resources[id.0 as usize];
        res.is_external || res.physical_index.is_some()
    }

    /// Construct a `wgpu::RenderPassColorAttachment` with explicit load
    /// semantics and automatic `StoreOp` deduction.
    ///
    /// # Load Semantics (`RenderTargetOps`)
    ///
    /// | Variant    | GPU Effect | Notes |
    /// |------------|------------|-------|
    /// | `Clear(c)` | `LoadOp::Clear(c)` | Use when a known background is required. |
    /// | `Load`     | `LoadOp::Load`     | Only valid on resources with prior content (aliases or multi-write). |
    /// | `DontCare` | `LoadOp::Clear(BLACK)` | Full-screen replace — zero bandwidth on TBDR. |
    ///
    /// # Store Semantics (Automatic)
    ///
    /// - **`Discard`** when `last_use == current_timeline_index` and the
    ///   resource is not external.
    /// - **`Store`** otherwise.
    ///
    /// # Safety Validation
    ///
    /// In debug builds, using `RenderTargetOps::Load` on a freshly created
    /// transient resource (first write, non-alias, non-external) will
    /// **panic** — this catches uninitialised-memory reads that would
    /// produce visual artefacts and waste GPU bandwidth.
    ///
    /// # MSAA Resolve
    ///
    /// The optional `resolve_target` specifies a single-sample texture for
    /// hardware MSAA resolve.  If the target was culled (no allocation),
    /// it is silently ignored.
    ///
    /// Returns `None` if the primary resource was culled.
    #[must_use]
    pub fn get_color_attachment(
        &self,
        id: TextureNodeId,
        ops: RenderTargetOps,
        resolve_target: Option<TextureNodeId>,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let view = self.try_get_base_mip_view(id)?;

        let res = &self.resources[id.0 as usize];
        let ti = self.current_timeline_index;

        let is_first_write = res.first_use == ti && !res.is_external && res.alias_of.is_none();

        // Validate: Load on an uninitialised transient resource is always a bug.
        assert!(
            !(matches!(ops, RenderTargetOps::Load) && is_first_write),
            "RDG Validation Error: LoadOp::Load on freshly created transient \
             resource '{name}' (node {id:?}).  This reads uninitialised GPU \
             memory and wastes bandwidth.  Use RenderTargetOps::DontCare for \
             full-screen replace shaders, or RenderTargetOps::Clear(color) \
             when a specific background is needed.",
            name = res.name,
        );

        // For alias resources the caller should pass `Load` explicitly;
        // the graph guarantees content inheritance from the prior version.
        let load = if is_first_write {
            ops.to_wgpu_load_op()
        } else {
            // Aliases and subsequent writes always load prior content
            // unless the caller explicitly requests Clear/DontCare.
            match ops {
                RenderTargetOps::Load => wgpu::LoadOp::Load,
                other => other.to_wgpu_load_op(),
            }
        };

        let store = if res.last_use == ti && !res.is_external {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        let resolve_view = resolve_target.and_then(|rt| self.try_get_base_mip_view(rt));

        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: resolve_view,
            ops: wgpu::Operations { load, store },
            depth_slice: None,
        })
    }

    /// Auto-deduce `LoadOp` and `StoreOp` for a depth-stencil attachment.
    ///
    /// Rules mirror [`get_color_attachment`]:
    /// - First use of a non-alias, non-external resource →
    ///   `Clear(clear_depth)`, otherwise `Load`.
    /// - Last use on a non-external resource → `Discard`, otherwise `Store`.
    ///
    /// Returns `None` if the resource was culled.
    #[must_use]
    pub fn get_depth_stencil_attachment(
        &self,
        id: TextureNodeId,
        clear_depth: f32,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        let view = self.try_get_texture_view(id)?;
        let res = &self.resources[id.0 as usize];
        let ti = self.current_timeline_index;

        let load = if res.first_use == ti && res.alias_of.is_none() {
            wgpu::LoadOp::Clear(clear_depth)
        } else {
            wgpu::LoadOp::Load
        };

        let store = if res.last_use == ti && !res.is_external {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations { load, store }),
            stencil_ops: None,
        })
    }
}
