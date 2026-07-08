//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Declarative Render Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Resource Ownership (Explicit Wiring)
//!
//! The Composer registers only the **external** `Surface_Out` resource and
//! the routing-level `LDR_Intermediate`.  All scene-level transient resources
//! (`Scene_Color_HDR`, `Scene_Depth`, MSAA intermediates, specular MRT, etc.)
//! are created by their **producer passes** inside `add_to_graph()`.  Each
//! pass returns typed output structs (`PrepassOutputs`, `OpaqueOutputs`, …)
//! carrying `TextureNodeId` values that the Composer threads to downstream
//! consumers — no blackboard lookups remain for mutable resources.
//!
//! # Rendering Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Unified RDG Pipeline                        │
//! │                                                                │
//! │  HighFidelity:                                                 │
//! │  BRDF LUT → IBL → Shadow → Prepass → SSAO → Opaque →         │
//! │  SSSS → Skybox → TransmissionCopy → Transparent →             │
//! │  [Bloom_System: Extract → DS_1..N → US_N..0 → Composite] →   │
//! │  ToneMap → FXAA → [User Hooks] → Surface                     │
//! │                                                                │
//! │  BasicForward:                                                 │
//! │  BRDF LUT → IBL → Shadow → Skybox(prepare) →                  │
//! │  SimpleForward → [User Hooks] → Surface                       │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
//!         let new_surface = rdg.add_pass("UI_Pass", |builder| {
//!             let out = builder.mutate_texture(bb.surface_out, "Surface_With_UI");
//!             (ui_node, out)
//!         });
//!         GraphBlackboard { surface_out: new_surface, ..bb }
//!     })
//!     .render();
//! ```

use crate::core::binding::GlobalBindGroupCache;
use crate::core::gpu::{CubeSourceType, Tracked};
use crate::core::{ResourceManager, WgpuContext};
use crate::graph::ExtractedScene;
use crate::graph::RenderState;
use crate::graph::core::ClusteredScreenBindings;
use crate::graph::core::GraphStorage;
use crate::graph::core::graph::FrameConfig;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, FrameArena, GraphBlackboard, HookStage, PassNode,
    PrepareContext, RenderGraph, TextureDesc, TransientPool, ViewResolver,
};
use crate::graph::extracted::SceneFeatures;
use crate::graph::frame::{PreparedSkyboxDraw, RenderLists};
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::utils::add_msaa_resolve_pass;
use crate::graph::passes::{
    AtmosphereFeature, BloomFeature, BrdfLutFeature, CasFeature, ClusteredLightingFeature,
    ClusteredLightingInputs, EquirectToCubeFeature, FxaaFeature, HiZFeature, IblComputeFeature,
    MsaaSyncFeature, OpaqueFeature, PrepassFeature, PresentFeature, ShadowFeature,
    SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsgiFeature, SsrFeature, SsssFeature,
    TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use crate::pipeline::PipelineCache;
use crate::pipeline::ShaderManager;
use crate::renderer::FrameTime;
use crate::settings::RendererSettings;
use myth_assets::AssetServer;
use myth_resources::uniforms::LightBufferMetadata;
use myth_scene::Scene;
use myth_scene::camera::RenderCamera;

pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,

    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,
    pub renderer_settings: &'a RendererSettings,
    pub clustered_lighting_enabled: bool,

    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// Render lists (populated by `SceneCullPass`)
    pub render_lists: &'a mut RenderLists,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: RenderCamera,
    pub assets: &'a AssetServer,
    pub frame_time: FrameTime,

    pub graph_storage: &'a mut GraphStorage,
    pub transient_pool: &'a mut TransientPool,
    // pub sampler_registry: &'a mut SamplerRegistry,
    pub frame_arena: &'a FrameArena,

    // ─── RDG Features ────────────────────────────────────────────────────
    // Post-processing
    pub fxaa_pass: &'a mut FxaaFeature,
    pub taa_pass: &'a mut TaaFeature,
    pub cas_pass: &'a mut CasFeature,
    pub tone_map_pass: &'a mut ToneMappingFeature,
    pub present_pass: &'a mut PresentFeature,
    pub bloom_pass: &'a mut BloomFeature,
    pub ssao_pass: &'a mut SsaoFeature,
    pub hiz_pass: &'a mut HiZFeature,
    pub ssgi_pass: &'a mut SsgiFeature,
    pub ssr_pass: &'a mut SsrFeature,
    // Scene rendering
    pub prepass: &'a mut PrepassFeature,
    pub opaque_pass: &'a mut OpaqueFeature,
    pub skybox_pass: &'a mut SkyboxFeature,
    pub transparent_pass: &'a mut TransparentFeature,
    pub transmission_copy_pass: &'a mut TransmissionCopyFeature,
    pub simple_forward_pass: &'a mut SimpleForwardFeature,
    pub ssss_pass: &'a mut SsssFeature,
    pub msaa_sync_pass: &'a mut MsaaSyncFeature,

    // Shadow + Compute
    pub shadow_pass: &'a mut ShadowFeature,
    pub brdf_pass: &'a mut BrdfLutFeature,
    pub equirect_to_cube_pass: &'a mut EquirectToCubeFeature,
    pub ibl_pass: &'a mut IblComputeFeature,
    pub atmosphere_pass: &'a mut AtmosphereFeature,
    pub clustered_lighting_pass: &'a mut ClusteredLightingFeature,

    #[cfg(feature = "3dgs")]
    // Gaussian Splatting
    pub gaussian_splatting_pass: &'a mut GaussianSplattingFeature,

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub debug_view_pass: &'a mut crate::graph::passes::DebugViewFeature,
}

pub struct GraphBuilderContext<'a, 'g> {
    pub graph: &'g mut RenderGraph<'a>,
    pub pipeline_cache: &'a PipelineCache,
    pub frame_config: &'g FrameConfig,
    scene_local_light_count: u32,
}

impl GraphBuilderContext<'_, '_> {
    #[inline]
    pub(crate) fn new<'a, 'g>(
        graph: &'g mut RenderGraph<'a>,
        pipeline_cache: &'a PipelineCache,
        frame_config: &'g FrameConfig,
        scene_local_light_count: u32,
    ) -> GraphBuilderContext<'a, 'g> {
        GraphBuilderContext {
            graph,
            pipeline_cache,
            frame_config,
            scene_local_light_count,
        }
    }

    #[cfg(feature = "rdg_inspector")]
    pub fn with_group<F, R>(&mut self, group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.graph.push_group(group_name);
        let result = f(self);
        self.graph.pop_group();
        result
    }

    /// Zero-cost fallback when the inspector is disabled.
    #[cfg(not(feature = "rdg_inspector"))]
    #[inline]
    pub fn with_group<F, R>(&mut self, _group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        f(self)
    }

    /// Returns the logical descriptor of an RDG buffer visible to the current
    /// graph build. User hooks can use this to size derived transient buffers
    /// without reaching into renderer internals.
    #[inline]
    #[must_use]
    pub fn buffer_desc(&self, id: BufferNodeId) -> BufferDesc {
        self.graph.storage.resources[id.index() as usize].buffer_desc()
    }

    /// Returns the extracted CPU local-light count for the current frame.
    /// This is the authoritative emptiness check for local-light hooks; the
    /// imported storage buffer may still be a one-slot dummy allocation.
    #[inline]
    #[must_use]
    pub const fn scene_local_light_count(&self) -> u32 {
        self.scene_local_light_count
    }
}

struct SceneEnvironmentGraphResources<'a> {
    base_cube: crate::graph::core::TextureNodeId,
    pmrem: crate::graph::core::TextureNodeId,
    source_type: CubeSourceType,
    compute_state: Option<&'a crate::core::gpu::EnvironmentComputeState>,
}

#[derive(Clone, Copy)]
pub struct GpuLightBuffers {
    pub light_metadata: BufferNodeId,
    pub light_storage: BufferNodeId,
    pub indirect_count_buffer: Option<BufferNodeId>,
}

struct SceneLightingImportPassNode;

impl PassNode<'_> for SceneLightingImportPassNode {
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
}

fn import_scene_lighting(
    ctx: &mut GraphBuilderContext<'_, '_>,
    render_lists: &RenderLists,
) -> GpuLightBuffers {
    let light_metadata_buffer = render_lists
        .gpu_local_light_metadata_buffer
        .as_ref()
        .expect("scene light metadata buffer missing");
    let light_storage_buffer = render_lists
        .gpu_local_light_storage_buffer
        .as_ref()
        .expect("scene light storage buffer missing");

    ctx.graph.add_pass("Scene_Lighting_Import", |builder| {
        let light_metadata = builder.read_external_buffer(
            "Scene_Local_Light_Metadata",
            BufferDesc::new(
                std::mem::size_of::<LightBufferMetadata>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            light_metadata_buffer,
        );
        let light_storage = builder.read_external_buffer(
            "Scene_Local_Lights",
            BufferDesc::new(
                light_storage_buffer.size(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            light_storage_buffer,
        );

        (
            SceneLightingImportPassNode,
            GpuLightBuffers {
                light_metadata,
                light_storage,
                indirect_count_buffer: None,
            },
        )
    })
}

fn base_cube_desc(texture: &wgpu::Texture) -> TextureDesc {
    TextureDesc::new(
        texture.width(),
        texture.height(),
        6,
        texture.mip_level_count(),
        1,
        wgpu::TextureDimension::D2,
        crate::HDR_TEXTURE_FORMAT,
        wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
    )
}

fn pmrem_desc(texture: &wgpu::Texture) -> TextureDesc {
    TextureDesc::new(
        texture.width(),
        texture.height(),
        6,
        texture.mip_level_count(),
        1,
        wgpu::TextureDimension::D2,
        crate::HDR_TEXTURE_FORMAT,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

/// Frame Composer
///
/// Holds all context references needed to render a single frame and provides
/// a fluent API for injecting custom RDG passes via hooks.
///
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    frame_config: FrameConfig,
    #[allow(clippy::type_complexity)]
    gpu_local_light_hook: Option<
        Box<dyn for<'g> FnMut(&mut GraphBuilderContext<'a, 'g>) -> Option<GpuLightBuffers> + 'a>,
    >,
    #[allow(clippy::type_complexity)]
    hooks: smallvec::SmallVec<
        [(
            HookStage,
            Option<Box<dyn FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a>>,
        ); 4],
    >,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    pub(crate) fn new(ctx: ComposerContext<'a>, size: (u32, u32)) -> Self {
        let frame_config = FrameConfig {
            width: size.0,
            height: size.1,
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
            surface_format: ctx.wgpu_ctx.surface_view_format,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        };

        Self {
            ctx,
            frame_config,
            gpu_local_light_hook: None,
            hooks: smallvec::SmallVec::new(),
        }
    }

    /// Returns a reference to the wgpu device.
    #[inline]
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.wgpu_ctx.device
    }

    /// Returns a reference to the resource manager.
    ///
    /// Useful for user-land passes that need to resolve engine resources
    /// (e.g. texture handles) before the RDG prepare phase.
    #[inline]
    #[must_use]
    pub fn resource_manager(&self) -> &ResourceManager {
        self.ctx.resource_manager
    }

    /// Registers a custom pass hook that will be invoked during graph building.
    ///
    /// The closure receives a mutable reference to the [`RenderGraph`] and
    /// the [`GraphBlackboard`] containing the frame's well-known resource slots.
    /// It must return an (optionally updated) [`GraphBlackboard`] — the Rust
    /// type system enforces that every hook path returns a valid blackboard.
    ///
    /// Hooks registered with [`HookStage::AfterPostProcess`] run after all
    /// built-in post-processing (Bloom, ToneMap, FXAA) and are typically used
    /// for UI rendering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
    ///         let new_surface = rdg.add_pass("MyPass", |builder| {
    ///             let out = builder.mutate_texture(bb.surface_out, "Surface_MyPass");
    ///             (MyPassNode { target: out }, out)
    ///         });
    ///         GraphBlackboard { surface_out: new_surface, ..bb }
    ///     })
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_custom_pass<F>(mut self, stage: HookStage, hook: F) -> Self
    where
        F: FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a,
    {
        self.hooks.push((stage, Some(Box::new(hook))));
        self
    }

    /// Registers a hook that can inject an optional GPU-generated local-light
    /// track before clustered lighting and forward shading are wired.
    #[inline]
    #[must_use]
    pub fn inject_gpu_local_lights<F>(mut self, hook: F) -> Self
    where
        F: for<'g> FnMut(&mut GraphBuilderContext<'a, 'g>) -> Option<GpuLightBuffers> + 'a,
    {
        self.gpu_local_light_hook = Some(Box::new(hook));
        self
    }

    /// Executes the full rendering pipeline.
    ///
    /// In **windowed mode** the result is presented to the swap-chain surface.
    /// In **headless mode** the result is written to the offscreen render target
    /// (available for readback via [`Renderer::readback_pixels`]).
    ///
    /// # Architecture
    ///
    /// 1. **Acquire Render Target** — swap-chain back buffer or headless texture
    /// 2. **Build RDG** — register resources, wire passes (compute, shadow,
    ///    scene, post-processing), invoke user hooks
    /// 3. **Compile & Execute** — topological sort, allocate transients,
    ///    prepare, execute, submit
    /// 4. **Present** — present swap-chain (windowed only)
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    pub fn render(mut self) {
        // ━━━ 1. Acquire Render Target ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let view_format = self.ctx.wgpu_ctx.surface_view_format;

        // Acquire either the swap-chain back buffer or the headless texture view.
        // `surface_output` is `Some` only in windowed mode and holds the
        // `SurfaceTexture` that must be queued for presentation after submission.
        let (surface_view, width, height, surface_output);

        if let Some(surface) = &self.ctx.wgpu_ctx.surface {
            let output = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame) => frame,
                wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                    if let Some(config) = &self.ctx.wgpu_ctx.config {
                        surface.configure(&self.ctx.wgpu_ctx.device, config);
                    }
                    frame
                }
                _ => {
                    log::error!("Failed to acquire swap-chain surface");
                    return;
                }
            };

            surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                ..Default::default()
            });
            width = output.texture.width();
            height = output.texture.height();
            surface_output = Some(output);
        } else if let Some(tex) = &self.ctx.wgpu_ctx.headless_texture {
            surface_view = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                ..Default::default()
            });
            width = tex.width();
            height = tex.height();
            surface_output = None;
        } else {
            log::error!("No render target available (neither surface nor headless texture)");
            return;
        }

        // ━━━ 2. Build Unified RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let mut graph = RenderGraph::new(self.ctx.graph_storage, self.ctx.frame_arena);

        let surface_desc = TextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_view_tracked = Tracked::with_id(surface_view, 0);

        let surface_out =
            graph.import_external_resource("Surface_View", surface_desc, &surface_view_tracked);

        let mut graph_ctx = GraphBuilderContext::new(
            &mut graph,
            self.ctx.pipeline_cache,
            &self.frame_config,
            self.ctx.extracted_scene.local_light_count() as u32,
        );

        // ── 2a. Register Resources ──────────────────────────────────────
        // Only the swapchain surface is truly external.
        // Scene colour and depth are transient — owned and aliased by the RDG.

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let msaa_samples = self.ctx.wgpu_ctx.msaa_samples;
        let is_msaa = msaa_samples > 1;

        // ── 2b. Scene Configuration ────────────────────────────────────
        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let ssgi_enabled = is_high_fidelity
            && self
                .ctx
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSGI);
        let ssr_enabled = is_high_fidelity
            && self
                .ctx
                .extracted_scene
                .scene_variants
                .contains(SceneFeatures::USE_SSR);
        let needs_scene_hiz = ssgi_enabled || ssr_enabled;

        let needs_feature_id = is_high_fidelity && (self.ctx.scene.ssss.enabled || ssr_enabled);

        #[cfg(feature = "debug_view")]
        let (dbg_needs_normal, dbg_needs_velocity) = {
            use crate::graph::render_state::DebugViewTarget;
            let target = DebugViewTarget::from_mode(self.ctx.render_state.debug_view_mode);
            (
                target == DebugViewTarget::SceneNormal,
                target == DebugViewTarget::Velocity,
            )
        };

        #[cfg(not(feature = "debug_view"))]
        let (dbg_needs_normal, dbg_needs_velocity) = (false, false);

        let taa_enabled = self.ctx.camera.aa_mode.is_taa();
        let needs_normal = ssao_enabled || ssgi_enabled || needs_feature_id || dbg_needs_normal;
        let needs_velocity = taa_enabled || ssgi_enabled || ssr_enabled || dbg_needs_velocity;

        // let needs_normal = ssao_enabled || needs_feature_id;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let ssss_enabled = self.ctx.scene.ssss.enabled;
        let has_transmission = self.ctx.render_lists.use_transmission;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        let has_active_environment = matches!(
            self.ctx.scene.background.mode,
            myth_scene::background::BackgroundMode::Procedural(_)
        ) || self.ctx.scene.environment.has_env_map();
        // let fxaa_enabled = self.ctx.wgpu_ctx.fxaa_enabled && is_high_fidelity;
        // let taa_enabled = self.ctx.wgpu_ctx.taa_enabled && is_high_fidelity;

        {
            let scene_id = self.ctx.scene.id();
            let scene_gpu_environment = if has_active_environment {
                self.ctx.resource_manager.gpu_environment(scene_id)
            } else {
                None
            };

            let scene_environment_resources =
                scene_gpu_environment.map(|gpu_env| SceneEnvironmentGraphResources {
                    base_cube: graph_ctx.graph.import_external_resource(
                        "Scene_Environment_BaseCube",
                        base_cube_desc(&gpu_env.base_cube_texture),
                        &gpu_env.base_cube_view,
                    ),
                    pmrem: graph_ctx.graph.import_external_resource(
                        "Scene_Environment_PMREM",
                        pmrem_desc(&gpu_env.pmrem_texture),
                        &gpu_env.pmrem_view,
                    ),
                    source_type: gpu_env.source_type,
                    compute_state: gpu_env.source_ready.then_some(&gpu_env.compute_state),
                });

            // let mut env_dependency_base = None;
            let mut env_dependency_pmrem = None;
            let mut procedural_skybox_dependencies = [None, None];
            let mut atmosphere_transmittance = None;
            let mut atmosphere_bake_params = None;

            // ── 2c. Wire Compute + Shadow Passes ───────────────────────────
            graph_ctx.with_group("Compute", |c| {
                if self.ctx.resource_manager.needs_brdf_compute {
                    self.ctx.brdf_pass.add_to_graph(c);
                }

                if let (Some(env), Some(gpu_env)) =
                    (scene_environment_resources.as_ref(), scene_gpu_environment)
                {
                    match env.source_type {
                        CubeSourceType::Procedural => {
                            if let myth_scene::background::BackgroundMode::Procedural(params) =
                                &self.ctx.scene.background.mode
                            {
                                let atmosphere_output = self.ctx.atmosphere_pass.add_to_graph(
                                    c,
                                    scene_id,
                                    params,
                                    env.base_cube,
                                    &gpu_env.base_cube_storage_view,
                                    env.compute_state,
                                );
                                procedural_skybox_dependencies =
                                    atmosphere_output.skybox_dependencies();
                                atmosphere_transmittance = Some(atmosphere_output.transmittance);
                                atmosphere_bake_params = Some(atmosphere_output.bake_params);
                                // env_dependency_base = atmosphere_output.baked_base_cube;
                                env_dependency_pmrem = self
                                    .ctx
                                    .ibl_pass
                                    .add_to_graph(
                                        c,
                                        scene_id,
                                        env.base_cube,
                                        env.pmrem,
                                        env.source_type,
                                        env.compute_state,
                                    )
                                    .updated_pmrem;
                            }
                        }
                        CubeSourceType::Equirectangular | CubeSourceType::Cubemap => {
                            self.ctx.equirect_to_cube_pass.add_to_graph(
                                c,
                                scene_id,
                                env.source_type,
                                env.base_cube,
                                env.compute_state,
                            );
                            env_dependency_pmrem = self
                                .ctx
                                .ibl_pass
                                .add_to_graph(
                                    c,
                                    scene_id,
                                    env.base_cube,
                                    env.pmrem,
                                    env.source_type,
                                    env.compute_state,
                                )
                                .updated_pmrem;
                        }
                    }
                }
            });

            let shadow_output = if self.ctx.extracted_scene.has_shadow_casters() {
                graph_ctx.with_group("Shadow", |c| self.ctx.shadow_pass.add_to_graph(c))
            } else {
                crate::graph::passes::shadow::ShadowOutput {
                    shadow_2d: None,
                    shadow_cube: None,
                }
            };

            // ── 2d. Wire Scene Rendering Passes (explicit data-flow) ──────
            //
            // Each pass's `add_to_graph` creates its own transient resources
            // internally and returns typed output structs.  The Composer
            // threads `TextureNodeId` values from producer to consumer —
            // no blackboard lookups remain for mutable resources.

            // Track scene_color / scene_depth for the GraphBlackboard (hooks).
            let mut bb_scene_color = None;
            let mut bb_scene_depth = None;
            let mut bb_scene_hiz = None;

            // Debug view: capture intermediate texture IDs for safe resolution.
            #[cfg(feature = "debug_view")]
            let mut dbg_normals: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_velocity: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssao: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssgi_raw: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssgi_denoised: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_raw: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_resolved: Option<crate::graph::core::TextureNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_ssr_trace_diagnostic: Option<
                crate::graph::core::TextureNodeId,
            > = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_clustered_params: Option<crate::graph::core::BufferNodeId> = None;
            #[cfg(feature = "debug_view")]
            let mut dbg_clustered_records: Option<crate::graph::core::BufferNodeId> = None;

            // Final surface handle, written by the unified PresentPass at the
            // tail of whichever pipeline path runs below.
            let current_surface;

            if is_high_fidelity {
                // ────────────────────────────────────────────────────────────
                // HighFidelity pipeline: separate passes, explicit wiring.
                // ────────────────────────────────────────────────────────────

                // ── Scene Rendering Group ──────────────────────────────────

                // let taa_enabled = self.ctx.camera.aa_mode.is_taa();

                let cas_enabled = if let Some(s) = self.ctx.camera.aa_mode.taa_settings() {
                    s.sharpen_intensity > 0.0
                } else {
                    false
                };

                let fxaa_enabled = self.ctx.camera.aa_mode.is_fxaa();

                let (mut active_color, mut scene_depth, mut scene_hiz) =
                    graph_ctx.with_group("Scene", |c| {
                        let scene_lights = c.with_group("Scene_Lighting", |c| {
                            import_scene_lighting(c, self.ctx.render_lists)
                        });
                        let injected_gpu_lights = self
                            .gpu_local_light_hook
                            .as_mut()
                            .and_then(|hook| c.with_group("Inject_GPU_Local_Lights", |c| hook(c)));
                        let clustered_out = self.ctx.clustered_lighting_pass.add_to_graph(
                            c,
                            ClusteredLightingInputs {
                                enabled: self.ctx.clustered_lighting_enabled,
                                cpu_light_metadata_buffer: scene_lights.light_metadata,
                                cpu_light_data_buffer: scene_lights.light_storage,
                                injected_gpu_lights,
                            },
                        );
                        let scene_lighting = ClusteredScreenBindings {
                            light_metadata: Some(clustered_out.final_light_metadata_buffer),
                            lights: Some(clustered_out.final_light_data_buffer),
                            params: self
                                .ctx
                                .clustered_lighting_enabled
                                .then_some(clustered_out.params_buffer),
                            records: if self.ctx.clustered_lighting_enabled {
                                clustered_out.cluster_records
                            } else {
                                None
                            },
                            light_indices: if self.ctx.clustered_lighting_enabled {
                                clustered_out.light_indices
                            } else {
                                None
                            },
                            atmosphere_transmittance,
                            atmosphere_bake_params,
                        };

                        #[cfg(feature = "debug_view")]
                        {
                            dbg_clustered_params = Some(clustered_out.params_buffer);
                            dbg_clustered_records = clustered_out.cluster_records;
                        }

                        // 1. Prepass
                        let prepass_out = self.ctx.prepass.add_to_graph(
                            c,
                            needs_normal,
                            needs_feature_id,
                            needs_velocity,
                        );

                        let scene_depth = prepass_out.scene_depth;

                        // 2. SSAO
                        let ssao_output = if ssao_enabled {
                            Some(
                                self.ctx.ssao_pass.add_to_graph(
                                    c,
                                    scene_depth,
                                    prepass_out
                                        .scene_normals
                                        .expect("SSAO requires scene normals from Prepass"),
                                ),
                            )
                        } else {
                            None
                        };

                        // 3. Opaque
                        let opaque_out = self.ctx.opaque_pass.add_to_graph(
                            c,
                            scene_depth,
                            self.ctx.extracted_scene.background.clear_color(),
                            ssss_enabled || ssr_enabled,
                            ssgi_enabled || ssr_enabled,
                            ssao_output,
                            shadow_output.shadow_2d,
                            shadow_output.shadow_cube,
                            // env_dependency_base,
                            env_dependency_pmrem,
                            scene_lighting,
                        );

                        let mut active_color = opaque_out.active_color;

                        let scene_hiz =
                            needs_scene_hiz.then(|| self.ctx.hiz_pass.add_to_graph(c, scene_depth));

                        // 4. SSSS
                        if ssss_enabled {
                            if is_msaa {
                                let hdr_desc = TextureDesc::new_2d(
                                    c.frame_config.width,
                                    c.frame_config.height,
                                    crate::HDR_TEXTURE_FORMAT,
                                    wgpu::TextureUsages::RENDER_ATTACHMENT
                                        | wgpu::TextureUsages::TEXTURE_BINDING
                                        | wgpu::TextureUsages::COPY_SRC,
                                );
                                // If MSAA is enabled, resolve the opaque output to an intermediate non-MSAA texture for SSSS input.
                                active_color = add_msaa_resolve_pass(c, active_color, hdr_desc);
                            }

                            active_color = self.ctx.ssss_pass.add_to_graph(
                                c,
                                active_color,
                                prepass_out.scene_depth,
                                prepass_out.scene_normals.unwrap(),
                                prepass_out.feature_id.unwrap(),
                                opaque_out.specular_mrt.unwrap(),
                            );

                            if is_msaa {
                                // If MSAA is enabled, synchronize the SSSS output back to an MSAA texture for downstream passes (Skybox, Transparent) that expect MSAA input.
                                // This avoids redundant MSAA resolve + re-multisample operations.
                                active_color =
                                    self.ctx.msaa_sync_pass.add_to_graph(c, active_color);
                            }
                        }

                        // 5. Skybox
                        if needs_skybox {
                            active_color = self.ctx.skybox_pass.add_to_graph(
                                c,
                                active_color,
                                opaque_out.active_depth,
                                procedural_skybox_dependencies,
                            );
                        }

                        if ssgi_enabled {
                            let taa_history_view = if taa_enabled {
                                self.ctx.taa_pass.history_color_view()
                            } else {
                                None
                            };

                            let ssgi_out = self.ctx.ssgi_pass.add_to_graph(
                                c,
                                active_color,
                                scene_depth,
                                scene_hiz.expect("SSGI requires Hi-Z pyramid"),
                                prepass_out
                                    .scene_normals
                                    .expect("SSGI requires scene normals from Prepass"),
                                prepass_out
                                    .velocity_buffer
                                    .expect("SSGI requires motion vectors from Prepass"),
                                opaque_out
                                    .material_mrt
                                    .expect("SSGI requires opaque material MRT"),
                                env_dependency_pmrem,
                                taa_history_view,
                            );

                            #[cfg(feature = "debug_view")]
                            {
                                dbg_ssgi_raw = Some(ssgi_out.raw_indirect);
                                dbg_ssgi_denoised = Some(ssgi_out.clean_indirect);
                            }

                            active_color = ssgi_out.merged_color;
                        }

                        if ssr_enabled {
                            let ssr_out = self.ctx.ssr_pass.add_to_graph(
                                c,
                                active_color,
                                scene_depth,
                                scene_hiz.expect("SSR requires Hi-Z pyramid"),
                                prepass_out
                                    .scene_normals
                                    .expect("SSR requires scene normals from Prepass"),
                                prepass_out
                                    .velocity_buffer
                                    .expect("SSR requires motion vectors from Prepass"),
                                opaque_out
                                    .material_mrt
                                    .expect("SSR requires opaque material MRT"),
                                opaque_out
                                    .specular_mrt
                                    .expect("SSR requires opaque specular MRT"),
                            );

                            #[cfg(feature = "debug_view")]
                            {
                                dbg_ssr_raw = Some(ssr_out.raw_reflection);
                                dbg_ssr_resolved = Some(ssr_out.clean_reflection);
                                dbg_ssr_trace_diagnostic = ssr_out.raw_diagnostic;
                            }

                            active_color = ssr_out.merged_color;
                        }

                        // ── 6. TAA Resolve ────────────────────────────────────────────
                        // Resolve temporal anti-aliasing before bloom/tone-mapping.
                        // The resolved colour replaces post_transparent_color for
                        // downstream post-processing.
                        if taa_enabled && let Some(velocity) = prepass_out.velocity_buffer {
                            c.with_group("TAA_System", |c| {
                                active_color = self.ctx.taa_pass.add_to_graph(
                                    c,
                                    active_color,
                                    velocity,
                                    scene_depth,
                                );

                                // ── 6b. CAS (Contrast Adaptive Sharpening) ────────────
                                // Recover fine detail lost to temporal filtering.
                                if cas_enabled {
                                    active_color = self.ctx.cas_pass.add_to_graph(c, active_color);
                                }
                            });
                        }

                        #[cfg(feature = "3dgs")]
                        {
                            // 7a. Gaussian Splatting
                            c.with_group("3D_Gaussian_Splatting", |c| {
                                active_color = self.ctx.gaussian_splatting_pass.add_to_graph(
                                    c,
                                    active_color,
                                    opaque_out.active_depth,
                                );
                            });
                        }

                        // 7. Transmission Copy
                        let transmission_tex = if has_transmission {
                            Some(
                                self.ctx
                                    .transmission_copy_pass
                                    .add_to_graph(c, active_color),
                            )
                        } else {
                            None
                        };

                        // 8. Transparent
                        let active_color = self.ctx.transparent_pass.add_to_graph(
                            c,
                            active_color,
                            opaque_out.active_depth,
                            transmission_tex,
                            ssao_output,
                            shadow_output.shadow_2d,
                            shadow_output.shadow_cube,
                            scene_lighting,
                        );

                        // Capture intermediate IDs for debug view resolution.
                        #[cfg(feature = "debug_view")]
                        {
                            dbg_normals = prepass_out.scene_normals;
                            dbg_velocity = prepass_out.velocity_buffer;
                            dbg_ssao = ssao_output;
                        }

                        (active_color, scene_depth, scene_hiz)
                    });

                // ── Before-Post-Process Hooks ──────────────────────────────
                {
                    let mut blackboard = GraphBlackboard {
                        scene_color: Some(active_color),
                        scene_depth: Some(scene_depth),
                        scene_hiz,
                        atmosphere_transmittance,
                        atmosphere_bake_params,
                        surface_out,
                    };
                    for (stage, hook_opt) in &mut self.hooks {
                        if *stage == HookStage::BeforePostProcess
                            && let Some(hook) = hook_opt.take()
                        {
                            blackboard = hook(graph_ctx.graph, blackboard);
                        }
                    }

                    active_color = blackboard.scene_color.unwrap_or(active_color);
                    scene_depth = blackboard.scene_depth.unwrap_or(scene_depth);
                    scene_hiz = blackboard.scene_hiz;
                }

                // ── Post-Processing Group ──────────────────────────────────
                // Every stage renders into a transient LDR target.  The unified
                // PresentPass appended below is the sole writer of the surface,
                // so no post-process pass needs to know whether it happens to be
                // the final stage.
                #[cfg_attr(not(feature = "debug_view"), allow(unused_mut))]
                let mut current_target = graph_ctx.with_group("PostProcess", |ctx| {
                    // Bloom (internally flattened into Bloom_System subgroup)
                    if bloom_enabled {
                        active_color = self.ctx.bloom_pass.add_to_graph(
                            ctx,
                            active_color,
                            self.ctx.scene.bloom.karis_average,
                            self.ctx.scene.bloom.max_mip_levels(),
                        );
                    }

                    // ToneMapping: HDR → LDR, into a transient target.
                    active_color = self.ctx.tone_map_pass.add_to_graph(ctx, active_color);

                    // FXAA: anti-alias the LDR result into a second transient
                    // target.
                    if fxaa_enabled {
                        active_color = self.ctx.fxaa_pass.add_to_graph(ctx, active_color);
                    }

                    bb_scene_color = Some(active_color);
                    bb_scene_depth = Some(scene_depth);
                    bb_scene_hiz = scene_hiz;

                    active_color
                });

                // ── Debug View Override ────────────────────────────────────
                // Resolve the semantic DebugViewMode to a concrete
                // TextureNodeId, then blit it onto the surface.  Targets
                // whose producer was disabled (e.g. SSAO off) safely
                // resolve to None — no pass is injected.
                // Material-override modes are handled separately via shader
                // defines and do not use this post-process path.
                #[cfg(feature = "debug_view")]
                {
                    use crate::graph::render_state::DebugViewTarget;
                    let target = DebugViewTarget::from_mode(self.ctx.render_state.debug_view_mode);
                    let source: Option<crate::graph::core::TextureNodeId> = match target {
                        DebugViewTarget::SceneNormal => dbg_normals,
                        DebugViewTarget::Velocity => dbg_velocity,
                        DebugViewTarget::SsaoRaw => dbg_ssao,
                        DebugViewTarget::SceneDepth => Some(scene_depth),
                        DebugViewTarget::ClusterHeatmap => Some(scene_depth),
                        DebugViewTarget::SsgiRaw => dbg_ssgi_raw,
                        DebugViewTarget::SsgiDenoised => dbg_ssgi_denoised,
                        DebugViewTarget::SsrRaw => dbg_ssr_raw,
                        DebugViewTarget::SsrResolved => dbg_ssr_resolved,
                        DebugViewTarget::SsrTraceDebug => dbg_ssr_trace_diagnostic,
                        _ => None,
                    };

                    let is_depth = target.is_depth();

                    if let Some(src) = source {
                        let clustered = if target == DebugViewTarget::ClusterHeatmap {
                            ClusteredScreenBindings {
                                light_metadata: None,
                                lights: None,
                                params: dbg_clustered_params,
                                records: dbg_clustered_records,
                                light_indices: None,
                                atmosphere_transmittance: None,
                                atmosphere_bake_params: None,
                            }
                        } else {
                            ClusteredScreenBindings::default()
                        };

                        current_target = self.ctx.debug_view_pass.add_to_graph(
                            &mut graph_ctx,
                            src,
                            current_target,
                            is_depth,
                            clustered,
                        );
                    }
                }

                // ── Present ────────────────────────────────────────────────
                // Forward the final LDR target onto the swap-chain surface.  In
                // the common case (target already matches the surface format and
                // size) the graph compiler folds this blit away, rewiring the
                // last real pass to target the surface directly.
                current_surface =
                    self.ctx
                        .present_pass
                        .add_to_graph(&mut graph_ctx, current_target, surface_out);
            } else {
                // BasicForward pipeline: single-pass LDR rendering.

                let prepared_skybox = if needs_skybox {
                    let skybox_pipeline = self.ctx.skybox_pass.current_pipeline;
                    let skybox_bind_group = &self.ctx.skybox_pass.current_bind_group;

                    if let (Some(pipeline_id), Some(bg)) = (skybox_pipeline, skybox_bind_group) {
                        Some(PreparedSkyboxDraw {
                            pipeline: self.ctx.pipeline_cache.get_render_pipeline(pipeline_id),
                            bind_group: bg,
                            sampled_textures: procedural_skybox_dependencies,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                let scene_color = graph_ctx.with_group("BasicForward", |c| {
                    let scene_lights = c.with_group("Scene_Lighting", |c| {
                        import_scene_lighting(c, self.ctx.render_lists)
                    });
                    let injected_gpu_lights = self
                        .gpu_local_light_hook
                        .as_mut()
                        .and_then(|hook| c.with_group("Inject_GPU_Local_Lights", |c| hook(c)));
                    let clustered_out = self.ctx.clustered_lighting_pass.add_to_graph(
                        c,
                        ClusteredLightingInputs {
                            enabled: self.ctx.clustered_lighting_enabled,
                            cpu_light_metadata_buffer: scene_lights.light_metadata,
                            cpu_light_data_buffer: scene_lights.light_storage,
                            injected_gpu_lights,
                        },
                    );
                    let scene_lighting = ClusteredScreenBindings {
                        light_metadata: Some(clustered_out.final_light_metadata_buffer),
                        lights: Some(clustered_out.final_light_data_buffer),
                        params: self
                            .ctx
                            .clustered_lighting_enabled
                            .then_some(clustered_out.params_buffer),
                        records: if self.ctx.clustered_lighting_enabled {
                            clustered_out.cluster_records
                        } else {
                            None
                        },
                        light_indices: if self.ctx.clustered_lighting_enabled {
                            clustered_out.light_indices
                        } else {
                            None
                        },
                        atmosphere_transmittance,
                        atmosphere_bake_params,
                    };
                    self.ctx.simple_forward_pass.add_to_graph(
                        c,
                        self.ctx.extracted_scene.background.clear_color(),
                        prepared_skybox,
                        shadow_output.shadow_2d,
                        shadow_output.shadow_cube,
                        env_dependency_pmrem,
                        scene_lighting,
                    )
                });

                // ── Present ────────────────────────────────────────────────
                // Forward the rendered LDR image onto the swap-chain surface
                // (folded away at compile time in the common case).
                current_surface =
                    self.ctx
                        .present_pass
                        .add_to_graph(&mut graph_ctx, scene_color, surface_out);
            }

            // drop(graph_ctx);

            // ── After-Post-Process Hooks (UI, debug overlays) ──────────────
            {
                let mut blackboard = GraphBlackboard {
                    scene_color: bb_scene_color,
                    scene_depth: bb_scene_depth,
                    scene_hiz: bb_scene_hiz,
                    atmosphere_transmittance,
                    atmosphere_bake_params,
                    surface_out: current_surface,
                };
                for (stage, hook_opt) in &mut self.hooks {
                    if *stage == HookStage::AfterPostProcess
                        && let Some(hook) = hook_opt.take()
                    {
                        blackboard = hook(&mut graph, blackboard);
                    }
                }
            }

            // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            graph.compile(self.ctx.transient_pool, &self.ctx.wgpu_ctx.device);

            // ─── 3a. RDG Prepare: transient-only BindGroup assembly ────────
            //
            // Only the swapchain surface is truly external — all other textures
            // (scene_color, scene_depth, etc.) are RDG transient resources.

            let mut prepare_ctx = PrepareContext {
                views: ViewResolver {
                    resources: &graph.storage.resources,
                    pool: self.ctx.transient_pool,
                },
                device: &self.ctx.wgpu_ctx.device,
                queue: &self.ctx.wgpu_ctx.queue,
                pipeline_cache: self.ctx.pipeline_cache,
                sampler_registry: &self.ctx.resource_manager.sampler_registry,
                global_bind_group_cache: self.ctx.global_bind_group_cache,
                system_textures: &self.ctx.resource_manager.system_textures,
            };

            for &pass_idx in &graph.storage.execution_queue {
                let pass = graph.storage.passes[pass_idx].get_pass_mut();
                pass.prepare(&mut prepare_ctx);
            }

            // ─── 3c. Bake render commands ──────────────────────────────────
            //
            // Resolve every asset handle (geometry, material, pipeline) to its
            // physical wgpu reference.  After this point the execute phase is
            // "blind" — it processes only pre-resolved GPU state.
            let prepass_config = if is_high_fidelity {
                Some(crate::graph::bake::PrepassBakeConfig {
                    local_cache: self.ctx.prepass.local_cache(),
                    needs_normal: self.ctx.prepass.needs_normal(),
                    needs_feature_id: self.ctx.prepass.needs_feature_id(),
                    needs_velocity: self.ctx.prepass.needs_velocity(),
                })
            } else {
                None
            };

            let baked_lists = crate::graph::bake::bake_render_lists(
                self.ctx.render_lists,
                self.ctx.resource_manager,
                self.ctx.pipeline_cache,
                &prepass_config,
            );

            // ─── 3d. Execute ───────────────────────────────────────────────

            let mut execute_ctx = ExecuteContext {
                resources: &graph.storage.resources,
                pool: self.ctx.transient_pool,
                device: &self.ctx.wgpu_ctx.device,
                queue: &self.ctx.wgpu_ctx.queue,
                pipeline_cache: self.ctx.pipeline_cache,
                global_bind_group_cache: self.ctx.global_bind_group_cache,
                mipmap_generator: &self.ctx.resource_manager.mipmap_generator,
                baked_lists: &baked_lists,
                wgpu_ctx: &*self.ctx.wgpu_ctx,
                current_timeline_index: 0,
            };

            let mut encoder =
                self.ctx
                    .wgpu_ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Unified Encoder"),
                    });

            for (timeline_index, &pass_idx) in graph.storage.execution_queue.iter().enumerate() {
                execute_ctx.current_timeline_index = timeline_index;
                #[cfg(debug_assertions)]
                encoder.push_debug_group(graph.storage.passes[pass_idx].name);
                graph.storage.passes[pass_idx]
                    .get_pass_mut()
                    .execute(&execute_ctx, &mut encoder);
                #[cfg(debug_assertions)]
                encoder.pop_debug_group();
            }

            // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            self.ctx.wgpu_ctx.queue.submit(Some(encoder.finish()));
        };

        if let Some(output) = surface_output {
            self.ctx.wgpu_ctx.queue.present(output);
        }
    }
}
