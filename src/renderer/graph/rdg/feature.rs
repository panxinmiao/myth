//! Render Feature Infrastructure
//!
//! Defines the [`ExtractContext`] available during Feature resource preparation,
//! typed output structures for multi-output Features, and the [`RenderHook`]
//! trait for external plugin injection.
//!
//! # Architecture
//!
//! Each rendering capability is split into two layers:
//!
//! - **Feature** (persistent, cross-frame): holds cached GPU resources
//!   (pipelines, layouts, noise textures). Lives in [`RendererState`].
//! - **PassNode** (transient, per-frame): pure data carrier created by
//!   `Feature::add_to_graph()` and destroyed after execution.
//!
//! Features prepare persistent resources in [`ExtractContext`] *before* the
//! render graph is built, then inject transient pass nodes via imperative
//! `add_to_graph()` calls that return explicit [`TextureNodeId`] outputs.

use super::blackboard::GraphBlackboard;
use super::graph::RenderGraph;
use super::types::TextureNodeId;
use crate::assets::AssetServer;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::SamplerRegistry;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::frame::RenderLists;
use crate::renderer::graph::{ExtractedScene, RenderState};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};

/// Context available during the Feature **extract & prepare** phase.
///
/// Called once per frame for each active Feature, **before** the render graph
/// is constructed. Provides full access to GPU infrastructure so Features can:
///
/// - Create / cache `wgpu::BindGroupLayout`s
/// - Compile pipelines via [`PipelineCache`]
/// - Upload persistent GPU data (uniform buffers, noise textures, LUTs, etc.)
///
/// After this phase the Feature holds only lightweight IDs and `Arc`-wrapped
/// handles; the heavy infrastructure borrows are released.
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
}

/// Output from [`PrepassFeature::add_to_graph`].
///
/// Carries the depth texture node and optional normal / feature-ID nodes
/// so downstream Features can consume them with full type safety. Using
/// `Option<TextureNodeId>` makes configuration mismatches (e.g. enabling
/// SSAO without normal prepass) detectable at graph-assembly time rather
/// than at GPU execution time.
pub struct PrepassOutput {
    /// Scene depth buffer — always produced.
    pub depth: TextureNodeId,
    /// View-space normals (produced when SSAO or screen-space effects need them).
    pub normal: Option<TextureNodeId>,
    /// Material feature ID (produced when SSS/SSR needs stencil tagging).
    pub feature_id: Option<TextureNodeId>,
}

/// Configuration for the opaque pass builder.
pub struct OpaqueConfig {
    /// Whether a Z-prepass wrote depth data this frame.
    pub has_prepass: bool,
    /// Background clear colour.
    pub clear_color: wgpu::Color,
    /// Whether to output a specular MRT attachment for SSSSS.
    pub needs_specular: bool,
}

/// Output from [`OpaqueFeature::add_to_graph`].
pub struct OpaqueOutput {
    /// HDR scene colour after opaque rendering.
    pub color: TextureNodeId,
    /// Specular MRT texture (produced when `needs_specular` is set).
    pub specular: Option<TextureNodeId>,
}

/// Configuration for the skybox pass builder.
pub struct SkyboxConfig {
    pub background_mode: crate::scene::background::BackgroundMode,
    pub bg_uniforms_cpu_id: u64,
    pub bg_uniforms_gpu_id: u64,
    pub scene_id: u32,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
}

/// A user-supplied callback for injecting custom passes into the render graph.
///
/// External plugins (UI renderers, debug visualisations, custom filters)
/// implement this to participate in the render graph without coupling to
/// internal engine passes. The [`GraphBlackboard`] carries the current
/// frame's resource cursor — hooks may read or modify `current_color` to
/// chain into the main colour pipeline.
pub trait RenderHook {
    /// Insert custom passes into the graph.
    fn build(&mut self, rdg: &mut RenderGraph, bb: &mut GraphBlackboard);
}
