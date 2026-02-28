//! Strongly-typed pipeline cache keys.
//!
//! `wgpu` descriptor types (`ColorTargetState`, `DepthStencilState`, …) do not
//! implement `Hash` / `Eq`. This module defines *mirror* types that extract the
//! fields relevant for pipeline identity and derive the correct trait impls.
//!
//! Three key families are provided:
//!
//! - [`GraphicsPipelineKey`] — material-driven scene geometry pipelines
//!   (opaque, transparent, shadow).
//! - [`FullscreenPipelineKey`] — post-processing / fullscreen passes
//!   (bloom, SSAO, FXAA, tone map, SSSSS, skybox, prepass…).
//! - [`ComputePipelineKey`] — compute shader pipelines (BRDF LUT, IBL).

use std::hash::{Hash, Hasher};

// ─── Hashable Mirror Types ────────────────────────────────────────────────────

/// Hashable mirror of `wgpu::BlendComponent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponentKey {
    pub src_factor: wgpu::BlendFactor,
    pub dst_factor: wgpu::BlendFactor,
    pub operation: wgpu::BlendOperation,
}

impl From<wgpu::BlendComponent> for BlendComponentKey {
    fn from(b: wgpu::BlendComponent) -> Self {
        Self {
            src_factor: b.src_factor,
            dst_factor: b.dst_factor,
            operation: b.operation,
        }
    }
}

/// Hashable mirror of `wgpu::BlendState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendStateKey {
    pub color: BlendComponentKey,
    pub alpha: BlendComponentKey,
}

impl From<wgpu::BlendState> for BlendStateKey {
    fn from(b: wgpu::BlendState) -> Self {
        Self {
            color: b.color.into(),
            alpha: b.alpha.into(),
        }
    }
}

/// Hashable mirror of `wgpu::ColorTargetState`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColorTargetKey {
    pub format: wgpu::TextureFormat,
    pub blend: Option<BlendStateKey>,
    pub write_mask: u32, // wgpu::ColorWrites bits
}

impl From<wgpu::ColorTargetState> for ColorTargetKey {
    fn from(c: wgpu::ColorTargetState) -> Self {
        Self {
            format: c.format,
            blend: c.blend.map(Into::into),
            write_mask: c.write_mask.bits(),
        }
    }
}

/// Hashable mirror of `wgpu::StencilFaceState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceKey {
    pub compare: wgpu::CompareFunction,
    pub fail_op: wgpu::StencilOperation,
    pub depth_fail_op: wgpu::StencilOperation,
    pub pass_op: wgpu::StencilOperation,
}

impl From<wgpu::StencilFaceState> for StencilFaceKey {
    fn from(s: wgpu::StencilFaceState) -> Self {
        Self {
            compare: s.compare,
            fail_op: s.fail_op,
            depth_fail_op: s.depth_fail_op,
            pass_op: s.pass_op,
        }
    }
}

/// Hashable mirror of `wgpu::StencilState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilStateKey {
    pub front: StencilFaceKey,
    pub back: StencilFaceKey,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl From<wgpu::StencilState> for StencilStateKey {
    fn from(s: wgpu::StencilState) -> Self {
        Self {
            front: s.front.into(),
            back: s.back.into(),
            read_mask: s.read_mask,
            write_mask: s.write_mask,
        }
    }
}

/// Hashable mirror of `wgpu::DepthBiasState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthBiasKey {
    pub constant: i32,
    pub slope_scale_bits: u32,
    pub clamp_bits: u32,
}

impl From<wgpu::DepthBiasState> for DepthBiasKey {
    fn from(b: wgpu::DepthBiasState) -> Self {
        Self {
            constant: b.constant,
            slope_scale_bits: b.slope_scale.to_bits(),
            clamp_bits: b.clamp.to_bits(),
        }
    }
}

/// Hashable mirror of `wgpu::DepthStencilState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilKey {
    pub format: wgpu::TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub stencil: StencilStateKey,
    pub bias: DepthBiasKey,
}

impl From<wgpu::DepthStencilState> for DepthStencilKey {
    fn from(d: wgpu::DepthStencilState) -> Self {
        Self {
            format: d.format,
            depth_write_enabled: d.depth_write_enabled,
            depth_compare: d.depth_compare,
            stencil: d.stencil.into(),
            bias: d.bias.into(),
        }
    }
}

/// Hashable mirror of `wgpu::MultisampleState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleKey {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl From<wgpu::MultisampleState> for MultisampleKey {
    fn from(m: wgpu::MultisampleState) -> Self {
        Self {
            count: m.count,
            mask: m.mask,
            alpha_to_coverage_enabled: m.alpha_to_coverage_enabled,
        }
    }
}

// ─── Pipeline Keys ────────────────────────────────────────────────────────────

/// L2 cache key for material-driven scene geometry pipelines.
///
/// This is the successor to the old `PipelineKey`. It fully describes all
/// wgpu pipeline state that is relevant for deduplication. The `shader_hash`
/// collapses the (template + defines) tuple into a single `u64`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineKey {
    pub shader_hash: u64,
    pub vertex_layout_id: u64,
    /// `[Global, Material, Object, Screen]`
    pub bind_group_layout_ids: [u64; 4],
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub front_face: wgpu::FrontFace,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<BlendStateKey>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
    pub alpha_to_coverage: bool,
    pub is_specular_split: bool,
}

/// L2 cache key for post-processing / fullscreen passes.
///
/// Unlike `GraphicsPipelineKey`, these pipelines use a fixed fullscreen
/// triangle vertex layout (no vertex buffers) and a simpler bind-group
/// structure. The `shader_hash` is an xxh3-128 of the final WGSL source,
/// truncated to `u64` when used as a lookup key and stored as full `u128`
/// inside the shader manager.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FullscreenPipelineKey {
    /// Truncated xxh3-128 hash of the final WGSL source code.
    pub shader_hash: u128,
    /// Color render targets (usually 1).
    pub color_targets: smallvec::SmallVec<[ColorTargetKey; 2]>,
    /// Depth/stencil configuration (optional).
    pub depth_stencil: Option<DepthStencilKey>,
    /// Multisample state.
    pub multisample: MultisampleKey,
    /// Primitive topology (usually `TriangleList`).
    pub topology: wgpu::PrimitiveTopology,
    /// Cull mode (usually `None` for fullscreen).
    pub cull_mode: Option<wgpu::Face>,
    /// Front face winding (usually `Ccw`).
    pub front_face: wgpu::FrontFace,
}

/// L2 cache key for compute pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputePipelineKey {
    /// xxh3-128 hash of the final WGSL source code.
    pub shader_hash: u128,
}

// ─── Shadow Pipeline Key (unchanged from previous design) ─────────────────────

/// L2 cache key for depth-only shadow pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShadowPipelineKey {
    pub shader_hash: u64,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_format: wgpu::TextureFormat,
    pub front_face: wgpu::FrontFace,
}

// ─── Convenience helpers ──────────────────────────────────────────────────────

/// Compute a `u64` hash of any `Hash`-able value using `FxBuildHasher`.
#[inline]
pub fn fx_hash_key<K: Hash>(key: &K) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}
