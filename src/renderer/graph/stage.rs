//! Render Stage Definitions
//!
//! `RenderStage` defines the standard stage ordering of the render pipeline,
//! allowing users to insert custom render nodes at specified stages.

/// Render stage enumeration.
///
/// Defines the execution order of the render pipeline. Each stage may contain
/// multiple render nodes; nodes within the same stage execute in insertion order.
///
/// # Stage Overview
///
/// | Stage | Purpose | Typical Content |
/// |-------|---------|------------------|
/// | `PreProcess` | Resource upload, compute pre-processing | BRDF LUT generation, IBL pre-filtering |
/// | `ShadowMap` | Shadow map rendering | Cascaded shadows, point-light shadows |
/// | `Opaque` | Opaque object rendering | Forward / Deferred rendering |
/// | `Skybox` | Skybox rendering | Environment maps, procedural sky |
/// | `Transparent` | Translucent object rendering | Alpha-blended objects |
/// | `PostProcess` | Post-processing effects | ToneMapping, Bloom, FXAA |
/// | `UI` | User interface | egui, debug overlays |
///
/// # Example
///
/// ```ignore
/// // Insert an outline effect at the PostProcess stage
/// frame_builder.add_node(RenderStage::PostProcess, &outline_pass);
///
/// // Insert egui rendering at the UI stage
/// frame_builder.add_node(RenderStage::UI, &ui_pass);
/// ```
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
#[repr(u8)]
pub enum RenderStage {
    /// Pre-processing stage: resource upload, compute shader pre-processing.
    ///
    /// Suitable for: BRDF LUT generation, IBL pre-filtering, GPU particle computation.
    PreProcess = 0,

    /// Shadow map rendering stage.
    ///
    /// Suitable for: directional-light shadows, point-light shadows, cascaded shadow maps.
    ShadowMap = 1,

    /// Opaque object rendering stage (G-Buffer or Forward).
    ///
    /// Suitable for: standard PBR rendering, Deferred G-Buffer fill.
    Opaque = 2,

    /// Skybox rendering stage.
    ///
    /// Suitable for: cubemap skybox, procedural sky, atmospheric scattering.
    Skybox = 3,

    /// Pre-transparent stage (executes before Transparent).
    ///
    /// Suitable for: effects that must run before translucent objects — also the final opaque sub-stage, e.g. SSSSS, Transmission Copy.
    BeforeTransparent = 4,

    /// Translucent object rendering stage.
    ///
    /// Suitable for: alpha-blended objects, particle systems, glass / water surfaces.
    Transparent = 5,

    /// Post-processing stage.
    ///
    /// Suitable for: tone mapping, bloom, depth of field, FXAA / TAA.
    PostProcess = 6,

    /// User interface stage (executed last).
    ///
    /// Suitable for: egui, ImGui, debug overlays.
    UI = 7,
}

impl RenderStage {
    /// Returns the numeric index of the stage (used for sorting).
    #[inline]
    #[must_use]
    pub const fn order(self) -> u8 {
        self as u8
    }

    /// Stage name (for debugging).
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PreProcess => "PreProcess",
            Self::ShadowMap => "ShadowMap",
            Self::Opaque => "Opaque",
            Self::Skybox => "Skybox",
            Self::BeforeTransparent => "BeforeTransparent",
            Self::Transparent => "Transparent",
            Self::PostProcess => "PostProcess",
            Self::UI => "UI",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_ordering() {
        assert!(RenderStage::PreProcess < RenderStage::ShadowMap);
        assert!(RenderStage::ShadowMap < RenderStage::Opaque);
        assert!(RenderStage::Opaque < RenderStage::Skybox);
        assert!(RenderStage::Skybox < RenderStage::Transparent);
        assert!(RenderStage::Transparent < RenderStage::PostProcess);
        assert!(RenderStage::PostProcess < RenderStage::UI);
    }
}
