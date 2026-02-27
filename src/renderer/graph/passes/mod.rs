//! Render Pass Implementations
//!
//! Contains various concrete render pass implementations.
//!
//! # Pass Classification
//!
//! ## Data Preparation Pass
//! - [`SceneCullPass`][]: Scene culling, command generation, and sorting
//!
//! ## Render Pass (Simple Path - LDR)
//! - [`SimpleForwardPass`][]: Single pass rendering for opaque and transparent objects
//!
//! ## Render Pass (PBR Path - HDR)
//! - [`OpaquePass`][]: Renders only opaque objects
//! - [`TransmissionCopyPass`][]: Copies scene color for Transmission usage
//! - [`TransparentPass`][]: Renders only transparent objects
//!
//! ## Sky/Background Pass
//! - [`SkyboxPass`]: Renders skybox/gradient/panoramic background
//!
//! ## Compute Pass
//! - [`BRDFLutComputePass`]: Precomputes BRDF LUT
//! - [`IBLComputePass`]: Pre-filters IBL
//!
//! ## Post-Processing Pass
//! - [`ToneMapPass`]: Tone mapping (HDR → LDR)
//! - [`BloomPass`]: Physically-based bloom (HDR)
//! - [`FxaaPass`]: Fast Approximate Anti-Aliasing (LDR)
//!
//! # Render Pipeline Topology
//!
//! ## Simple Path (LDR/Fast)
//! ```text
//! SceneCullPass → SimpleForwardPass → Surface
//! ```
//!
//! ## PBR Path (HDR/Physical)
//! ```text
//! SceneCullPass → OpaquePass → [SkyboxPass] → [TransmissionCopyPass] → TransparentPass → [BloomPass] → ToneMapPass → [FxaaPass] → Surface
//! ```
mod bloom;
mod brdf_lut_compute;
mod cull;
mod fxaa;
mod ibl_compute;
mod opaque;
mod prepass;
mod shadow;
mod simple_forward;
mod skybox;
mod ssao;
mod ssss;
mod tone_mapping;
mod transmission_copy;
mod transparent;

pub use bloom::BloomPass;
pub use brdf_lut_compute::BRDFLutComputePass;
pub use cull::SceneCullPass;
pub use fxaa::FxaaPass;
pub use ibl_compute::IBLComputePass;
pub use opaque::OpaquePass;
pub use prepass::DepthNormalPrepass;
pub use shadow::ShadowPass;
pub use simple_forward::SimpleForwardPass;
pub use skybox::SkyboxPass;
pub use ssao::SsaoPass;
pub use ssss::SssssPass;
pub use tone_mapping::ToneMapPass;
pub use transmission_copy::TransmissionCopyPass;
pub use transparent::TransparentPass;
