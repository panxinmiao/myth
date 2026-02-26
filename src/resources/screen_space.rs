//! Screen Space Materials & SSSSS (Screen-Space Sub-Surface Scattering)
//!
//! This module defines the data infrastructure for the **Thin G-Buffer Hybrid Pipeline**:
//! a lightweight mixed rendering mechanism that enables physically correct screen-space
//! effects (SSS, SSR, etc.) without the full bandwidth overhead of a fat deferred G-Buffer.
//!
//! # Design Philosophy
//!
//! Instead of a fat G-Buffer carrying all material parameters through textures,
//! we:
//! 1. Store complex material configs as `ScreenSpaceProfile` **assets** accessible by handle.
//! 2. During **Extract**, map each live profile to a compact per-frame `u8` GPU ID (1–254).
//! 3. Write only that 8-bit ID into the `Normal.a` channel during the **Prepass**.
//!    Alpha = 0.0 → background, Alpha = 255/255 → valid geometry (no SS effects),
//!    Alpha ∈ [1/255 … 254/255] → SS-effect geometry with profile ID encoded as `round(a*255)`.
//! 4. A downstream **SSSSS Pass** reads the ID, fetches the full profile from a 12 KB
//!    GPU `StorageBuffer`, and executes the expensive kernel *only* on matching pixels.
//!
//! # Feature Flags (Bitmask)
//!
//! | Constant         | Value  | Description                            |
//! |------------------|--------|----------------------------------------|
//! | `FEATURE_NONE`   | `0x00` | No screen-space effects                |
//! | `FEATURE_SSS`    | `0x01` | Sub-surface scattering (skin, wax, …)  |
//! | `FEATURE_SSR`    | `0x02` | Screen-space reflections (reserved)    |
//!
//! # GPU Data Layout
//!
//! `ScreenSpaceMaterialData` is the element type of the global **Storage Buffer**
//! (`array<ScreenSpaceMaterialData, 256>`).  The first element (index 0) is always
//! the default "no effects" sentinel; real profiles occupy indices 1–254.
//!
//! # Limits
//!
//! - Max **254** distinct active `ScreenSpaceProfile` instances per frame.
//!   Exceeding this limit causes a `warn!` log and graceful fallback (ID 0 = no effect).
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use myth::resources::screen_space::ScreenSpaceProfile;
//! use myth::assets::ScreenSpaceProfileHandle;
//!
//! // Create a realistic skin SSS profile
//! let skin_profile = ScreenSpaceProfile::new_sss(
//!     Vec3::new(0.9, 0.3, 0.15),  // scatter color (reddish-pink)
//!     0.012,                        // radius in world units
//! );
//! let handle: ScreenSpaceProfileHandle = assets.screen_space_profiles.add(skin_profile);
//!
//! // Attach to a material
//! material.with_screen_space_profile(handle);
//! ```

use glam::Vec3;

// ============================================================================
// Feature Flags (CPU + GPU shared constants)
// ============================================================================

/// No screen-space effects.
pub const FEATURE_NONE: u32 = 0x00;

/// Sub-Surface Scattering (SSS) effect bit.
///
/// When set in `ScreenSpaceProfile.feature_flags`, the prepass will write the
/// `FEATURE_SSS` bit into the hardware stencil buffer, allowing the `SssssPass`
/// to use `StencilFaceState { compare: Equal }` to skip non-skin pixels entirely.
pub const FEATURE_SSS: u32 = 1 << 0; // 0x01

/// Screen-Space Reflections (SSR) – reserved for future use.
pub const FEATURE_SSR: u32 = 1 << 1; // 0x02

/// Stencil write/read mask: lower 4 bits reserved for screen-space feature flags.
///
/// Using a mask prevents the screen-space bits from conflicting with other
/// engine systems that might use upper stencil bits (e.g., outline rendering).
pub const STENCIL_FEATURE_MASK: u32 = 0x0F;

// ============================================================================
// ScreenSpaceMaterialData — GPU Storage Buffer Element
// ============================================================================

/// GPU-side representation of a `ScreenSpaceProfile`, laid out to satisfy
/// WebGPU's **16-byte alignment** requirement for `array<T>` elements in a
/// Storage Buffer.
///
/// # Memory Layout (48 bytes total)
///
/// | Field              | Offset | Size | Description                         |
/// |--------------------|--------|------|-------------------------------------|
/// | `feature_flags`    |   0    |  4   | Bitmask of active SS features       |
/// | `_padding`         |   4    | 12   | 16-byte alignment pad               |
/// | `data_0`           |  16    | 16   | vec4: SSS scatter (rgb) + radius    |
/// | `data_1`           |  32    | 16   | vec4: reserved (SSR roughness, …)   |
///
/// Rust's `#[repr(C)]` ensures the layout matches the WGSL declaration.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct ScreenSpaceMaterialData {
    /// Bitmask combining all active screen-space features for this material.
    pub feature_flags: u32,
    /// 12-byte padding to align the following vec4 fields to 16 bytes.
    pub _padding: [u32; 3],
    /// General purpose 4-float slot.
    /// For SSS: `[r, g, b, scatter_radius_world]`.
    pub data_0: [f32; 4],
    /// General purpose 4-float slot (reserved).
    /// For SSR: roughness or other parameters.
    pub data_1: [f32; 4],
}

impl Default for ScreenSpaceMaterialData {
    #[inline]
    fn default() -> Self {
        Self {
            feature_flags: FEATURE_NONE,
            _padding: [0; 3],
            data_0: [0.0; 4],
            data_1: [0.0; 4],
        }
    }
}

/// Generates the WGSL struct definition for `ScreenSpaceMaterialData`.
///
/// Used by `SssssPass` to inject the type into its shader.
impl ScreenSpaceMaterialData {
    #[must_use]
    pub fn wgsl_struct_def() -> &'static str {
        r#"struct ScreenSpaceMaterialData {
    feature_flags: u32,
    _padding_0: u32,
    _padding_1: u32,
    _padding_2: u32,
    data_0: vec4<f32>,
    data_1: vec4<f32>,
};"#
    }
}

// ============================================================================
// ScreenSpaceProfile — User-Facing Asset
// ============================================================================

/// A reusable **screen-space effect configuration** asset.
///
/// Decouples expensive screen-space parameters from the per-draw-call material
/// uniform.  Attach one to a `MeshPhysicalMaterial` via
/// `with_screen_space_profile(handle)` to opt into one or more SS effects.
///
/// # Supported Effects
///
/// - **SSS** (`FEATURE_SSS`): Create via [`ScreenSpaceProfile::new_sss`].
/// - **SSR** (`FEATURE_SSR`): Reserved for a future update.
///
/// # Example
///
/// ```rust,ignore
/// let jade = ScreenSpaceProfile::new_sss(Vec3::new(0.6, 0.9, 0.6), 0.008);
/// let handle = assets.screen_space_profiles.add(jade);
/// material.with_screen_space_profile(handle);
/// ```
#[derive(Clone, Debug)]
pub struct ScreenSpaceProfile {
    /// Bitmask of active screen-space features.
    pub feature_flags: u32,
    /// Slot 0 — interpretation depends on active features:
    /// - SSS: `[scatter_r, scatter_g, scatter_b, radius_world_units]`
    pub data_0: [f32; 4],
    /// Slot 1 — reserved; currently unused.
    pub data_1: [f32; 4],
}

impl ScreenSpaceProfile {
    /// Creates a **Sub-Surface Scattering** profile.
    ///
    /// # Parameters
    /// - `scatter_color`: RGB color of the scattered light.  Values between
    ///   `(0.7, 0.3, 0.1)` (skin) and `(0.6, 0.9, 0.6)` (jade) are typical.
    /// - `radius`: World-space scatter radius in meters.  Typical skin ≈ 0.01 m.
    ///
    /// # Example
    /// ```rust,ignore
    /// let skin = ScreenSpaceProfile::new_sss(Vec3::new(0.9, 0.3, 0.15), 0.012);
    /// ```
    #[must_use]
    pub fn new_sss(scatter_color: Vec3, radius: f32) -> Self {
        Self {
            feature_flags: FEATURE_SSS,
            data_0: [scatter_color.x, scatter_color.y, scatter_color.z, radius],
            data_1: [0.0; 4],
        }
    }

    /// Converts this profile to its compact GPU representation.
    #[must_use]
    #[inline]
    pub fn to_gpu_data(&self) -> ScreenSpaceMaterialData {
        ScreenSpaceMaterialData {
            feature_flags: self.feature_flags,
            _padding: [0; 3],
            data_0: self.data_0,
            data_1: self.data_1,
        }
    }
}

/// Per-pass uniforms for the SSSSS (Screen-Space Sub-Surface Scattering) blur shader.
///
/// Uploaded to a `wgpu::Buffer` (UNIFORM usage) and bound at `group(0) @binding(0)`.
///
/// # Memory Layout (32 bytes — one vec4 + one vec4)
///
/// | Field         | Offset | Size | Description                              |
/// |---------------|--------|------|------------------------------------------|
/// | `texel_size`  |   0    |  8   | `[1/width, 1/height]`                    |
/// | `direction`   |   8    |  8   | `[1,0]` = horizontal; `[0,1]` = vertical |
/// | `aspect_ratio`|  16    |  4   | `width / height` (X blur correction)     |
/// | `_padding`    |  20    | 12   | Pad to 32 bytes                          |
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsssUniforms {
    /// Per-texel UV size: `[1/width, 1/height]`.
    pub texel_size: [f32; 2],
    /// Blur direction for the current pass: `[1,0]` (horizontal) or `[0,1]` (vertical).
    pub direction: [f32; 2],
    /// Aspect ratio `width / height`, used by the horizontal pass to keep the
    /// physical blur radius equal between the H and V sub-passes.
    pub aspect_ratio: f32,
    /// Reserved; must be zero.
    pub _padding: [f32; 3],
}

// ============================================================================
// ScreenSpaceSettings — Scene-Level Toggle
// ============================================================================

/// Per-scene toggle for screen-space effects.
///
/// Add `scene.screen_space.enable_sss = true` to activate the SSSSS pass.
/// When `enable_sss` is `false` (the default), the entire feature has
/// **zero impact** on the render frame: no prepass changes, no extra passes,
/// no extra GPU memory.
///
/// # Example
///
/// ```rust,ignore
/// scene.screen_space.enable_sss = true;
/// ```
#[derive(Default, Clone, Debug)]
pub struct ScreenSpaceSettings {
    /// Activates the Screen-Space Sub-Surface Scattering pass.
    ///
    /// Requires `HighFidelity` render path and a depth-normal prepass.
    /// Setting this to `true` will:
    /// 1. Enable writing of `screen_space_id` into `Normal.a` during the prepass.
    /// 2. Enable stencil writes for SSS pixels (if depth format supports stencil).
    /// 3. Inject a ping-pong SSSSS blur pass after the opaque rendering.
    pub enable_sss: bool,
}
