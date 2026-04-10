//! Background Mode & Settings
//!
//! Defines the background rendering mode for a scene and the settings wrapper
//! that owns the GPU uniform buffer for skybox parameters.
//!
//! # Architecture
//!
//! [`BackgroundMode`] is the lightweight enum that describes *what* to draw
//! (solid color, gradient, texture, or procedural sky). [`BackgroundSettings`]
//! wraps the mode together with a `CpuBuffer<SkyboxParamsUniforms>` whose
//! version is automatically bumped only when setter methods actually change a
//! value. The render pass calls `ensure_buffer_id()` in its `prepare()` step
//! and never writes to the buffer — GPU sync happens only when needed.
//!
//! # Supported Modes
//!
//! - [`BackgroundMode::Color`]: Solid color clear (most efficient - uses hardware clear)
//! - [`BackgroundMode::Gradient`]: Vertical gradient (top → bottom)
//! - [`BackgroundMode::Texture`]: Texture-based background (cubemap, equirectangular, planar)
//! - [`BackgroundMode::Procedural`]: Physically-based atmosphere (Hillaire 2020)

use glam::{Vec3, Vec4};

use myth_resources::buffer::CpuBuffer;
use myth_resources::gpu_struct;
use myth_resources::texture::TextureSource;

// ============================================================================
// GPU Uniform Struct
// ============================================================================

/// Skybox per-draw parameters uploaded to the GPU.
///
/// Camera data (view_projection_inverse, camera_position) is obtained from
/// the global bind group's `RenderStateUniforms`, so only skybox-specific
/// values live here.
#[gpu_struct]
pub struct SkyboxParamsUniforms {
    pub color_top: Vec4,
    pub color_bottom: Vec4,
    #[default(0.0)]
    pub rotation: f32,
    #[default(1.0)]
    pub intensity: f32,
}

// ============================================================================
// Procedural Sky Parameters (Hillaire 2020 Atmosphere)
// ============================================================================

/// Physical atmosphere parameters for the Hillaire 2020 sky model.
///
/// Controls the procedural sky rendering and dynamic IBL bake pipeline.
/// When any parameter changes, the atmosphere LUTs and IBL cubemap are
/// regenerated automatically.
///
/// # Default
///
/// The default values produce a visually appealing "golden hour" sky:
/// sun at ~15° elevation with warm tones, suitable as an out-of-the-box
/// lighting environment for PBR models.
#[derive(Clone, Debug, PartialEq)]
pub struct ProceduralSkyParams {
    /// Normalized sun direction vector (world space, pointing toward the sun).
    pub sun_direction: Vec3,

    /// Normalized moon direction vector (world space, pointing toward the moon).
    pub moon_direction: Vec3,

    /// Celestial pole axis used to rotate the star field.
    pub star_axis: Vec3,

    /// Sun disk angular diameter in degrees (Earth's sun ≈ 0.53°).
    pub sun_disk_size: f32,

    /// Moon disk angular diameter in degrees.
    pub moon_disk_size: f32,

    /// Sun luminous intensity multiplier.
    pub sun_intensity: f32,

    /// Moon luminous intensity multiplier.
    pub moon_intensity: f32,

    /// Rayleigh scattering coefficients at sea level (per meter).
    ///
    /// Controls the blue sky color. Earth default: `(5.802e-6, 13.558e-6, 33.1e-6)`.
    pub rayleigh_scattering: Vec3,

    /// Rayleigh scale height in meters.
    ///
    /// Controls how quickly Rayleigh scattering diminishes with altitude.
    /// Earth default: `8000.0`.
    pub rayleigh_scale_height: f32,

    /// Mie scattering coefficient at sea level (scalar, per meter).
    ///
    /// Controls haze and sun halo intensity. Earth default: `3.996e-6`.
    pub mie_scattering: f32,

    /// Mie absorption coefficient at sea level (scalar, per meter).
    ///
    /// Earth default: `4.4e-6`.
    pub mie_absorption: f32,

    /// Mie scale height in meters.
    ///
    /// Controls haze distribution with altitude. Earth default: `1200.0`.
    pub mie_scale_height: f32,

    /// Mie phase function anisotropy factor (Henyey-Greenstein g parameter).
    ///
    /// Range `[-1, 1]`. Higher values produce a brighter, tighter sun halo.
    /// Earth default: `0.8`.
    pub mie_anisotropy: f32,

    /// Ozone absorption coefficients (per meter).
    ///
    /// Earth default: `(0.65e-6, 1.881e-6, 0.085e-6)`.
    pub ozone_absorption: Vec3,

    /// Planet radius in meters. Earth default: `6_360_000.0`.
    pub planet_radius: f32,

    /// Atmosphere radius in meters (ground to top of atmosphere).
    /// Earth default: `6_460_000.0`.
    pub atmosphere_radius: f32,

    /// Exposure multiplier applied to the final sky color.
    pub exposure: f32,

    /// Optional user-provided star-field texture used for the night sky.
    ///
    /// 2D textures are treated as equirectangular panoramas; cube textures are
    /// sampled directly as cubemaps.
    pub starbox_texture: Option<TextureSource>,

    /// Brightness multiplier applied to the star-field texture before exposure.
    pub star_intensity: f32,

    /// Rotation angle in radians applied around `star_axis`.
    pub star_rotation: f32,

    /// Internal version counter, incremented on every mutation.
    version: u64,
}

impl Default for ProceduralSkyParams {
    fn default() -> Self {
        Self::golden_hour()
    }
}

impl ProceduralSkyParams {
    /// A visually appealing "golden hour" preset with the sun at ~15° elevation.
    #[must_use]
    pub fn golden_hour() -> Self {
        let sun_elevation = 15.0_f32.to_radians();
        let sun_direction = Vec3::new(0.0, sun_elevation.sin(), -sun_elevation.cos()).normalize();
        Self {
            sun_direction,
            moon_direction: -sun_direction,
            star_axis: Vec3::Y,
            sun_disk_size: 0.53,
            moon_disk_size: 0.52,
            sun_intensity: 20.0,
            moon_intensity: 0.35,
            rayleigh_scattering: Vec3::new(5.802e-6, 13.558e-6, 33.1e-6),
            rayleigh_scale_height: 8000.0,
            mie_scattering: 3.996e-6,
            mie_absorption: 4.4e-6,
            mie_scale_height: 1200.0,
            mie_anisotropy: 0.8,
            ozone_absorption: Vec3::new(0.65e-6, 1.881e-6, 0.085e-6),
            planet_radius: 6_360_000.0,
            atmosphere_radius: 6_460_000.0,
            exposure: 10.0,
            starbox_texture: None,
            star_intensity: 50.0,
            star_rotation: 0.0,
            version: 0,
        }
    }

    /// Bright midday preset with the sun near zenith.
    #[must_use]
    pub fn midday() -> Self {
        let sun_elevation = 70.0_f32.to_radians();
        Self {
            sun_direction: Vec3::new(0.0, sun_elevation.sin(), -sun_elevation.cos()).normalize(),
            sun_intensity: 20.0,
            exposure: 10.0,
            ..Self::golden_hour()
        }
    }

    /// Sunset preset with the sun at 3° elevation.
    #[must_use]
    pub fn sunset() -> Self {
        let sun_elevation = 3.0_f32.to_radians();
        Self {
            sun_direction: Vec3::new(0.3, sun_elevation.sin(), -sun_elevation.cos()).normalize(),
            sun_intensity: 20.0,
            exposure: 12.0,
            ..Self::golden_hour()
        }
    }

    /// Returns the internal version counter.
    #[inline]
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Sets the sun direction and increments the version.
    pub fn set_sun_direction(&mut self, dir: Vec3) {
        let dir = dir.normalize_or_zero();
        if self.sun_direction != dir {
            self.sun_direction = dir;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the moon direction and increments the version.
    pub fn set_moon_direction(&mut self, dir: Vec3) {
        let dir = dir.normalize_or_zero();
        if self.moon_direction != dir {
            self.moon_direction = dir;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the celestial pole axis used for star rotation.
    pub fn set_star_axis(&mut self, axis: Vec3) {
        let axis = axis.normalize_or_zero();
        if self.star_axis != axis {
            self.star_axis = axis;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the sun intensity and increments the version.
    pub fn set_sun_intensity(&mut self, intensity: f32) {
        if self.sun_intensity != intensity {
            self.sun_intensity = intensity;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the moon intensity and increments the version.
    pub fn set_moon_intensity(&mut self, intensity: f32) {
        if self.moon_intensity != intensity {
            self.moon_intensity = intensity;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the exposure multiplier and increments the version.
    pub fn set_exposure(&mut self, exposure: f32) {
        if self.exposure != exposure {
            self.exposure = exposure;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the star texture intensity and increments the version.
    pub fn set_star_intensity(&mut self, intensity: f32) {
        if self.star_intensity != intensity {
            self.star_intensity = intensity.max(0.0);
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the star-field rotation angle and increments the version.
    pub fn set_star_rotation(&mut self, rotation: f32) {
        if self.star_rotation != rotation {
            self.star_rotation = rotation;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the optional star texture and increments the version.
    pub fn set_starbox_texture(&mut self, starbox_texture: Option<TextureSource>) {
        if self.starbox_texture != starbox_texture {
            self.starbox_texture = starbox_texture;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the Rayleigh scattering coefficients and increments the version.
    pub fn set_rayleigh_scattering(&mut self, coeffs: Vec3) {
        if self.rayleigh_scattering != coeffs {
            self.rayleigh_scattering = coeffs;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the Mie scattering coefficient and increments the version.
    pub fn set_mie_scattering(&mut self, coeff: f32) {
        if self.mie_scattering != coeff {
            self.mie_scattering = coeff;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the Mie phase function anisotropy and increments the version.
    pub fn set_mie_anisotropy(&mut self, g: f32) {
        if self.mie_anisotropy != g {
            self.mie_anisotropy = g.clamp(-1.0, 1.0);
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the sun disk angular size in degrees and increments the version.
    pub fn set_sun_disk_size(&mut self, degrees: f32) {
        if self.sun_disk_size != degrees {
            self.sun_disk_size = degrees;
            self.version = self.version.wrapping_add(1);
        }
    }

    /// Sets the moon disk size in degrees and increments the version.
    pub fn set_moon_disk_size(&mut self, degrees: f32) {
        if self.moon_disk_size != degrees {
            self.moon_disk_size = degrees;
            self.version = self.version.wrapping_add(1);
        }
    }
}

// ============================================================================
// BackgroundMode (lightweight enum — describes *what* to draw)
// ============================================================================

/// Background rendering mode.
///
/// Determines how the scene background is rendered. Each variant maps
/// to a different shader pipeline variant for optimal performance.
///
/// # Performance
///
/// - `Color`: Uses GPU hardware clear — zero draw calls, maximum throughput.
/// - `Gradient`: Renders a fullscreen triangle with per-vertex interpolation.
/// - `Texture`: Renders a fullscreen triangle with texture sampling + ray reconstruction.
#[derive(Clone, Debug)]
pub enum BackgroundMode {
    /// Solid color clear (most efficient).
    ///
    /// Uses the GPU's hardware clear operation — no draw calls needed.
    /// Alpha channel can be used for post-composition (typically 1.0).
    Color(Vec4),

    /// Vertical gradient (sky color → ground color).
    ///
    /// Renders a fullscreen triangle with smooth interpolation based
    /// on view direction's Y component.
    Gradient {
        /// Color at the top of the sky (Y = +1)
        top: Vec4,
        /// Color at the bottom/ground (Y = -1)
        bottom: Vec4,
    },

    /// Texture-based background (skybox / panorama / planar).
    ///
    /// Renders a fullscreen triangle with ray reconstruction from the
    /// depth buffer, sampling the background texture along the view direction.
    Texture {
        /// Texture source (asset handle or attachment)
        source: TextureSource,
        /// Y-axis rotation in radians
        rotation: f32,
        /// Brightness/exposure multiplier
        intensity: f32,
        /// Texture mapping method
        mapping: BackgroundMapping,
    },

    /// Physically-based procedural sky (Hillaire 2020 atmosphere model).
    ///
    /// Renders a photorealistic sky computed from atmospheric scattering
    /// parameters. Automatically generates IBL environment lighting for
    /// PBR rendering. No external texture assets required.
    Procedural(ProceduralSkyParams),
}

/// Texture mapping method for background rendering.
///
/// Each variant produces a different shader pipeline variant
/// (via `ShaderDefines`) to avoid dynamic branching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackgroundMapping {
    /// Standard cubemap sampling.
    ///
    /// The view direction is used directly as a cubemap lookup vector.
    /// Best for pre-processed environment cubemaps.
    Cube,

    /// Equirectangular (latitude-longitude) projection.
    ///
    /// Maps the view direction to UV coordinates using `atan2` / `asin`.
    /// Best for 360° panoramic HDR images.
    Equirectangular,

    /// Planar screen-space mapping.
    ///
    /// The texture is mapped directly to screen space (UV = NDC).
    /// Not affected by camera rotation or zoom — acts as a fixed backdrop.
    Planar,
}

impl Default for BackgroundMode {
    fn default() -> Self {
        // Default: dark grey, matching common 3D editor conventions
        Self::Color(Vec4::new(0.0, 0.0, 0.0, 1.0))
    }
}

impl BackgroundMode {
    /// Creates a solid color background.
    #[inline]
    #[must_use]
    pub fn color(r: f32, g: f32, b: f32) -> Self {
        Self::Color(Vec4::new(r, g, b, 1.0))
    }

    /// Creates a solid color background with alpha.
    #[inline]
    #[must_use]
    pub fn color_with_alpha(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color(Vec4::new(r, g, b, a))
    }

    /// Creates a vertical gradient background.
    #[inline]
    #[must_use]
    pub fn gradient(top: Vec4, bottom: Vec4) -> Self {
        Self::Gradient { top, bottom }
    }

    /// Creates a cubemap skybox background.
    #[inline]
    #[must_use]
    pub fn cubemap(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Cube,
        }
    }

    /// Creates an equirectangular panorama background.
    #[inline]
    #[must_use]
    pub fn equirectangular(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Equirectangular,
        }
    }

    /// Creates a planar (screen-space) background.
    #[inline]
    #[must_use]
    pub fn planar(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Planar,
        }
    }

    /// Creates a procedural sky background with default golden-hour parameters.
    #[inline]
    #[must_use]
    pub fn procedural() -> Self {
        Self::Procedural(ProceduralSkyParams::default())
    }

    /// Creates a procedural sky background with custom atmosphere parameters.
    #[inline]
    #[must_use]
    pub fn procedural_with(params: ProceduralSkyParams) -> Self {
        Self::Procedural(params)
    }

    /// Returns the clear color for the RenderPass.
    ///
    /// - `Color` mode: returns the specified color (hardware clear).
    /// - Other modes: returns black (skybox pass will overdraw).
    #[must_use]
    pub fn clear_color(&self) -> wgpu::Color {
        match self {
            Self::Color(c) => wgpu::Color {
                r: f64::from(c.x),
                g: f64::from(c.y),
                b: f64::from(c.z),
                a: f64::from(c.w),
            },
            // For gradient/texture/procedural modes, clear to black.
            // The SkyboxPass will fill uncovered pixels.
            Self::Gradient { .. } | Self::Texture { .. } | Self::Procedural(_) => {
                wgpu::Color::BLACK
            }
        }
    }

    /// Returns `true` if this mode requires a skybox draw call.
    ///
    /// `Color` mode uses hardware clear and needs no draw call.
    #[inline]
    #[must_use]
    pub fn needs_skybox_pass(&self) -> bool {
        !matches!(self, Self::Color(_))
    }
}

// ============================================================================
// BackgroundSettings (wraps mode + CpuBuffer — follows ToneMappingSettings pattern)
// ============================================================================

/// Background rendering configuration (mode + automatic uniform version control).
///
/// Wraps [`BackgroundMode`] together with a `CpuBuffer<SkyboxParamsUniforms>`
/// whose version is automatically bumped only when setter methods write new
/// values. The render pass only calls `ensure_buffer_id()` — no per-frame
/// buffer writes occur in the render pipeline.
///
/// # Usage
///
/// ```rust,ignore
/// // Set mode (automatically syncs uniform values)
/// scene.background.set_mode(BackgroundMode::equirectangular(tex, 1.0));
///
/// // Fine-tune individual parameters
/// scene.background.set_rotation(0.5);
/// scene.background.set_intensity(2.0);
/// ```
#[derive(Debug, Clone)]
pub struct BackgroundSettings {
    /// The current background rendering mode.
    #[doc(hidden)]
    pub mode: BackgroundMode,

    /// Skybox parameters uniform buffer (version-tracked).
    /// Updated via setter methods; render pass only reads.
    #[doc(hidden)]
    pub uniforms: CpuBuffer<SkyboxParamsUniforms>,
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        let mode = BackgroundMode::default();
        let uniforms = CpuBuffer::new(
            SkyboxParamsUniforms::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Skybox Params Uniforms"),
        );
        let mut settings = Self { mode, uniforms };
        settings.sync_uniforms_from_mode();
        settings
    }
}

impl BackgroundSettings {
    /// Returns a reference to the current background mode.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> &BackgroundMode {
        &self.mode
    }

    /// Sets the background mode and syncs uniform values accordingly.
    ///
    /// The `CpuBuffer` version is bumped only if the derived uniform values
    /// actually differ from the current state.
    pub fn set_mode(&mut self, mode: BackgroundMode) {
        self.mode = mode;
        self.sync_uniforms_from_mode();
    }

    /// Sets the Y-axis rotation (radians) for texture-based backgrounds.
    ///
    /// Also updates the `rotation` field inside `BackgroundMode::Texture`
    /// to keep the enum and buffer in sync.
    pub fn set_rotation(&mut self, rotation: f32) {
        if let BackgroundMode::Texture { rotation: r, .. } = &mut self.mode {
            *r = rotation;
        }
        self.uniforms.write().rotation = rotation;
    }

    /// Sets the brightness/exposure multiplier for texture-based backgrounds.
    ///
    /// Also updates the `intensity` field inside `BackgroundMode::Texture`
    /// to keep the enum and buffer in sync.
    pub fn set_intensity(&mut self, intensity: f32) {
        if let BackgroundMode::Texture { intensity: i, .. } = &mut self.mode {
            *i = intensity;
        }
        self.uniforms.write().intensity = intensity;
    }

    /// Sets gradient colors (top and bottom).
    ///
    /// Switches the mode to `Gradient` if it isn't already.
    pub fn set_gradient_colors(&mut self, top: Vec4, bottom: Vec4) {
        self.mode = BackgroundMode::Gradient { top, bottom };
        let mut p = self.uniforms.write();
        p.color_top = top;
        p.color_bottom = bottom;
        p.rotation = 0.0;
        p.intensity = 1.0;
    }

    // === Delegate methods from BackgroundMode ===

    /// Returns the clear color for the RenderPass.
    #[inline]
    #[must_use]
    pub fn clear_color(&self) -> wgpu::Color {
        self.mode.clear_color()
    }

    /// Returns `true` if the current mode requires a skybox draw call.
    #[inline]
    #[must_use]
    pub fn needs_skybox_pass(&self) -> bool {
        self.mode.needs_skybox_pass()
    }

    /// Returns a mutable reference to the procedural sky parameters,
    /// or `None` if the current mode is not `Procedural`.
    #[inline]
    #[must_use]
    pub fn procedural_sky_params_mut(&mut self) -> Option<&mut ProceduralSkyParams> {
        if let BackgroundMode::Procedural(params) = &mut self.mode {
            Some(params)
        } else {
            None
        }
    }

    /// Returns a reference to the procedural sky parameters,
    /// or `None` if the current mode is not `Procedural`.
    #[inline]
    #[must_use]
    pub fn procedural_sky_params(&self) -> Option<&ProceduralSkyParams> {
        if let BackgroundMode::Procedural(params) = &self.mode {
            Some(params)
        } else {
            None
        }
    }

    // === Internal ===

    /// Derives uniform values from the current `BackgroundMode` and writes
    /// them into the `CpuBuffer`.
    fn sync_uniforms_from_mode(&mut self) {
        let mut p = self.uniforms.write();
        match &self.mode {
            BackgroundMode::Color(c) => {
                p.color_top = *c;
                p.color_bottom = *c;
                p.rotation = 0.0;
                p.intensity = 1.0;
            }
            BackgroundMode::Gradient { top, bottom } => {
                p.color_top = *top;
                p.color_bottom = *bottom;
                p.rotation = 0.0;
                p.intensity = 1.0;
            }
            BackgroundMode::Texture {
                rotation,
                intensity,
                ..
            } => {
                p.color_top = Vec4::ZERO;
                p.color_bottom = Vec4::ZERO;
                p.rotation = *rotation;
                p.intensity = *intensity;
            }
            BackgroundMode::Procedural(_) => {
                p.color_top = Vec4::ZERO;
                p.color_bottom = Vec4::ZERO;
                p.rotation = 0.0;
                p.intensity = 1.0;
            }
        }
    }
}
