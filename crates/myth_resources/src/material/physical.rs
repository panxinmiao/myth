use bitflags::bitflags;
use glam::{Vec2, Vec3, Vec4};
use myth_macros::myth_material;
use parking_lot::RwLock;

use crate::ShaderDefines;
use crate::TextureHandle;
use crate::material::{AlphaMode, Side};
use crate::screen_space::FeatureId;
use crate::uniforms::Mat3Uniform;

bitflags! {
    /// Feature flags controlling which PBR extensions are active.
    ///
    /// Each enabled feature adds shader defines and may require
    /// additional uniform parameters.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PhysicalFeatures: u32 {
        const IBL = 1 << 0;
        const SPECULAR = 1 << 1;
        const IOR = 1 << 2;
        const CLEARCOAT = 1 << 3;
        const SHEEN = 1 << 4;
        const IRIDESCENCE = 1 << 5;
        const ANISOTROPY = 1 << 6;
        const TRANSMISSION = 1 << 7;
        const DISPERSION = 1 << 8;

        const SSS = 1 << 9;
        const SSR = 1 << 10;

        const STANDARD_PBR = Self::IBL.bits() | Self::SPECULAR.bits() | Self::IOR.bits();
    }
}

impl Default for PhysicalFeatures {
    fn default() -> Self {
        Self::STANDARD_PBR
    }
}

#[myth_material(shader = "entry/main/physical", crate_path = "crate")]
pub struct PhysicalMaterial {
    /// Base color.
    #[uniform(default = "Vec4::ONE")]
    pub color: Vec4,

    /// Emissive color.
    #[uniform]
    pub emissive: Vec3,

    /// Emissive intensity.
    #[uniform(default = "1.0")]
    pub emissive_intensity: f32,

    /// Roughness factor.
    #[uniform(default = "1.0")]
    pub roughness: f32,

    /// Metalness factor.
    #[uniform]
    pub metalness: f32,

    /// Opacity value.
    #[uniform(default = "1.0")]
    pub opacity: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,

    /// Normal map scale.
    #[uniform(default = "Vec2::ONE")]
    pub normal_scale: Vec2,

    /// AO map intensity.
    #[uniform(default = "1.0")]
    pub ao_map_intensity: f32,

    /// Index of Refraction.
    #[uniform(default = "1.5")]
    pub ior: f32,

    /// Specular color.
    #[uniform(default = "Vec3::ONE")]
    pub specular_color: Vec3,

    /// Specular intensity.
    #[uniform(default = "1.0")]
    pub specular_intensity: f32,

    /// Clearcoat factor.
    #[uniform]
    pub clearcoat: f32,

    /// Clearcoat roughness factor.
    #[uniform]
    pub clearcoat_roughness: f32,

    /// Clearcoat normal map scale.
    #[uniform(hidden, default = "Vec2::ONE")]
    pub clearcoat_normal_scale: Vec2,

    /// The sheen tint. Default is (0, 0, 0), black.
    #[uniform]
    pub sheen_color: Vec3,

    /// The sheen roughness. Default is 1.0.
    #[uniform(default = "1.0")]
    pub sheen_roughness: f32,

    /// The intensity of the iridescence layer, simulating RGB color shift based on the angle between the surface and the viewer.
    #[uniform]
    pub iridescence: f32,

    /// The strength of the iridescence RGB color shift effect, represented by an index-of-refraction. Default is 1.3.
    #[uniform(default = "1.3")]
    pub iridescence_ior: f32,

    /// The minimum thickness of the thin-film layer given in nanometers. Default is 100 nm.
    #[uniform(default = "100.0")]
    pub iridescence_thickness_min: f32,

    /// The maximum thickness of the thin-film layer given in nanometers. Default is 400 nm.
    #[uniform(default = "400.0")]
    pub iridescence_thickness_max: f32,

    /// Anisotropy direction vector (computed from angle and intensity).
    #[uniform(hidden)]
    pub anisotropy_vector: Vec2,

    /// The transmission factor controlling the amount of light that passes through the surface.
    #[uniform]
    pub transmission: f32,

    /// The thickness of the object used for subsurface absorption.
    #[uniform]
    pub thickness: f32,

    /// The color that light is attenuated towards as it passes through the material.
    #[uniform(default = "Vec3::ONE")]
    pub attenuation_color: Vec3,

    /// The distance that light travels through the material before it is absorbed.
    #[uniform(default = "-1.0f32")]
    pub attenuation_distance: f32,

    /// The amount of chromatic dispersion in the transmitted light.
    #[uniform]
    pub dispersion: f32,

    /// Subsurface scattering feature ID.
    #[uniform(hidden)]
    pub sss_id: u32,

    /// Screen-space reflections feature ID.
    #[uniform(hidden)]
    pub ssr_id: u32,

    /// The color map.
    #[texture]
    pub map: TextureSlot,

    /// The normal map.
    #[texture]
    pub normal_map: TextureSlot,

    /// The roughness map.
    #[texture]
    pub roughness_map: TextureSlot,

    /// The metalness map.
    #[texture]
    pub metalness_map: TextureSlot,

    /// The emissive map.
    #[texture]
    pub emissive_map: TextureSlot,

    /// The AO map.
    #[texture]
    pub ao_map: TextureSlot,

    /// The specular map.
    #[texture]
    pub specular_map: TextureSlot,

    /// The specular intensity map.
    #[texture]
    pub specular_intensity_map: TextureSlot,

    /// The clearcoat map.
    #[texture]
    pub clearcoat_map: TextureSlot,

    /// The clearcoat roughness map.
    #[texture]
    pub clearcoat_roughness_map: TextureSlot,

    /// The clearcoat normal map.
    #[texture]
    pub clearcoat_normal_map: TextureSlot,

    /// The sheen color map.
    #[texture]
    pub sheen_color_map: TextureSlot,

    /// The sheen roughness map.
    #[texture]
    pub sheen_roughness_map: TextureSlot,

    /// The iridescence map.
    #[texture]
    pub iridescence_map: TextureSlot,

    /// The iridescence thickness map.
    #[texture]
    pub iridescence_thickness_map: TextureSlot,

    /// The anisotropy map.
    #[texture]
    pub anisotropy_map: TextureSlot,

    /// The transmission map.
    #[texture]
    pub transmission_map: TextureSlot,

    /// The thickness map.
    #[texture]
    pub thickness_map: TextureSlot,

    /// Material feature flags.
    #[internal(
        default = "parking_lot::RwLock::new(PhysicalFeatures::default())",
        clone_with = "|s: &Self| parking_lot::RwLock::new(*s.features.read())"
    )]
    pub(crate) features: RwLock<PhysicalFeatures>,
}

impl PhysicalMaterial {
    /// Creates a new PBR material with the given base color.
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        Self::from_uniforms(PhysicalUniforms {
            color,
            ..Default::default()
        })
    }

    // -- Core builder methods --

    /// Sets the base color (builder).
    #[must_use]
    pub fn with_color(self, color: Vec4) -> Self {
        self.uniforms.write().color = color;
        self
    }

    /// Sets the roughness factor (builder).
    #[must_use]
    pub fn with_roughness(self, roughness: f32) -> Self {
        self.uniforms.write().roughness = roughness;
        self
    }

    /// Sets the metalness factor (builder).
    #[must_use]
    pub fn with_metalness(self, metalness: f32) -> Self {
        self.uniforms.write().metalness = metalness;
        self
    }

    /// Sets the emissive color and intensity (builder).
    #[must_use]
    pub fn with_emissive(self, color: Vec3, intensity: f32) -> Self {
        {
            let mut u = self.uniforms.write();
            u.emissive = color;
            u.emissive_intensity = intensity;
        }
        self
    }

    /// Sets the opacity (builder).
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        self.uniforms.write().opacity = opacity;
        self
    }

    /// Sets the normal map scale (builder).
    #[must_use]
    pub fn with_normal_scale(self, scale: Vec2) -> Self {
        self.uniforms.write().normal_scale = scale;
        self
    }

    /// Sets the color map texture (builder).
    #[must_use]
    pub fn with_map(self, handle: TextureHandle) -> Self {
        self.set_map(Some(handle));
        self
    }

    /// Sets the normal map texture (builder).
    #[must_use]
    pub fn with_normal_map(self, handle: TextureHandle) -> Self {
        self.set_normal_map(Some(handle));
        self
    }

    /// Sets the roughness map texture (builder).
    #[must_use]
    pub fn with_roughness_map(self, handle: TextureHandle) -> Self {
        self.set_roughness_map(Some(handle));
        self
    }

    /// Sets the metalness map texture (builder).
    #[must_use]
    pub fn with_metalness_map(self, handle: TextureHandle) -> Self {
        self.set_metalness_map(Some(handle));
        self
    }

    /// Sets the emissive map texture (builder).
    #[must_use]
    pub fn with_emissive_map(self, handle: TextureHandle) -> Self {
        self.set_emissive_map(Some(handle));
        self
    }

    /// Sets the AO map texture (builder).
    #[must_use]
    pub fn with_ao_map(self, handle: TextureHandle) -> Self {
        self.set_ao_map(Some(handle));
        self
    }

    /// Sets the face culling side (builder).
    #[must_use]
    pub fn with_side(self, side: Side) -> Self {
        self.set_side(side);
        self
    }

    /// Sets the alpha mode (builder).
    #[must_use]
    pub fn with_alpha_mode(self, mode: AlphaMode) -> Self {
        self.set_alpha_mode(mode);
        self
    }

    /// Sets depth write (builder).
    #[must_use]
    pub fn with_depth_write(self, enabled: bool) -> Self {
        self.set_depth_write(enabled);
        self
    }

    // -- Feature-based shader defines --

    pub(crate) fn extra_defines(&self, defines: &mut ShaderDefines) {
        let features = *self.features.read();

        if features.contains(PhysicalFeatures::IBL) {
            defines.set("USE_IBL", "1");
        }
        if features.contains(PhysicalFeatures::CLEARCOAT) {
            defines.set("USE_CLEARCOAT", "1");
        }
        if features.contains(PhysicalFeatures::IOR) {
            defines.set("USE_IOR", "1");
        }
        if features.contains(PhysicalFeatures::SPECULAR) {
            defines.set("USE_SPECULAR", "1");
        }
        if features.contains(PhysicalFeatures::SHEEN) {
            defines.set("USE_SHEEN", "1");
        }
        if features.contains(PhysicalFeatures::IRIDESCENCE) {
            defines.set("USE_IRIDESCENCE", "1");
        }
        if features.contains(PhysicalFeatures::ANISOTROPY) {
            defines.set("USE_ANISOTROPY", "1");
        }
        if features.contains(PhysicalFeatures::TRANSMISSION) {
            defines.set("USE_TRANSMISSION", "1");
        }
        if features.contains(PhysicalFeatures::DISPERSION) {
            defines.set("USE_DISPERSION", "1");
        }
    }

    // -- Feature toggle --

    fn toggle_feature(&self, feature: PhysicalFeatures, enabled: bool) {
        let mut guard = self.features.write();
        let old = *guard;
        if enabled {
            guard.insert(feature);
        } else {
            guard.remove(feature);
        }

        if *guard != old {
            self.version
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Disables the given PBR feature.
    pub fn disable_feature(&self, feature: PhysicalFeatures) {
        self.toggle_feature(feature, false);
    }

    /// Enables the given PBR feature.
    pub fn enable_feature(&self, feature: PhysicalFeatures) {
        self.toggle_feature(feature, true);
    }

    // -- Advanced feature builders --

    /// Enables the clearcoat layer (builder).
    #[must_use]
    pub fn with_clearcoat(self, factor: f32, roughness: f32) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.clearcoat = factor;
            uniforms.clearcoat_roughness = roughness;
        }
        self.toggle_feature(PhysicalFeatures::CLEARCOAT, true);
        self
    }

    /// Enables the sheen layer (builder).
    #[must_use]
    pub fn with_sheen(self, color: Vec3, roughness: f32) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.sheen_color = color;
            uniforms.sheen_roughness = roughness;
        }
        self.toggle_feature(PhysicalFeatures::SHEEN, true);
        self
    }

    /// Enables iridescence (builder).
    #[must_use]
    pub fn with_iridescence(
        self,
        intensity: f32,
        ior: f32,
        thickness_min: f32,
        thickness_max: f32,
    ) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.iridescence = intensity;
            uniforms.iridescence_ior = ior;
            uniforms.iridescence_thickness_min = thickness_min;
            uniforms.iridescence_thickness_max = thickness_max;
        }
        self.toggle_feature(PhysicalFeatures::IRIDESCENCE, true);
        self
    }

    /// Enables anisotropy (builder).
    #[must_use]
    pub fn with_anisotropy(self, anisotropy: f32, rotation: f32) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            let direction = Vec2::new(rotation.cos(), rotation.sin()) * anisotropy;
            uniforms.anisotropy_vector = direction;
        }
        self.toggle_feature(PhysicalFeatures::ANISOTROPY, true);
        self
    }

    /// Enables light transmission (builder).
    #[must_use]
    pub fn with_transmission(
        self,
        transmission: f32,
        thickness: f32,
        attenuation_distance: f32,
        attenuation_color: Vec3,
    ) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.transmission = transmission;
            uniforms.thickness = thickness;
            uniforms.attenuation_distance = attenuation_distance;
            uniforms.attenuation_color = attenuation_color;
        }
        self.toggle_feature(PhysicalFeatures::TRANSMISSION, true);
        self
    }

    /// Enables chromatic dispersion (builder).
    #[must_use]
    pub fn with_dispersion(self, dispersion: f32) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.dispersion = dispersion;
        }
        self.toggle_feature(PhysicalFeatures::DISPERSION, true);
        self
    }

    // -- Screen-space effects --

    /// Sets the subsurface scattering feature ID.
    pub fn set_sss_id(&self, id: Option<FeatureId>) {
        let mut u = self.uniforms.write();
        u.sss_id = id.map_or(0, super::super::screen_space::FeatureId::to_u32);
        self.version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.toggle_feature(PhysicalFeatures::SSS, id.is_some());
    }

    /// Returns the subsurface scattering feature ID.
    pub fn sss_id(&self) -> Option<FeatureId> {
        let u = self.uniforms.read();
        FeatureId::from_u32(u.sss_id)
    }

    /// Sets the screen-space reflections feature ID.
    pub fn set_ssr_id(&self, id: Option<FeatureId>) {
        let mut u = self.uniforms.write();
        u.ssr_id = id.map_or(0, super::super::screen_space::FeatureId::to_u32);
        self.version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.toggle_feature(PhysicalFeatures::SSR, id.is_some());
    }

    /// Returns the screen-space reflections feature ID.
    pub fn ssr_id(&self) -> Option<FeatureId> {
        let u = self.uniforms.read();
        FeatureId::from_u32(u.ssr_id)
    }

    /// Enables subsurface scattering (builder).
    #[must_use]
    pub fn with_sss_id(self, id: FeatureId) -> Self {
        self.set_sss_id(Some(id));
        self
    }

    /// Enables screen-space reflections (builder).
    #[must_use]
    pub fn with_ssr_id(self, id: FeatureId) -> Self {
        self.set_ssr_id(Some(id));
        self
    }
}

impl Default for PhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
