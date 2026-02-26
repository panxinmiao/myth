use std::sync::atomic::AtomicU64;

use bitflags::bitflags;
use glam::{Vec2, Vec3, Vec4};
use parking_lot::RwLock;

use crate::assets::{ScreenSpaceProfileHandle, TextureHandle};
use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{AlphaMode, MaterialSettings, Side, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshPhysicalUniforms;
use crate::{ShaderDefines, impl_material_api, impl_material_trait};

bitflags! {
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

        //
        const STANDARD_PBR = Self::IBL.bits() | Self::SPECULAR.bits() | Self::IOR.bits();
    }
}

impl Default for PhysicalFeatures {
    fn default() -> Self {
        Self::STANDARD_PBR
    }
}

#[derive(Clone, Default, Debug)]
pub struct MeshPhysicalTextureSet {
    pub map: TextureSlot,
    pub normal_map: TextureSlot,
    pub roughness_map: TextureSlot,
    pub metalness_map: TextureSlot,
    pub ao_map: TextureSlot,
    pub emissive_map: TextureSlot,
    pub specular_map: TextureSlot,
    pub specular_intensity_map: TextureSlot,
    pub clearcoat_map: TextureSlot,
    pub clearcoat_roughness_map: TextureSlot,
    pub clearcoat_normal_map: TextureSlot,
    pub sheen_color_map: TextureSlot,
    pub sheen_roughness_map: TextureSlot,
    pub iridescence_map: TextureSlot,
    pub iridescence_thickness_map: TextureSlot,
    pub anisotropy_map: TextureSlot,
    pub transmission_map: TextureSlot,
    pub thickness_map: TextureSlot,

    pub map_sampler: Option<SamplerSource>,
    pub normal_map_sampler: Option<SamplerSource>,
    pub roughness_map_sampler: Option<SamplerSource>,
    pub metalness_map_sampler: Option<SamplerSource>,
    pub ao_map_sampler: Option<SamplerSource>,
    pub emissive_map_sampler: Option<SamplerSource>,
    pub specular_map_sampler: Option<SamplerSource>,
    pub specular_intensity_map_sampler: Option<SamplerSource>,
    pub clearcoat_map_sampler: Option<SamplerSource>,
    pub clearcoat_roughness_map_sampler: Option<SamplerSource>,
    pub clearcoat_normal_map_sampler: Option<SamplerSource>,
    pub sheen_color_map_sampler: Option<SamplerSource>,
    pub sheen_roughness_map_sampler: Option<SamplerSource>,
    pub iridescence_map_sampler: Option<SamplerSource>,
    pub iridescence_thickness_map_sampler: Option<SamplerSource>,
    pub anisotropy_map_sampler: Option<SamplerSource>,

    pub transmission_map_sampler: Option<SamplerSource>,
    pub thickness_map_sampler: Option<SamplerSource>,
}

#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhysicalUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) textures: RwLock<MeshPhysicalTextureSet>,

    pub(crate) features: RwLock<PhysicalFeatures>,

    pub(crate) version: AtomicU64,
    pub auto_sync_texture_to_uniforms: bool,

    /// Optional screen-space effect profile.
    ///
    /// When set, the Extract system maps this handle to a per-frame GPU ID (1–254)
    /// and writes it into `screen_space_id` + `screen_space_flags` in the material's
    /// Uniform buffer. The Prepass shader then encodes the ID into `Normal.a`, and the
    /// `SssssPass` uses it for the lookup into the global Storage Buffer.
    ///
    /// Set via [`with_screen_space_profile`](Self::with_screen_space_profile).
    pub screen_space_profile: Option<ScreenSpaceProfileHandle>,
}

impl MeshPhysicalMaterial {
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhysicalUniforms {
            color,
            ..Default::default()
        };

        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhysicalUniforms"),
            ),
            settings: RwLock::new(MaterialSettings::default()),
            textures: RwLock::new(MeshPhysicalTextureSet::default()),
            features: RwLock::new(PhysicalFeatures::default()),

            version: AtomicU64::new(0),
            auto_sync_texture_to_uniforms: false,
            screen_space_profile: None,
        }
    }

    /// Attaches a `ScreenSpaceProfile` to this material (builder).
    ///
    /// Subsequent frames will pick up the profile during the Extract phase and encode
    /// the per-frame GPU ID into the material's Uniform buffer.  Setting this to `None`
    /// effectively disables all screen-space effects for this material.
    ///
    /// # Example
    /// ```rust,ignore
    /// let handle = assets.screen_space_profiles.add(ScreenSpaceProfile::new_sss(…));
    /// let mat = MeshPhysicalMaterial::new(Vec4::ONE)
    ///     .with_screen_space_profile(handle);
    /// ```
    #[must_use]
    pub fn with_screen_space_profile(mut self, handle: ScreenSpaceProfileHandle) -> Self {
        self.screen_space_profile = Some(handle);
        // Bump version so the pipeline cache notices the change on next frame.
        self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self
    }

    /// Sets or clears the screen-space profile at runtime.
    pub fn set_screen_space_profile(&mut self, handle: Option<ScreenSpaceProfileHandle>) {
        self.screen_space_profile = handle;
        self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // -- Core builder methods (chainable at construction time) --

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

    // -- Advanced feature builders (existing) --

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

    pub fn disable_feature(&self, feature: PhysicalFeatures) {
        self.toggle_feature(feature, false);
    }

    pub fn enable_feature(&self, feature: PhysicalFeatures) {
        self.toggle_feature(feature, true);
    }

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

    #[must_use]
    pub fn with_dispersion(self, dispersion: f32) -> Self {
        {
            let mut uniforms = self.uniforms_mut();
            uniforms.dispersion = dispersion;
        }
        self.toggle_feature(PhysicalFeatures::DISPERSION, true);
        self
    }
}

impl_material_api!(
    MeshPhysicalMaterial,
    MeshPhysicalUniforms,
    uniforms: [
        (color,               Vec4, "Base color."),
        (alpha_test,          f32,  "Alpha test threshold."),
        (roughness,           f32,  "Roughness factor."),
        (metalness,           f32,  "Metalness factor."),
        (opacity,             f32,  "Opacity value."),
        (emissive,            Vec3, "Emissive color."),
        (emissive_intensity,  f32,  "Emissive intensity."),
        (normal_scale,        Vec2, "Normal map scale."),
        (ao_map_intensity,    f32,  "AO map intensity."),
        (ior,                 f32,  "Index of Refraction."),
        (specular_color,      Vec3, "Specular color."),
        (specular_intensity,  f32,  "Specular intensity."),

        (clearcoat,           f32,  "Clearcoat factor."),
        (clearcoat_roughness, f32, "Clearcoat roughness factor."),

        (sheen_color,         Vec3,  "The sheen tint. Default is (0, 0, 0), black."),
        (sheen_roughness,     f32,   "The sheen roughness. Default is 1.0."),

        (iridescence,               f32,  "The intensity of the iridescence layer, simulating RGB color shift based on the angle between the surface and the viewer."),
        (iridescence_ior,           f32,  "The strength of the iridescence RGB color shift effect, represented by an index-of-refraction. Default is 1.3."),
        (iridescence_thickness_min, f32,  "The minimum thickness of the thin-film layer given in nanometers. Default is 100 nm."),
        (iridescence_thickness_max, f32,  "The maximum thickness of the thin-film layer given in nanometers. Default is 400 nm."),

        (transmission,            f32,  "The transmission factor controlling the amount of light that passes through the surface."),
        (thickness,               f32,  "The thickness of the object used for subsurface absorption."),
        (attenuation_color,       Vec3, "The color that light is attenuated towards as it passes through the material."),
        (attenuation_distance,    f32,  "The distance that light travels through the material before it is absorbed."),
        (dispersion,              f32,  "The amount of chromatic dispersion in the transmitted light."),

    ],
    textures: [
        (map,                    "The color map."),
        (normal_map,             "The normal map."),
        (roughness_map,          "The roughness map."),
        (metalness_map,          "The metalness map."),
        (ao_map,                 "The AO map."),
        (emissive_map,           "The emissive map."),
        (specular_map,           "The specular map."),
        (specular_intensity_map, "The specular intensity map."),
        (clearcoat_map,          "The clearcoat map."),
        (clearcoat_roughness_map, "The clearcoat roughness map."),
        (clearcoat_normal_map,   "The clearcoat normal map."),
        (sheen_color_map,        "The sheen color map."),
        (sheen_roughness_map,    "The sheen roughness map."),
        (iridescence_map,        "The iridescence map."),
        (iridescence_thickness_map, "The iridescence thickness map."),
        (anisotropy_map,         "The anisotropy map."),
        (transmission_map,       "The transmission map."),
        (thickness_map,          "The thickness map."),
    ],
    manual_clone_fields: {
        features: |s: &Self| parking_lot::RwLock::new(*s.features.read()),
        screen_space_profile: |s: &Self| s.screen_space_profile,
    }
);

impl_material_trait!(
    MeshPhysicalMaterial,
    "templates/mesh_physical",
    MeshPhysicalUniforms,
    textures: [
        map,
        normal_map,
        roughness_map,
        metalness_map,
        ao_map,
        emissive_map,
        specular_map,
        specular_intensity_map,
        clearcoat_map,
        clearcoat_roughness_map,
        clearcoat_normal_map,
        sheen_color_map,
        sheen_roughness_map,
        iridescence_map,
        iridescence_thickness_map,
        anisotropy_map,
        transmission_map,
        thickness_map,
    ]
);

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
