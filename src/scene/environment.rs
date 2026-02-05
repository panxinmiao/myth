//! Environment - Pure data structure
//!
//! Describes IBL/skybox configuration

use crate::resources::texture::TextureSource;

/// IBL environment map configuration
#[derive(Clone, Debug)]
pub struct Environment {
    /// User-set original environment map (may be 2D HDR or Cube)
    pub source_env_map: Option<TextureSource>,
    /// Normalized `CubeMap` source
    /// If `source_env_map` is Cube, this field equals `source_env_map`
    /// If `source_env_map` is 2D, this field points to the converted `CubeMap`
    pub(crate) processed_env_map: Option<TextureSource>,
    /// Pre-filtered environment map (PMREM, used for PBR Specular IBL)
    pub pmrem_map: Option<TextureSource>,
    /// BRDF LUT texture
    pub brdf_lut: Option<TextureSource>,
    /// Maximum mip level of environment map (used for roughness LOD)
    pub env_map_max_mip_level: f32,
    /// Environment light intensity
    pub intensity: f32,
    /// Environment map rotation angle (radians)
    pub rotation: f32,
    /// Environment ambient color
    pub ambient_color: glam::Vec3,

    /// Version number (used to track changes affecting Pipeline)
    version: u64,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Environment {
    fn eq(&self, other: &Self) -> bool {
        self.source_env_map == other.source_env_map
            && self.processed_env_map == other.processed_env_map
            && self.pmrem_map == other.pmrem_map
            && self.brdf_lut == other.brdf_lut
            && self.env_map_max_mip_level == other.env_map_max_mip_level
            && self.intensity == other.intensity
            && self.rotation == other.rotation
            && self.ambient_color == other.ambient_color
    }
}

impl Environment {
    #[must_use]
    pub fn new() -> Self {
        Self {
            source_env_map: None,
            processed_env_map: None,
            pmrem_map: None,
            brdf_lut: None,
            env_map_max_mip_level: 0.0,
            intensity: 1.0,
            rotation: 0.0,
            ambient_color: glam::Vec3::ZERO,
            version: 0,
        }
    }

    /// Gets the version number
    #[inline]
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Sets the environment map
    pub fn set_env_map(&mut self, texture_handle: Option<impl Into<TextureSource>>) {
        let was_some = self.source_env_map.is_some();
        let is_some = texture_handle.is_some();

        let texture_handle = texture_handle.map(std::convert::Into::into);
        if self.source_env_map != texture_handle {
            self.source_env_map = texture_handle;
            self.processed_env_map = None;
            self.pmrem_map = None;
            self.env_map_max_mip_level = 0.0;

            // Presence/absence state change affects shader_defines
            if was_some != is_some {
                self.version = self.version.wrapping_add(1);
            }
        }
    }

    /// Sets the BRDF LUT
    pub fn set_brdf_lut(&mut self, handle: Option<TextureSource>) {
        self.brdf_lut = handle;
    }

    /// Sets the environment light intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }

    /// Sets the environment ambient color
    pub fn set_ambient_color(&mut self, color: glam::Vec3) {
        self.ambient_color = color;
    }

    /// Whether there is a valid environment map
    #[must_use]
    pub fn has_env_map(&self) -> bool {
        self.source_env_map.is_some()
    }

    /// Gets the processed environment map (for Skybox and other places requiring `CubeMap`)
    /// Only returns `processed_env_map`, does not fall back to `source_env_map`
    /// Because `source_env_map` might be a 2D texture, while Skybox requires `CubeMap`
    #[must_use]
    pub fn get_processed_env_map(&self) -> Option<&TextureSource> {
        self.processed_env_map.as_ref()
    }
}
