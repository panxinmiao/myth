//! Environment - Pure data structure
//!
//! Describes IBL/skybox configuration.
//! Internal GPU textures (processed cube map, PMREM, BRDF LUT) are managed
//! by `ResourceManager` and are **not** stored here.

use myth_resources::texture::TextureSource;

pub const DEFAULT_ENV_BASE_CUBE_SIZE: u32 = 1024;
pub const DEFAULT_ENV_PMREM_SIZE: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvironmentMapConfig {
    pub base_cube_size: u32,
    pub pmrem_size: u32,
}

impl Default for EnvironmentMapConfig {
    fn default() -> Self {
        Self {
            base_cube_size: DEFAULT_ENV_BASE_CUBE_SIZE,
            pmrem_size: DEFAULT_ENV_PMREM_SIZE,
        }
    }
}

/// IBL environment map configuration
#[derive(Clone, Debug)]
pub struct Environment {
    /// User-set original environment map (may be 2D HDR or Cube)
    #[doc(hidden)]
    pub source_env_map: Option<TextureSource>,
    /// Environment light intensity
    pub intensity: f32,
    /// Environment map rotation angle (radians)
    pub rotation: f32,
    /// Environment ambient light
    pub ambient: glam::Vec3,

    /// Persistent GPU environment texture sizing.
    pub map_config: EnvironmentMapConfig,

    /// Version number (used to track changes affecting Pipeline)
    version: u64,

    /// Version number for source-driven GPU recomputation.
    source_version: u64,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Environment {
    fn eq(&self, other: &Self) -> bool {
        self.source_env_map == other.source_env_map
            && self.intensity == other.intensity
            && self.rotation == other.rotation
            && self.ambient == other.ambient
            && self.map_config == other.map_config
    }
}

impl Environment {
    #[must_use]
    pub fn new() -> Self {
        Self {
            source_env_map: None,
            intensity: 1.0,
            rotation: 0.0,
            ambient: glam::Vec3::ZERO,
            map_config: EnvironmentMapConfig::default(),
            version: 0,
            source_version: 0,
        }
    }

    /// Gets the version number
    #[inline]
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Gets the source version used by GPU environment baking.
    #[inline]
    #[must_use]
    pub fn source_version(&self) -> u64 {
        self.source_version
    }

    /// Sets the environment map
    pub fn set_env_map(&mut self, texture_handle: Option<impl Into<TextureSource>>) {
        let was_some = self.source_env_map.is_some();
        let is_some = texture_handle.is_some();

        let texture_handle = texture_handle.map(std::convert::Into::into);
        if self.source_env_map != texture_handle {
            self.source_env_map = texture_handle;
            self.source_version = self.source_version.wrapping_add(1);

            // Presence/absence state change affects shader_defines
            if was_some != is_some {
                self.version = self.version.wrapping_add(1);
            }
        }
    }

    /// Sets the environment light intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }

    /// Sets the environment ambient light
    pub fn set_ambient_light(&mut self, color: glam::Vec3) {
        self.ambient = color;
    }

    pub fn set_base_cube_size(&mut self, size: u32) {
        let size = size.max(1);
        if self.map_config.base_cube_size != size {
            self.map_config.base_cube_size = size;
            self.source_version = self.source_version.wrapping_add(1);
        }
    }

    pub fn set_pmrem_size(&mut self, size: u32) {
        let size = size.max(1);
        if self.map_config.pmrem_size != size {
            self.map_config.pmrem_size = size;
            self.source_version = self.source_version.wrapping_add(1);
        }
    }

    #[inline]
    #[must_use]
    pub fn map_config(&self) -> EnvironmentMapConfig {
        self.map_config
    }

    /// Whether there is a valid environment map
    #[must_use]
    pub fn has_env_map(&self) -> bool {
        self.source_env_map.is_some()
    }

    /// Returns a reference to the source environment map, if set.
    #[inline]
    #[must_use]
    pub fn source_env_map(&self) -> Option<&TextureSource> {
        self.source_env_map.as_ref()
    }
}
