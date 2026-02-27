use glam::Vec3;
use std::num::NonZeroU8;

pub const STENCIL_FEATURE_SSS: u32 = 1 << 0;
pub const STENCIL_FEATURE_SSR: u32 = 1 << 1;
pub const STENCIL_WRITE_MASK: u32 = 0x0F;
// ============================================================================
// 1. Basic Types: Globally Stable 8-bit ID
// ============================================================================

/// Strongly-typed feature ID wrapping `NonZeroU8`, so `Option<FeatureId>` is only 1 byte.
/// Valid range: 1–255. 0 represents no feature.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureId(pub NonZeroU8);

impl FeatureId {
    /// Converts the strongly-typed ID to the underlying `u32` required by shaders
    #[inline]
    #[must_use]
    pub fn to_u32(self) -> u32 {
        u32::from(self.0.get())
    }

    /// Attempts to reconstruct the strongly-typed ID from a shader's `u32` value (0 becomes `None`)
    #[inline]
    pub fn from_u32(val: u32) -> Option<Self> {
        std::num::NonZeroU8::new(val as u8).map(FeatureId)
    }
}

// ============================================================================
// 2. SSS (Subsurface Scattering) Dedicated Structures and Registry
// ============================================================================

/// GPU-side SSS data structure (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct SssProfileData {
    pub scatter_color: [f32; 3], // Scatter color
    pub scatter_radius: f32,     // Scatter radius (screen-space percentage or world-space units)
}

impl Default for SssProfileData {
    fn default() -> Self {
        Self {
            scatter_color: [0.0; 3],
            scatter_radius: 0.0,
        }
    }
}

/// User-facing SSS profile asset
#[derive(Clone, Debug)]
pub struct SssProfile {
    pub scatter_color: Vec3,
    pub scatter_radius: f32,
}

impl SssProfile {
    #[must_use]
    pub fn new(scatter_color: Vec3, scatter_radius: f32) -> Self {
        Self {
            scatter_color,
            scatter_radius,
        }
    }

    #[must_use]
    pub fn to_gpu_data(&self) -> SssProfileData {
        SssProfileData {
            scatter_color: self.scatter_color.into(),
            scatter_radius: self.scatter_radius,
        }
    }
}

/// SSS-dedicated global fixed-length allocator
pub struct SssRegistry {
    /// GPU-layout-aligned array; ID 0 is always the default
    pub buffer_data: [SssProfileData; 256],
    /// Free ID list
    free_list: Vec<u8>,
    /// Version number used to trigger GPU buffer uploads (diff-sync)
    pub version: u64,
}

impl Default for SssRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SssRegistry {
    #[must_use]
    pub fn new() -> Self {
        // From 255 down to 1, so pop() allocates lower IDs first
        let free_list = (1..=255).rev().collect();
        Self {
            buffer_data: [SssProfileData::default(); 256],
            free_list,
            version: 1,
        }
    }

    /// Registers a new SSS profile and returns a globally stable ID
    pub fn add(&mut self, profile: &SssProfile) -> Option<FeatureId> {
        if let Some(id) = self.free_list.pop() {
            self.buffer_data[id as usize] = profile.to_gpu_data();
            self.version += 1;
            Some(FeatureId(NonZeroU8::new(id).unwrap()))
        } else {
            log::warn!("SssRegistry is full (max 255 profiles).");
            None
        }
    }

    /// Updates an existing profile (e.g., for dynamic UI adjustments)
    pub fn update(&mut self, id: FeatureId, profile: &SssProfile) {
        self.buffer_data[id.0.get() as usize] = profile.to_gpu_data();
        self.version += 1;
    }

    /// Removes a profile and recycles its ID for future reuse
    pub fn remove(&mut self, id: FeatureId) {
        let index = id.0.get();
        self.buffer_data[index as usize] = SssProfileData::default();
        self.free_list.push(index);
        self.version += 1;
    }
}

// ============================================================================
// 3. Scene-Level Global Toggles
// ============================================================================
#[derive(Default, Clone, Debug)]
pub struct ScreenSpaceSettings {
    pub enable_sss: bool,
    pub enable_ssr: bool, // Reserved for future use
}
