//! Strongly-typed resource handles.
//!
//! These are lightweight `SlotMap` keys for safe, generation-checked
//! references to asset-managed resources.

use slotmap::{KeyData, new_key_type};

new_key_type! {
    /// Handle to a geometry asset in the [`AssetStorage`](crate::geometry::Geometry).
    pub struct GeometryHandle;
    /// Handle to a material asset.
    pub struct MaterialHandle;
    /// Handle to a raw image data asset in the [`AssetStorage`].
    pub struct ImageHandle;
    /// Handle to a texture asset (lightweight glue combining an image with sampling config).
    pub struct TextureHandle;
    /// Opaque handle into the GPU buffer arena.
    ///
    /// Internally a SlotMap key with built-in generation checking.
    pub struct GpuBufferHandle;
}

const DUMMY_ENV_MAP_ID: u64 = 0xFFFF_FFFF_FFFF_FFFF;

impl TextureHandle {
    /// Creates a reserved handle for internal system use.
    #[inline]
    #[must_use]
    pub fn system_reserved(index: u64) -> Self {
        let data = KeyData::from_ffi(index);
        Self::from(data)
    }

    /// Returns a sentinel handle used as a placeholder for missing environment maps.
    #[inline]
    #[must_use]
    pub fn dummy_env_map() -> Self {
        let data = KeyData::from_ffi(DUMMY_ENV_MAP_ID);
        Self::from(data)
    }
}

impl GpuBufferHandle {
    /// Pack the handle into a `u64` suitable for atomic storage.
    #[inline]
    #[must_use]
    pub fn to_bits(self) -> u64 {
        self.0.as_ffi()
    }

    /// Reconstruct a handle from a previously packed `u64`.
    ///
    /// Returns `None` for the sentinel value `0` (used to indicate
    /// "no handle assigned yet").
    #[inline]
    #[must_use]
    pub fn from_bits(bits: u64) -> Option<Self> {
        if bits == 0 {
            None
        } else {
            Some(Self(slotmap::KeyData::from_ffi(bits)))
        }
    }
}
