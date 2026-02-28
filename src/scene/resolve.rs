//! Smart resource resolution traits.
//!
//! These traits allow scene helper methods to accept **either** a pre-registered
//! asset handle **or** a raw resource struct. When a struct is passed the scene's
//! built-in [`AssetServer`] automatically registers it and returns a handle.
//!
//! This is the foundation of the "invisible resource management" API:
//!
//! ```rust,ignore
//! // Both of these work:
//! scene.spawn(Geometry::new_box(1.0, 1.0, 1.0), Material::new_unlit(color));
//! scene.spawn(existing_geo_handle, existing_mat_handle);
//! ```

use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::resources::geometry::Geometry;
use crate::resources::material::{Material, PhongMaterial, PhysicalMaterial, UnlitMaterial};

// ---------------------------------------------------------------------------
// Material resolution
// ---------------------------------------------------------------------------

/// Trait for types that can be resolved into a [`MaterialHandle`].
///
/// Implemented for:
/// - `MaterialHandle` — returned as-is (zero cost).
/// - `Material` — auto-registered in `AssetServer`.
/// - `UnlitMaterial`, `PhongMaterial`, `PhysicalMaterial` — converted
///   to `Material` and auto-registered.
pub trait ResolveMaterial {
    fn resolve(self, assets: &AssetServer) -> MaterialHandle;
}

impl ResolveMaterial for MaterialHandle {
    #[inline]
    fn resolve(self, _assets: &AssetServer) -> MaterialHandle {
        self
    }
}

impl ResolveMaterial for Material {
    #[inline]
    fn resolve(self, assets: &AssetServer) -> MaterialHandle {
        assets.materials.add(self)
    }
}

impl ResolveMaterial for UnlitMaterial {
    #[inline]
    fn resolve(self, assets: &AssetServer) -> MaterialHandle {
        assets.materials.add(Material::from(self))
    }
}

impl ResolveMaterial for PhongMaterial {
    #[inline]
    fn resolve(self, assets: &AssetServer) -> MaterialHandle {
        assets.materials.add(Material::from(self))
    }
}

impl ResolveMaterial for PhysicalMaterial {
    #[inline]
    fn resolve(self, assets: &AssetServer) -> MaterialHandle {
        assets.materials.add(Material::from(self))
    }
}

// ---------------------------------------------------------------------------
// Geometry resolution
// ---------------------------------------------------------------------------

/// Trait for types that can be resolved into a [`GeometryHandle`].
///
/// Implemented for:
/// - `GeometryHandle` — returned as-is (zero cost).
/// - `Geometry` — auto-registered in `AssetServer`.
pub trait ResolveGeometry {
    fn resolve(self, assets: &AssetServer) -> GeometryHandle;
}

impl ResolveGeometry for GeometryHandle {
    #[inline]
    fn resolve(self, _assets: &AssetServer) -> GeometryHandle {
        self
    }
}

impl ResolveGeometry for Geometry {
    #[inline]
    fn resolve(self, assets: &AssetServer) -> GeometryHandle {
        assets.geometries.add(self)
    }
}
