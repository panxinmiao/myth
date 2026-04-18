#[cfg(feature = "gltf")]
pub mod gltf;
#[cfg(feature = "gltf")]
pub use gltf::GltfLoader;

pub mod ply;
pub use ply::load_gaussian_ply;

#[cfg(feature = "gaussian-npz")]
pub mod npz;
#[cfg(feature = "gaussian-npz")]
pub use npz::load_gaussian_npz;
