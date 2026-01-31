use slotmap::{KeyData, new_key_type};
use std::path::Path;
use std::sync::Arc;

use crate::assets::storage::AssetStorage;
use crate::resources::geometry::Geometry;
use crate::resources::material::Material;
use crate::resources::texture::{Sampler, Texture};

// Strongly-typed handles
new_key_type! {
    pub struct GeometryHandle;
    pub struct MaterialHandle;
    pub struct TextureHandle;
    pub struct SamplerHandle;
}


const DUMMY_ENV_MAP_ID: u64 = 0xFFFFFFFF_FFFFFFFF; 
// const DUMMY_SAMPLER_ID: u64 = 0xFFFFFFFF_FFFFFFFE;

impl TextureHandle {
    /// Creates a reserved handle for internal system use
    #[inline]
    pub fn system_reserved(index: u64) -> Self {
        let data = KeyData::from_ffi(index); 
        Self::from(data)
    }

    #[inline]
    pub fn dummy_env_map() -> Self {
        // Construct a Handle pointing to a specific ID
        let data = KeyData::from_ffi(DUMMY_ENV_MAP_ID);
        Self::from(data)
    }
}

// 2. Asset Server

#[derive(Clone)] // AssetServer is now lightweight and can be cloned freely
pub struct AssetServer {
    pub geometries: Arc<AssetStorage<GeometryHandle, Geometry>>,
    pub materials:  Arc<AssetStorage<MaterialHandle, Material>>,
    pub textures:   Arc<AssetStorage<TextureHandle, Texture>>,
    pub samplers:   Arc<AssetStorage<SamplerHandle, Sampler>>,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    pub fn new() -> Self {
        Self {
            geometries: Arc::new(AssetStorage::new()),
            materials:  Arc::new(AssetStorage::new()),
            textures:   Arc::new(AssetStorage::new()),
            samplers:   Arc::new(AssetStorage::new()),
        }
    }



    pub fn load_texture_from_file(&mut self, path: impl AsRef<Path>, color_space: crate::assets::ColorSpace, generate_mipmaps: bool) -> anyhow::Result<TextureHandle> {
        let mut texture = crate::assets::load_texture_from_file(path, color_space)?;
        texture.generate_mipmaps = generate_mipmaps;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    pub fn load_cube_texture_from_files(&mut self, paths: [impl AsRef<Path>; 6], color_space: crate::assets::ColorSpace, generate_mipmaps: bool) -> anyhow::Result<TextureHandle> {
        let mut texture = crate::assets::load_cube_texture_from_files(paths, color_space)?;
        texture.generate_mipmaps = generate_mipmaps;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads an HDR format environment map (Equirectangular format)
    pub fn load_hdr_texture(&mut self, path: impl AsRef<Path>) -> anyhow::Result<TextureHandle> {
        let texture = crate::assets::load_hdr_texture(path)?;
        let handle = self.textures.add(texture);
        Ok(handle)
    }
}