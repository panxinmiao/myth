use slotmap::{KeyData, new_key_type};
use std::path::Path;
use std::sync::Arc;

use crate::assets::storage::AssetStorage;
use crate::resources::geometry::Geometry;
use crate::resources::material::Material;
use crate::resources::texture::{Sampler, Texture};

// 强类型句柄 (Handle)
new_key_type! {
    pub struct GeometryHandle;
    pub struct MaterialHandle;
    pub struct TextureHandle;
    pub struct SamplerHandle;
}


const DUMMY_ENV_MAP_ID: u64 = 0xFFFFFFFF_FFFFFFFF; 
// const DUMMY_SAMPLER_ID: u64 = 0xFFFFFFFF_FFFFFFFE;

impl TextureHandle {
    /// 创建一个系统内部使用的保留 Handle
    /// index: 必须是一个极大的数，避免与 SlotMap 正常分配的 ID 冲突
    #[inline]
    pub fn system_reserved(index: u64) -> Self {
        let data = KeyData::from_ffi(index); 
        Self::from(data)
    }

    #[inline]
    pub fn dummy_env_map() -> Self {
        // 构造一个指向特定 ID 的 Handle
        // 注意：这里假设你的 Handle 是基于 slotmap 的 new_key_type!
        let data = KeyData::from_ffi(DUMMY_ENV_MAP_ID);
        Self::from(data)
    }
}

// 2. 资产服务器 (AssetServer)

#[derive(Clone)] // 现在的 AssetServer 是轻量级的，可以随意 Clone
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

    // === Geometry ===
    // pub fn add_geometry(&mut self, geometry: Geometry) -> GeometryHandle {
    //     let uuid = geometry.uuid; 
    //     let handle = self.geometries.insert(geometry);
    //     self.lookup_geo.insert(uuid, handle);
    //     handle
    // }

    // pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&Geometry> {
    //     self.geometries.get(handle)
    // }

    // pub fn get_geometry_mut(&mut self, handle: GeometryHandle) -> Option<&mut Geometry> {
    //     self.geometries.get_mut(handle)
    // }

    // === Material ===
    // pub fn add_material(&mut self, material: impl Into<Material>) -> MaterialHandle {
    //     let material = material.into();
    //     let uuid = material.uuid;
    //     let handle = self.materials.insert(material);
    //     self.lookup_mat.insert(uuid, handle);
    //     handle
    // }

    // pub fn get_material(&self, handle: MaterialHandle) -> Option<&Material> {
    //     self.materials.get(handle)
    // }

    // pub fn get_material_mut(&mut self, handle: MaterialHandle) -> Option<&mut Material> {
    //     self.materials.get_mut(handle)
    // }

    // === Texture ===
    // pub fn add_texture(&mut self, texture: Texture) -> TextureHandle {
    //     let uuid = texture.uuid;
    //     let handle = self.textures.insert(texture);
    //     self.lookup_tex.insert(uuid, handle);
    //     handle
    // }

    // pub fn get_texture(&self, handle: TextureHandle) -> Option<&Texture> {
    //     self.textures.get(handle)
    // }

    // pub fn get_texture_mut(&mut self, handle: TextureHandle) -> Option<&mut Texture> {
    //     self.textures.get_mut(handle)
    // }

    // pub fn add_sampler(&mut self, sampler: Sampler) -> SamplerHandle {
    //     let uuid = sampler.uuid;
    //     let handle = self.samplers.insert(sampler);
    //     self.lookup_sampler.insert(uuid, handle);
    //     handle
    // }

    // pub fn get_sampler(&self, handle: SamplerHandle) -> Option<&Sampler> {
    //     self.samplers.get(handle)
    // }

    // pub fn get_sampler_mut(&mut self, handle: SamplerHandle) -> Option<&mut Sampler> {
    //     self.samplers.get_mut(handle)
    // }

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

    /// 加载 HDR 格式的环境贴图 (Equirectangular format)
    pub fn load_hdr_texture(&mut self, path: impl AsRef<Path>) -> anyhow::Result<TextureHandle> {
        let texture = crate::assets::load_hdr_texture(path)?;
        let handle = self.textures.add(texture);
        Ok(handle)
    }
}