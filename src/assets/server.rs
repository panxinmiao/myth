use slotmap::{KeyData, SlotMap, new_key_type};
use rustc_hash::FxHashMap;
use uuid::Uuid;
use std::path::Path;

use crate::resources::geometry::Geometry;
use crate::resources::material::Material;
use crate::resources::texture::Texture;

// 强类型句柄 (Handle)
new_key_type! {
    pub struct GeometryHandle;
    pub struct MaterialHandle;
    pub struct TextureHandle;
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
pub struct AssetServer {
    // 主存储：使用 SlotMap 存放核心资源
    pub geometries: SlotMap<GeometryHandle, Geometry>,
    pub materials: SlotMap<MaterialHandle, Material>,
    pub textures: SlotMap<TextureHandle, Texture>,

    // UUID 映射：用于通过 UUID (通常来自文件加载) 反查运行时 Handle
    // 这是一个辅助索引，渲染循环中不应使用它
    pub(crate) lookup_geo: FxHashMap<Uuid, GeometryHandle>,
    pub(crate) lookup_mat: FxHashMap<Uuid, MaterialHandle>,
    pub(crate) lookup_tex: FxHashMap<Uuid, TextureHandle>,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    pub fn new() -> Self {
        Self {
            geometries: SlotMap::with_key(),
            materials: SlotMap::with_key(),
            textures: SlotMap::with_key(),
            lookup_geo: FxHashMap::default(),
            lookup_mat: FxHashMap::default(),
            lookup_tex: FxHashMap::default(),
        }
    }

    // === Geometry ===
    pub fn add_geometry(&mut self, geometry: Geometry) -> GeometryHandle {
        let uuid = geometry.uuid; 
        let handle = self.geometries.insert(geometry);
        self.lookup_geo.insert(uuid, handle);
        handle
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&Geometry> {
        self.geometries.get(handle)
    }

    pub fn get_geometry_mut(&mut self, handle: GeometryHandle) -> Option<&mut Geometry> {
        self.geometries.get_mut(handle)
    }

    // === Material ===
    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        let uuid = material.uuid;
        let handle = self.materials.insert(material);
        self.lookup_mat.insert(uuid, handle);
        handle
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&Material> {
        self.materials.get(handle)
    }

    pub fn get_material_mut(&mut self, handle: MaterialHandle) -> Option<&mut Material> {
        self.materials.get_mut(handle)
    }

    // === Texture ===
    pub fn add_texture(&mut self, texture: Texture) -> TextureHandle {
        let uuid = texture.uuid;
        let handle = self.textures.insert(texture);
        self.lookup_tex.insert(uuid, handle);
        handle
    }

    pub fn get_texture(&self, handle: TextureHandle) -> Option<&Texture> {
        self.textures.get(handle)
    }

    pub fn get_texture_mut(&mut self, handle: TextureHandle) -> Option<&mut Texture> {
        self.textures.get_mut(handle)
    }

    pub fn load_texture_from_file(&mut self, path: impl AsRef<Path>, color_space: crate::assets::ColorSpace) -> anyhow::Result<TextureHandle> {
        let texture = crate::assets::load_texture_from_file(path, color_space)?;
        let handle = self.add_texture(texture);
        Ok(handle)
    }

    pub fn load_cube_texture_from_files(&mut self, paths: [impl AsRef<Path>; 6], color_space: crate::assets::ColorSpace) -> anyhow::Result<TextureHandle> {
        let texture = crate::assets::load_cube_texture_from_files(paths, color_space)?;
        let handle = self.add_texture(texture);
        Ok(handle)
    }
}