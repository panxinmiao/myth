use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use glam::{Vec4};
use bitflags::bitflags;

use crate::resources::buffer::BufferRef;
use crate::resources::uniforms::{MeshBasicUniforms, MeshStandardUniforms, MeshPhongUniforms};
use crate::assets::TextureHandle;

// Shader 编译选项
bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct MaterialFeatures: u32 {
        const USE_MAP           = 1 << 0;
        const USE_NORMAL_MAP    = 1 << 1;
        const USE_ROUGHNESS_MAP = 1 << 2;
        const USE_METALNESS_MAP = 1 << 3;
        const USE_EMISSIVE_MAP  = 1 << 4;
        const USE_AO_MAP        = 1 << 5;
        const USE_SPECULAR_MAP  = 1 << 6;
    }
}

// ============================================================================
// 具体材质定义 (Specific Materials)
// ============================================================================

// MeshBasicMaterial
// ----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub uniforms: MeshBasicUniforms,
    // 内部使用的 Uniform Buffer，自动管理
    pub(crate) uniform_buffer: BufferRef,
    
    // 直接持有 Texture 引用，不再是 Uuid
    pub map: Option<TextureHandle>, 
}

impl MeshBasicMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniforms = MeshBasicUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(
            &[uniforms], 
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            Some("MeshBasicUniforms")
        );
        
        Self {
            uniforms,
            uniform_buffer,
            map: None,
        }
    }
}

// 提供默认实现，方便用户先创建后修改
impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}


// MeshPhongMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshPhongMaterial {
    pub uniforms: MeshPhongUniforms,
    pub(crate) uniform_buffer: BufferRef,
    
    pub map: Option<TextureHandle>,
    pub normal_map: Option<TextureHandle>,
    pub specular_map: Option<TextureHandle>,
    pub emissive_map: Option<TextureHandle>,
}

impl MeshPhongMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniforms = MeshPhongUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(
            &[uniforms], 
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            Some("MeshPhongUniforms")
        );

        Self {
            uniforms,
            uniform_buffer,
            map: None,
            normal_map: None,
            specular_map: None,
            emissive_map: None,
        }
    }
}

impl Default for MeshPhongMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}


// MeshStandardMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshStandardMaterial {
    pub uniforms: MeshStandardUniforms,
    pub(crate) uniform_buffer: BufferRef,
    
    pub map: Option<TextureHandle>,
    pub normal_map: Option<TextureHandle>,
    pub roughness_map: Option<TextureHandle>,
    pub metalness_map: Option<TextureHandle>,
    pub emissive_map: Option<TextureHandle>,
    pub ao_map: Option<TextureHandle>,
}

impl MeshStandardMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniforms = MeshStandardUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(
            &[uniforms], 
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            Some("MeshStandardUniforms")
        );
        
        Self {
            uniforms,
            uniform_buffer,
            map: None,
            normal_map: None,
            roughness_map: None,
            metalness_map: None,
            emissive_map: None,
            ao_map: None,
        }
    }
}

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}

// ============================================================================
// 核心材质枚举 (Material Data Enum)
// ============================================================================

#[derive(Debug)]
pub enum MaterialData {
    Basic(MeshBasicMaterial),
    Phong(MeshPhongMaterial),
    Standard(MeshStandardMaterial),
    // 可以在这里扩展更多类型，例如 PBR, CustomShader 等
}

impl MaterialData {
    pub fn shader_name(&self) -> &'static str {
        match self {
            Self::Basic(_) => "mesh_basic",
            Self::Phong(_) => "mesh_phong",
            Self::Standard(_) => "mesh_standard",
        }
    }

    pub fn flush_uniforms(&self) {
        match self {
            Self::Basic(m) => m.uniform_buffer.update(&[m.uniforms]),
            Self::Phong(m) => m.uniform_buffer.update(&[m.uniforms]),
            Self::Standard(m) => m.uniform_buffer.update(&[m.uniforms]),
        }
    }

    pub fn get_features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        match self {
            Self::Basic(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
            }
            Self::Phong(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.specular_map.is_some() { features |= MaterialFeatures::USE_SPECULAR_MAP; }
                if m.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
            }
            Self::Standard(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
                if m.metalness_map.is_some() { features |= MaterialFeatures::USE_METALNESS_MAP; }
                if m.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
                if m.ao_map.is_some() { features |= MaterialFeatures::USE_AO_MAP; }
            }
        }
        features
    }
}

// ============================================================================
// 材质主结构 (Material Wrapper)
// ============================================================================

#[derive(Debug)]
pub struct Material {
    pub uuid: Uuid,
    pub version: AtomicU64,
    pub name: Option<String>,
    
    // 核心数据变成了 Enum，不仅类型安全，而且没有 Box 的堆分配开销
    pub data: MaterialData, 
    
    // 通用渲染状态 (Render States)
    pub transparent: bool,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32,
}

impl Material {
    /// 基础构造函数
    pub fn new(data: MaterialData) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
            name: None,
            data,
            transparent: false,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    // 辅助构造：Basic
    pub fn new_basic(color: Vec4) -> Self {
        let data = MeshBasicMaterial::new(color);
        Self::from(data)
    }

    pub fn new_phong(color: Vec4) -> Self {
        let data = MeshPhongMaterial::new(color);
        Self::from(data)
    }

    // 辅助构造：Standard
    pub fn new_standard(color: Vec4) -> Self {
        let data = MeshStandardMaterial::new(color);
        Self::from(data)
    }

    pub fn as_basic(&self) -> Option<&MeshBasicMaterial> {
        match &self.data {
            MaterialData::Basic(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_basic_mut(&mut self) -> Option<&mut MeshBasicMaterial> {
        match &mut self.data {
            MaterialData::Basic(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_phong(&self) -> Option<&MeshPhongMaterial> {
        match &self.data {
            MaterialData::Phong(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_phong_mut(&mut self) -> Option<&mut MeshPhongMaterial> {
        match &mut self.data {
            MaterialData::Phong(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_standard(&self) -> Option<&MeshStandardMaterial> {
        match &self.data {
            MaterialData::Standard(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_standard_mut(&mut self) -> Option<&mut MeshStandardMaterial> {
        match &mut self.data {
            MaterialData::Standard(m) => Some(m),
            _ => None,
        }
    }

    // 代理方法：直接转发给内部数据
    pub fn shader_name(&self) -> &'static str { self.data.shader_name() }
    pub fn flush_uniforms(&self) { self.data.flush_uniforms() }
    pub fn get_features(&self) -> MaterialFeatures { self.data.get_features() }

    pub fn mark_dirty(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
}

// ============================================================================
// 语法糖：允许从 具体材质 直接转为 通用材质
// ============================================================================

impl From<MeshBasicMaterial> for Material {
    fn from(data: MeshBasicMaterial) -> Self {
        Material::new(MaterialData::Basic(data))
    }
}

impl std::ops::Deref for MeshBasicMaterial {
    type Target = MeshBasicUniforms;

    fn deref(&self) -> &Self::Target {
        &self.uniforms
     }
}

impl std::ops::DerefMut for MeshBasicMaterial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uniforms
     }
}

impl From<MeshPhongMaterial> for Material {
    fn from(data: MeshPhongMaterial) -> Self {
        Material::new(MaterialData::Phong(data))
    }
}

impl std::ops::Deref for MeshPhongMaterial {
    type Target = MeshPhongUniforms;
    
    fn deref(&self) -> &Self::Target {
        &self.uniforms
     }
}

impl std::ops::DerefMut for MeshPhongMaterial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uniforms
     }
}

impl From<MeshStandardMaterial> for Material {
    fn from(data: MeshStandardMaterial) -> Self {
        Material::new(MaterialData::Standard(data))
    }
}

impl std::ops::Deref for MeshStandardMaterial {
    type Target = MeshStandardUniforms;
    
    fn deref(&self) -> &Self::Target {
        &self.uniforms
     }
}

impl std::ops::DerefMut for MeshStandardMaterial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uniforms
     }
}