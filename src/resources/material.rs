use uuid::Uuid;
use std::borrow::Cow;
use std::ops::Deref;
use glam::{Vec4};
use bitflags::bitflags;

use crate::resources::buffer::CpuBuffer;
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
        const USE_IBL           = 1 << 7;
    }
}

// ============================================================================
// 具体材质定义 (Specific Materials)
// ============================================================================

// ============================================================================
// 材质纯数据结构 (POD - Plain Old Data)
// ============================================================================

/// 资源绑定数据 - 对应 BindGroup 变化
#[derive(Default, Clone, Debug)]
pub struct MaterialBindings {
    pub map: Option<TextureHandle>,
    pub normal_map: Option<TextureHandle>,
    pub roughness_map: Option<TextureHandle>,
    pub metalness_map: Option<TextureHandle>,
    pub emissive_map: Option<TextureHandle>,
    pub ao_map: Option<TextureHandle>,
    pub specular_map: Option<TextureHandle>,
}

/// 材质设置 - 对应 Pipeline 变化
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct MaterialSettings {
    pub transparent: bool,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: true, 
            depth_test: true,  
            cull_mode: Some(wgpu::Face::Back), 
            side: 0,
        }
    }
}

// ============================================================================
// 具体材质定义 (使用三级版本控制)
// ============================================================================

// MeshBasicMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub uniforms: CpuBuffer<MeshBasicUniforms>,
    pub bindings: MaterialBindings,
    pub settings: MaterialSettings,
    
    pub binding_version: u64,
    pub layout_version: u64,
}

impl MeshBasicMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshBasicUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data, 
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshBasicUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshBasicUniforms> {
        self.uniforms.write()
    }
    
    pub fn set_color(&mut self, color: Vec4) {
        self.uniforms.write().color = color;
    }
    
    pub fn set_opacity(&mut self, opacity: f32) {
        self.uniforms.write().opacity = opacity;
    }
    
    pub fn set_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.map.is_some() != texture.is_some();
        self.bindings.map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.binding_version = self.binding_version.wrapping_add(1);
        } else {
            self.binding_version = self.binding_version.wrapping_add(1);
        }
    }
}

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}


// MeshPhongMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshPhongMaterial {
    pub uniforms: CpuBuffer<MeshPhongUniforms>,
    pub bindings: MaterialBindings,
    pub settings: MaterialSettings,
    
    pub binding_version: u64,
    pub layout_version: u64,
}

impl MeshPhongMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhongUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhongUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshPhongUniforms> {
        self.uniforms.write()
    }
    
    pub fn set_color(&mut self, color: Vec4) {
        self.uniforms.write().color = color;
    }
    
    pub fn set_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.map.is_some() != texture.is_some();
        self.bindings.map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.binding_version = self.binding_version.wrapping_add(1);
        } else {
            self.binding_version = self.binding_version.wrapping_add(1);
        }
    }
    
    pub fn set_normal_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.normal_map.is_some() != texture.is_some();
        self.bindings.normal_map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.binding_version = self.binding_version.wrapping_add(1);
        } else {
            self.binding_version = self.binding_version.wrapping_add(1);
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
    pub uniforms: CpuBuffer<MeshStandardUniforms>,
    pub bindings: MaterialBindings,
    pub settings: MaterialSettings,
    
    pub binding_version: u64,
    pub layout_version: u64,
}

impl MeshStandardMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshStandardUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshStandardUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshStandardUniforms> {
        self.uniforms.write()
    }
    
    pub fn set_color(&mut self, color: Vec4) {
        self.uniforms.write().color = color;
    }
    
    pub fn set_roughness(&mut self, roughness: f32) {
        self.uniforms.write().roughness = roughness;
    }
    
    pub fn set_metalness(&mut self, metalness: f32) {
        self.uniforms.write().metalness = metalness;
    }
    
    pub fn set_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.map.is_some() != texture.is_some();
        self.bindings.map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.binding_version = self.binding_version.wrapping_add(1);
        } else {
            self.binding_version = self.binding_version.wrapping_add(1);
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
}

impl MaterialData {
    pub fn shader_name(&self) -> &'static str {
        match self {
            Self::Basic(_) => "mesh_basic",
            Self::Phong(_) => "mesh_phong",
            Self::Standard(_) => "mesh_physical",
        }
    }

    pub fn get_features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        match self {
            Self::Basic(m) => {
                if m.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
            }
            Self::Phong(m) => {
                if m.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.bindings.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.bindings.specular_map.is_some() { features |= MaterialFeatures::USE_SPECULAR_MAP; }
                if m.bindings.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
            }
            Self::Standard(m) => {
                features |= MaterialFeatures::USE_IBL;
                if m.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.bindings.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.bindings.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
                if m.bindings.metalness_map.is_some() { features |= MaterialFeatures::USE_METALNESS_MAP; }
                if m.bindings.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
                if m.bindings.ao_map.is_some() { features |= MaterialFeatures::USE_AO_MAP; }
            }
        }
        features
    }
    
    pub fn uniform_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.uniforms.buffer.version,
            Self::Phong(m) => m.uniforms.buffer.version,
            Self::Standard(m) => m.uniforms.buffer.version,
        }
    }
    
    pub fn binding_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.binding_version,
            Self::Phong(m) => m.binding_version,
            Self::Standard(m) => m.binding_version,
        }
    }
    
    pub fn layout_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.layout_version,
            Self::Phong(m) => m.layout_version,
            Self::Standard(m) => m.layout_version,
        }
    }
    
    pub fn settings(&self) -> &MaterialSettings {
        match self {
            Self::Basic(m) => &m.settings,
            Self::Phong(m) => &m.settings,
            Self::Standard(m) => &m.settings,
        }
    }
}

// ============================================================================
// 材质主结构 (Material Wrapper)
// ============================================================================

#[derive(Debug)]
pub struct Material {
    pub uuid: Uuid,
    pub name: Option<Cow<'static, str>>,
    pub data: MaterialData, 
}

impl Material {
    pub fn new(data: MaterialData) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: None,
            data,
        }
    }

    // 辅助构造方法
    pub fn new_basic(color: Vec4) -> Self {
        Self::from(MeshBasicMaterial::new(color))
    }

    pub fn new_phong(color: Vec4) -> Self {
        Self::from(MeshPhongMaterial::new(color))
    }

    pub fn new_standard(color: Vec4) -> Self {
        Self::from(MeshStandardMaterial::new(color))
    }

    // 类型转换辅助方法
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

    // 代理方法
    pub fn shader_name(&self) -> &'static str { 
        self.data.shader_name() 
    }
    
    pub fn get_features(&self) -> MaterialFeatures { 
        self.data.get_features() 
    }
    
    pub fn get_settings(&self) -> &MaterialSettings {
        self.data.settings()
    }
    
    // 便捷访问器（兼容旧代码）
    pub fn transparent(&self) -> bool {
        self.get_settings().transparent
    }
    
    pub fn depth_write(&self) -> bool {
        self.get_settings().depth_write
    }
    
    pub fn depth_test(&self) -> bool {
        self.get_settings().depth_test
    }
    
    pub fn cull_mode(&self) -> Option<wgpu::Face> {
        self.get_settings().cull_mode
    }
    
    // 版本号（组合三级版本）
    // pub fn version(&self) -> u64 {
    //     use std::hash::{Hash, Hasher};
    //     use std::collections::hash_map::DefaultHasher;
        
    //     let mut hasher = DefaultHasher::new();
    //     self.data.layout_version().hash(&mut hasher);
    //     self.data.binding_version().hash(&mut hasher);
    //     hasher.finish()
    // }
}

// ============================================================================
// 语法糖：允许从具体材质直接转为通用材质
// ============================================================================

impl From<MeshBasicMaterial> for Material {
    fn from(data: MeshBasicMaterial) -> Self {
        Material::new(MaterialData::Basic(data))
    }
}

impl From<MeshPhongMaterial> for Material {
    fn from(data: MeshPhongMaterial) -> Self {
        Material::new(MaterialData::Phong(data))
    }
}

impl From<MeshStandardMaterial> for Material {
    fn from(data: MeshStandardMaterial) -> Self {
        Material::new(MaterialData::Standard(data))
    }
}

impl Deref for Material {
    type Target = MaterialData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}