mod basic;
mod phong;
mod standard;
mod physical;

pub use basic::MeshBasicMaterial;
pub use phong::MeshPhongMaterial;
pub use standard::MeshStandardMaterial;
pub use physical::MeshPhysicalMaterial;

use std::{any::Any, borrow::Cow, ops::Deref};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::texture::TextureSource;
use bitflags::bitflags;
use glam::Vec4;
use uuid::Uuid;


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
// MaterialTrait: 核心抽象接口
// ============================================================================

/// 所有材质类型必须实现的核心 Trait
/// 
/// 这个 Trait 定义了材质系统的统一接口，支持：
/// - 内置材质的静态分发（高性能）
/// - 自定义材质的动态分发（可扩展）
pub trait MaterialTrait: Any + Send + Sync + std::fmt::Debug {
    /// 返回着色器名称，用于 Shader 识别和加载
    fn shader_name(&self) -> &'static str;

    /// 返回材质特性标志位，用于控制 Shader 变体
    fn features(&self) -> MaterialFeatures;

    /// 返回材质设置（透明度、深度测试等）
    fn settings(&self) -> &MaterialSettings;

    /// 返回材质绑定资源
    fn bindings(&self) -> &MaterialBindings;

    /// 定义 GPU 资源绑定
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);

    /// 获取 Uniform Buffer 引用（用于 GPU 资源确保）
    fn uniform_buffer(&self) -> &BufferRef;

    /// 获取 Uniform 数据字节切片（用于 GPU 数据上传）
    fn uniform_bytes(&self) -> &[u8];

    /// Uniform 数据版本号（用于脏检查）
    fn uniform_version(&self) -> u64;

    /// 布局版本号（Pipeline/Layout 相关变更）
    fn layout_version(&self) -> u64;

    /// 绑定版本号（BindGroup 相关变更）
    fn binding_version(&self) -> u64;

    /// 向下转型支持（用于 Custom 材质的类型恢复）
    fn as_any(&self) -> &dyn Any;

    /// 向下转型支持（可变引用）
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ============================================================================
// Guard 结构体（自动版本管理）
// ============================================================================

pub struct BindingsGuard<'a> {
    bindings: &'a mut MaterialBindings,
    binding_version: &'a mut u64,
    layout_version: &'a mut u64,
    initial_layout: MaterialBindings,
}

impl<'a> std::ops::Deref for BindingsGuard<'a> {
    type Target = MaterialBindings;
    fn deref(&self) -> &Self::Target {
        self.bindings
    }
}

impl<'a> std::ops::DerefMut for BindingsGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bindings
    }
}

impl<'a> Drop for BindingsGuard<'a> {
    fn drop(&mut self) {
        let layout_changed = 
            (self.initial_layout.map.is_some() != self.bindings.map.is_some()) ||
            (self.initial_layout.normal_map.is_some() != self.bindings.normal_map.is_some()) ||
            (self.initial_layout.roughness_map.is_some() != self.bindings.roughness_map.is_some()) ||
            (self.initial_layout.metalness_map.is_some() != self.bindings.metalness_map.is_some()) ||
            (self.initial_layout.emissive_map.is_some() != self.bindings.emissive_map.is_some()) ||
            (self.initial_layout.ao_map.is_some() != self.bindings.ao_map.is_some()) ||
            (self.initial_layout.specular_map.is_some() != self.bindings.specular_map.is_some());
        
        if layout_changed {
            *self.layout_version = self.layout_version.wrapping_add(1);
        }
        *self.binding_version = self.binding_version.wrapping_add(1);
    }
}

pub struct SettingsGuard<'a> {
    settings: &'a mut MaterialSettings,
    layout_version: &'a mut u64,
    initial_settings: MaterialSettings,
}

impl<'a> std::ops::Deref for SettingsGuard<'a> {
    type Target = MaterialSettings;
    fn deref(&self) -> &Self::Target {
        self.settings
    }
}

impl<'a> std::ops::DerefMut for SettingsGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.settings
    }
}

impl<'a> Drop for SettingsGuard<'a> {
    fn drop(&mut self) {
        if self.settings != &self.initial_settings {
            *self.layout_version = self.layout_version.wrapping_add(1);
        }
    }
}

// ============================================================================
// 具体材质定义 (Specific Materials)
// ============================================================================

/// 资源绑定数据 - 对应 BindGroup 变化
#[derive(Default, Clone, Debug)]
pub struct MaterialBindings {
    pub map: Option<TextureSource>,
    // Todo: 可独立设置的采样器Asset
    // pub map_sampler: Option<SamplerSource>,
    pub normal_map: Option<TextureSource>,
    pub roughness_map: Option<TextureSource>,
    pub metalness_map: Option<TextureSource>,
    pub emissive_map: Option<TextureSource>,
    pub ao_map: Option<TextureSource>,
    pub specular_map: Option<TextureSource>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Side {
    Front,
    Back,
    Double,
}

/// 材质设置 - 对应 Pipeline 变化
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct MaterialSettings {
    pub transparent: bool,
    pub depth_write: bool,
    pub depth_test: bool,
    pub side: Side,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: true, 
            depth_test: true,  
            side: Side::Double,
        }
    }
}


// ============================================================================
// 核心材质枚举 (Material Data Enum)
// ============================================================================

/// 材质数据枚举
/// 
/// 采用"静态分发 + 动态逃生舱"的混合策略：
/// - 内置材质（Basic/Phong/Standard/Physical）使用静态分发，保证高性能
/// - Custom 变体允许用户扩展自定义材质，通过 dyn Trait 动态分发
#[derive(Debug)]
pub enum MaterialData {
    Basic(MeshBasicMaterial),
    Phong(MeshPhongMaterial),
    Standard(MeshStandardMaterial),
    Physical(MeshPhysicalMaterial),
    Custom(Box<dyn MaterialTrait>),
}

impl MaterialData {
    pub fn shader_name(&self) -> &'static str {
        match self {
            Self::Basic(m) => m.shader_name(),
            Self::Phong(m) => m.shader_name(),
            Self::Standard(m) => m.shader_name(),
            Self::Physical(m) => m.shader_name(),
            Self::Custom(m) => m.shader_name(),
        }
    }

    pub fn get_features(&self) -> MaterialFeatures {
        match self {
            Self::Basic(m) => m.features(),
            Self::Phong(m) => m.features(),
            Self::Standard(m) => m.features(),
            Self::Physical(m) => m.features(),
            Self::Custom(m) => m.features(),
        }
    }

    pub fn uniform_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.uniform_version(),
            Self::Phong(m) => m.uniform_version(),
            Self::Standard(m) => m.uniform_version(),
            Self::Physical(m) => m.uniform_version(),
            Self::Custom(m) => m.uniform_version(),
        }
    }

    pub fn binding_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.binding_version(),
            Self::Phong(m) => m.binding_version(),
            Self::Standard(m) => m.binding_version(),
            Self::Physical(m) => m.binding_version(),
            Self::Custom(m) => m.binding_version(),
        }
    }

    pub fn layout_version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.layout_version(),
            Self::Phong(m) => m.layout_version(),
            Self::Standard(m) => m.layout_version(),
            Self::Physical(m) => m.layout_version(),
            Self::Custom(m) => m.layout_version(),
        }
    }

    pub fn uniform_buffer(&self) -> &BufferRef {
        match self {
            Self::Basic(m) => m.uniform_buffer(),
            Self::Phong(m) => m.uniform_buffer(),
            Self::Standard(m) => m.uniform_buffer(),
            Self::Physical(m) => m.uniform_buffer(),
            Self::Custom(m) => m.uniform_buffer(),
        }
    }

    pub fn uniform_bytes(&self) -> &[u8] {
        match self {
            Self::Basic(m) => m.uniform_bytes(),
            Self::Phong(m) => m.uniform_bytes(),
            Self::Standard(m) => m.uniform_bytes(),
            Self::Physical(m) => m.uniform_bytes(),
            Self::Custom(m) => m.uniform_bytes(),
        }
    }

    pub fn settings(&self) -> &MaterialSettings {
        match self {
            Self::Basic(m) => m.settings(),
            Self::Phong(m) => m.settings(),
            Self::Standard(m) => m.settings(),
            Self::Physical(m) => m.settings(),
            Self::Custom(m) => m.settings(),
        }
    }

    pub fn bindings(&self) -> &MaterialBindings {
        match self {
            Self::Basic(m) => MaterialTrait::bindings(m),
            Self::Phong(m) => MaterialTrait::bindings(m),
            Self::Standard(m) => MaterialTrait::bindings(m),
            Self::Physical(m) => MaterialTrait::bindings(m),
            Self::Custom(m) => m.bindings(),
        }
    }

    pub fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        match self {
            Self::Basic(m) => m.define_bindings(builder),
            Self::Phong(m) => m.define_bindings(builder),
            Self::Standard(m) => m.define_bindings(builder),
            Self::Physical(m) => m.define_bindings(builder),
            Self::Custom(m) => m.define_bindings(builder),
        }
    }

    /// 尝试向下转型为具体类型（用于 Custom 材质）
    pub fn as_custom<T: MaterialTrait + 'static>(&self) -> Option<&T> {
        match self {
            Self::Custom(m) => m.as_any().downcast_ref::<T>(),
            _ => None,
        }
    }

    /// 尝试向下转型为具体类型的可变引用（用于 Custom 材质）
    pub fn as_custom_mut<T: MaterialTrait + 'static>(&mut self) -> Option<&mut T> {
        match self {
            Self::Custom(m) => m.as_any_mut().downcast_mut::<T>(),
            _ => None,
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

    /// 从自定义材质创建 Material
    /// 
    /// 用于用户扩展的自定义材质类型
    pub fn new_custom<T: MaterialTrait + 'static>(custom_material: T) -> Self {
        Self::new(MaterialData::Custom(Box::new(custom_material)))
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

    pub fn new_physical(color: Vec4) -> Self {
        Self::from(MeshPhysicalMaterial::new(color))
    }

    /// 获取自定义材质的引用
    pub fn as_custom<T: MaterialTrait + 'static>(&self) -> Option<&T> {
        self.data.as_custom::<T>()
    }

    /// 获取自定义材质的可变引用
    pub fn as_custom_mut<T: MaterialTrait + 'static>(&mut self) -> Option<&mut T> {
        self.data.as_custom_mut::<T>()
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

    pub fn as_physical(&self) -> Option<&MeshPhysicalMaterial> {
        match &self.data {
            MaterialData::Physical(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_physical_mut(&mut self) -> Option<&mut MeshPhysicalMaterial> {
        match &mut self.data {
            MaterialData::Physical(m) => Some(m),
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
    
    pub fn side(&self) -> &Side {
        &self.get_settings().side
    }
    
    
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

impl From<MeshPhysicalMaterial> for Material {
    fn from(data: MeshPhysicalMaterial) -> Self {
        Material::new(MaterialData::Physical(data))
    }
}

impl Deref for Material {
    type Target = MaterialData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}