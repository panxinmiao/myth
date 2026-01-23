mod basic;
mod phong;
mod standard;
mod physical;
mod macros;

pub use basic::MeshBasicMaterial;
pub use phong::MeshPhongMaterial;
pub use standard::MeshStandardMaterial;
pub use physical::MeshPhysicalMaterial;

use std::{any::Any, borrow::Cow, ops::Deref};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::texture::{SamplerSource, TextureSource};
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


/// [普通用户接口]
/// 所有材质都实现的标记 Trait。
/// 用户平时使用时，只需将其视为 `dyn Material`。
pub trait MaterialTrait: Any + Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}


/// [高级接口] 渲染系统接口
/// 
/// 普通用户不需要导入此 Trait。
/// 只有在自定义新材质类型，或编写渲染管线时才需要使用。
pub trait RenderableMaterialTrait: MaterialTrait {
    // 逻辑属性
    fn shader_name(&self) -> &'static str;
    fn version(&self) -> u64;
    fn features(&self) -> MaterialFeatures;
    fn settings(&self) -> &MaterialSettings;
    
    // 底层资源绑定 (对普通用户隐藏细节的关键)
    // 这里的 MaterialBindings 依然是 pub 的结构体，但其字段可以是 pub(crate)
    fn bindings(&self) -> &MaterialBindings; 
    
    // 访问器
    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource));
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
    fn uniform_buffer(&self) -> &BufferRef;
    fn uniform_bytes(&self) -> &[u8];
}


/// 资源绑定数据 - 对应 BindGroup 变化
#[derive(Default, Clone, Debug)]
pub struct MaterialBindings {
    pub(crate) map: Option<TextureSource>,
    pub(crate) map_sampler: Option<SamplerSource>,
    pub(crate) normal_map: Option<TextureSource>,
    pub(crate) normal_map_sampler: Option<SamplerSource>,
    pub(crate) roughness_map: Option<TextureSource>,
    pub(crate) roughness_map_sampler: Option<SamplerSource>,
    pub(crate) metalness_map: Option<TextureSource>,
    pub(crate) metalness_map_sampler: Option<SamplerSource>,
    pub(crate) emissive_map: Option<TextureSource>,
    pub(crate) emissive_map_sampler: Option<SamplerSource>,
    pub(crate) ao_map: Option<TextureSource>,
    pub(crate) ao_map_sampler: Option<SamplerSource>,
    pub(crate) specular_map: Option<TextureSource>,
    pub(crate) specular_map_sampler: Option<SamplerSource>,
    #[allow(dead_code)]
    pub(crate) specular_intensity_map: Option<TextureSource>,
    #[allow(dead_code)]
    pub(crate) specular_intensity_map_sampler: Option<SamplerSource>,
}

#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum Side {
    Front,
    Back,
    Double,
}

/// 材质设置 - 对应 Pipeline 变化
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct MaterialSettings {
    pub(crate) transparent: bool,
    pub(crate) depth_write: bool,
    pub(crate) depth_test: bool,
    pub(crate) side: Side,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: true, 
            depth_test: true,  
            side: Side::Front,
        }
    }
}

/// Settings 修改守卫
/// 
/// 当 Settings 发生变化时自动递增材质版本号，用于 Pipeline 缓存检测
pub(crate) struct SettingsGuard<'a> {
    settings: &'a mut MaterialSettings,
    version: &'a mut u64,
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
            *self.version = self.version.wrapping_add(1);
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
pub enum MaterialType {
    Basic(MeshBasicMaterial),
    Phong(MeshPhongMaterial),
    Standard(MeshStandardMaterial),
    Physical(MeshPhysicalMaterial),
    Custom(Box<dyn RenderableMaterialTrait>),
}

impl MaterialTrait for MaterialType {
    fn as_any(&self) -> &dyn Any {
        match self {
            Self::Basic(m) => m.as_any(),
            Self::Phong(m) => m.as_any(),
            Self::Standard(m) => m.as_any(),
            Self::Physical(m) => m.as_any(),
            Self::Custom(m) => m.as_any(),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        match self {
            Self::Basic(m) => m.as_any_mut(),
            Self::Phong(m) => m.as_any_mut(),
            Self::Standard(m) => m.as_any_mut(),
            Self::Physical(m) => m.as_any_mut(),
            Self::Custom(m) => m.as_any_mut(),
        }
    }
}

impl RenderableMaterialTrait for MaterialType {
    fn shader_name(&self) -> &'static str {
        match self {
            Self::Basic(m) => m.shader_name(),
            Self::Phong(m) => m.shader_name(),
            Self::Standard(m) => m.shader_name(),
            Self::Physical(m) => m.shader_name(),
            Self::Custom(m) => m.shader_name(),
        }
    }

    fn version(&self) -> u64 {
        match self {
            Self::Basic(m) => m.version(),
            Self::Phong(m) => m.version(),
            Self::Standard(m) => m.version(),
            Self::Physical(m) => m.version(),
            Self::Custom(m) => m.version(),
        }
    }

    fn features(&self) -> MaterialFeatures {
        match self {
            Self::Basic(m) => m.features(),
            Self::Phong(m) => m.features(),
            Self::Standard(m) => m.features(),
            Self::Physical(m) => m.features(),
            Self::Custom(m) => m.features(),
        }
    }

    fn settings(&self) -> &MaterialSettings {
        match self {
            Self::Basic(m) => m.settings(),
            Self::Phong(m) => m.settings(),
            Self::Standard(m) => m.settings(),
            Self::Physical(m) => m.settings(),
            Self::Custom(m) => m.settings(),
        }
    }

    fn bindings(&self) -> &MaterialBindings {
        match self {
            Self::Basic(m) => m.bindings(),
            Self::Phong(m) => m.bindings(),
            Self::Standard(m) => m.bindings(),
            Self::Physical(m) => m.bindings(),
            Self::Custom(m) => m.bindings(),
        }
    }

    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource)) {
        match self {
            Self::Basic(m) => m.visit_textures(visitor),
            Self::Phong(m) => m.visit_textures(visitor),
            Self::Standard(m) => m.visit_textures(visitor),
            Self::Physical(m) => m.visit_textures(visitor),
            Self::Custom(m) => m.visit_textures(visitor),
        }
    }

    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        match self {
            Self::Basic(m) => m.define_bindings(builder),
            Self::Phong(m) => m.define_bindings(builder),
            Self::Standard(m) => m.define_bindings(builder),
            Self::Physical(m) => m.define_bindings(builder),
            Self::Custom(m) => m.define_bindings(builder),
        }
    }

    fn uniform_buffer(&self) -> &BufferRef {
        match self {
            Self::Basic(m) => m.uniform_buffer(),
            Self::Phong(m) => m.uniform_buffer(),
            Self::Standard(m) => m.uniform_buffer(),
            Self::Physical(m) => m.uniform_buffer(),
            Self::Custom(m) => m.uniform_buffer(),
        }
    }

    fn uniform_bytes(&self) -> &[u8] {
        match self {
            Self::Basic(m) => m.uniform_bytes(),
            Self::Phong(m) => m.uniform_bytes(),
            Self::Standard(m) => m.uniform_bytes(),
            Self::Physical(m) => m.uniform_bytes(),
            Self::Custom(m) => m.uniform_bytes(),
        }
    }
}

impl MaterialType {
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
    pub data: MaterialType, 
}

impl Material {
    pub fn new(data: MaterialType) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: None,
            data,
        }
    }

    /// 从自定义材质创建 Material
    pub fn new_custom<T: RenderableMaterialTrait + 'static>(custom_material: T) -> Self {
        Self::new(MaterialType::Custom(Box::new(custom_material)))
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

    /// 暴露渲染行为接口
    #[inline]
    pub fn as_renderable(&self) -> &dyn RenderableMaterialTrait {
        &self.data
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
            MaterialType::Basic(m) => Some(m),
            _ => None,
        }
    }
    
    pub fn as_basic_mut(&mut self) -> Option<&mut MeshBasicMaterial> {
        match &mut self.data {
            MaterialType::Basic(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_phong(&self) -> Option<&MeshPhongMaterial> {
        match &self.data {
            MaterialType::Phong(m) => Some(m),
            _ => None,
        }
    }
    
    pub fn as_phong_mut(&mut self) -> Option<&mut MeshPhongMaterial> {
        match &mut self.data {
            MaterialType::Phong(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_standard(&self) -> Option<&MeshStandardMaterial> {
        match &self.data {
            MaterialType::Standard(m) => Some(m),
            _ => None,
        }
    }
    
    pub fn as_standard_mut(&mut self) -> Option<&mut MeshStandardMaterial> {
        match &mut self.data {
            MaterialType::Standard(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_physical(&self) -> Option<&MeshPhysicalMaterial> {
        match &self.data {
            MaterialType::Physical(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_physical_mut(&mut self) -> Option<&mut MeshPhysicalMaterial> {
        match &mut self.data {
            MaterialType::Physical(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_any(&self) -> &dyn Any {
        self.data.as_any()
    }

    pub fn as_any_mut(&mut self) -> &mut dyn Any {
        self.data.as_any_mut()
    }

    // 代理方法
    #[inline]
    pub fn shader_name(&self) -> &'static str { 
        self.data.shader_name() 
    }
    
    #[inline]
    pub fn features(&self) -> MaterialFeatures { 
        self.data.features() 
    }
    
    #[inline]
    pub fn settings(&self) -> &MaterialSettings {
        self.data.settings()
    }
    
    // 便捷访问器
    #[inline]
    pub fn transparent(&self) -> bool {
        self.settings().transparent
    }
    
    #[inline]
    pub fn depth_write(&self) -> bool {
        self.settings().depth_write
    }
    
    #[inline]
    pub fn depth_test(&self) -> bool {
        self.settings().depth_test
    }
    
    #[inline]
    pub fn side(&self) -> &Side {
        &self.settings().side
    }

    /// 定义 GPU 资源绑定（代理到内部数据）
    #[inline]
    pub fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder)
    }
}

// ============================================================================
// 语法糖：允许从具体材质直接转为通用材质
// ============================================================================

impl From<MeshBasicMaterial> for Material {
    fn from(data: MeshBasicMaterial) -> Self {
        Material::new(MaterialType::Basic(data))
    }
}

impl From<MeshPhongMaterial> for Material {
    fn from(data: MeshPhongMaterial) -> Self {
        Material::new(MaterialType::Phong(data))
    }
}

impl From<MeshStandardMaterial> for Material {
    fn from(data: MeshStandardMaterial) -> Self {
        Material::new(MaterialType::Standard(data))
    }
}

impl From<MeshPhysicalMaterial> for Material {
    fn from(data: MeshPhysicalMaterial) -> Self {
        Material::new(MaterialType::Physical(data))
    }
}

impl Deref for Material {
    type Target = MaterialType;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}