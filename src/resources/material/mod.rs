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

use crate::assets::TextureHandle;
use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use glam::{Mat3A, Vec2, Vec4};
use uuid::Uuid;

// ============================================================================
// TextureSlot 架构
// ============================================================================

/// 纹理变换参数
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    pub offset: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
}

impl Default for TextureTransform {
    #[inline]
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }
}

/// 纹理槽位：封装纹理引用与变换参数
#[derive(Clone, Debug, Default)]
pub struct TextureSlot {
    pub texture: Option<TextureHandle>,
    pub transform: TextureTransform,
    pub channel: u8,
}

impl TextureSlot {
    #[inline]
    pub fn new(handle: TextureHandle) -> Self {
        Self {
            texture: Some(handle),
            transform: TextureTransform::default(),
            channel: 0,
        }
    }

    #[inline]
    pub fn with_transform(handle: TextureHandle, transform: TextureTransform) -> Self {
        Self {
            texture: Some(handle),
            transform,
            channel: 0,
        }
    }

    /// 计算 UV 变换矩阵 (3x3)
    /// 变换顺序: Translate * Rotate * Scale
    #[inline]
    pub fn compute_matrix(&self) -> Mat3A {
        let (s, c) = self.transform.rotation.sin_cos();
        let sx = self.transform.scale.x;
        let sy = self.transform.scale.y;
        
        // 列主序矩阵 (Column-Major):
        // | sx*c   -sy*s   tx |
        // | sx*s    sy*c   ty |
        // |  0       0      1 |
        Mat3A::from_cols_array(&[
            sx * c, sx * s, 0.0,
            -sy * s, sy * c, 0.0,
            self.transform.offset.x, self.transform.offset.y, 1.0,
        ])
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        self.texture.is_some()
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        self.texture.is_none()
    }
    
    /// 设置纹理
    #[inline]
    pub fn set_texture(&mut self, handle: Option<TextureHandle>) {
        self.texture = handle;
    }
}

// ============================================================================
// TextureSlotGuard - 纹理槽位修改守卫
// ============================================================================

/// 纹理槽位修改守卫
/// 
/// 当纹理的有/无状态发生变化时（影响 Shader 宏），自动递增版本号。
/// 使用 RAII 模式确保版本控制的正确性。
pub struct TextureSlotGuard<'a> {
    slot: &'a mut TextureSlot,
    version: &'a mut u64,
    was_some: bool,
}

impl<'a> TextureSlotGuard<'a> {
    /// 创建纹理槽位守卫
    #[inline]
    pub fn new(slot: &'a mut TextureSlot, version: &'a mut u64) -> Self {
        let was_some = slot.texture.is_some();
        Self { slot, version, was_some }
    }
}

impl std::ops::Deref for TextureSlotGuard<'_> {
    type Target = TextureSlot;
    
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.slot
    }
}

impl std::ops::DerefMut for TextureSlotGuard<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slot
    }
}

impl Drop for TextureSlotGuard<'_> {
    fn drop(&mut self) {
        let is_some = self.slot.texture.is_some();
        if self.was_some != is_some {
            *self.version = self.version.wrapping_add(1);
        }
    }
}

impl From<TextureHandle> for TextureSlot {
    #[inline]
    fn from(handle: TextureHandle) -> Self {
        Self::new(handle)
    }
}

impl From<Option<TextureHandle>> for TextureSlot {
    #[inline]
    fn from(opt: Option<TextureHandle>) -> Self {
        Self {
            texture: opt,
            transform: TextureTransform::default(),
            channel: 0,
        }
    }
}

impl From<TextureSource> for TextureSlot {
    #[inline]
    fn from(source: TextureSource) -> Self {
        match source {
            TextureSource::Asset(handle) => Self::new(handle),
            TextureSource::Attachment(_, _) => Self::default(),
        }
    }
}

impl From<Option<TextureSource>> for TextureSlot {
    #[inline]
    fn from(opt: Option<TextureSource>) -> Self {
        match opt {
            Some(TextureSource::Asset(handle)) => Self::new(handle),
            _ => Self::default(),
        }
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
    /// 着色器模板名称
    fn shader_name(&self) -> &'static str;
    /// 材质版本号（用于缓存失效）
    fn version(&self) -> u64;
    /// 获取材质的 Shader 宏定义
    fn shader_defines(&self) -> ShaderDefines;
    /// 材质渲染设置
    fn settings(&self) -> &MaterialSettings;
    
    /// 访问所有纹理
    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource));
    /// 定义 GPU 资源绑定
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
    /// 获取 Uniform 缓冲区引用
    fn uniform_buffer(&self) -> &BufferRef;
    /// 获取 Uniform 数据字节
    fn uniform_bytes(&self) -> &[u8];
}

#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum Side {
    Front,
    Back,
    Double,
}

#[derive(PartialEq, Clone, Debug, Copy)]
pub enum AlphaMode {
    Opaque,
    Mask(f32), // alpha cutoff
    Blend,
}

/// 材质设置 - 对应 Pipeline 变化
#[derive(PartialEq, Clone, Debug)]
pub struct MaterialSettings {
    pub(crate) alpha_mode: AlphaMode,
    pub(crate) depth_write: bool,
    pub(crate) depth_test: bool,
    pub(crate) side: Side,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            alpha_mode: AlphaMode::Opaque,
            depth_write: true, 
            depth_test: true,
            side: Side::Front,
        }
    }
}

impl MaterialSettings {
    /// 生成 Shader 宏定义
    pub(crate) fn generate_shader_defines(&self, defines: &mut ShaderDefines) {
        // Alpha Mode
        match self.alpha_mode {
            AlphaMode::Opaque => {
                defines.set("ALPHA_MODE", "OPAQUE");
            }
            AlphaMode::Mask(_cutoff) => {
                defines.set("ALPHA_MODE", "MASK");
            }
            AlphaMode::Blend => {
                defines.set("ALPHA_MODE", "BLEND");
            }
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

    fn shader_defines(&self) -> ShaderDefines {
        match self {
            Self::Basic(m) => m.shader_defines(),
            Self::Phong(m) => m.shader_defines(),
            Self::Standard(m) => m.shader_defines(),
            Self::Physical(m) => m.shader_defines(),
            Self::Custom(m) => m.shader_defines(),
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
    pub(crate) fn shader_name(&self) -> &'static str { 
        self.data.shader_name() 
    }
    
    #[inline]
    pub(crate) fn shader_defines(&self) -> ShaderDefines { 
        self.data.shader_defines() 
    }
    
    #[inline]
    pub(crate) fn settings(&self) -> &MaterialSettings {
        self.data.settings()
    }
    
    // 便捷访问器
    // #[inline]
    // pub fn transparent(&self) -> bool {
    //     self.settings().transparent
    // }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.settings().alpha_mode
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

    #[inline]
    pub fn auto_sync_texture_to_uniforms(&self) -> bool {
        match &self.data {
            MaterialType::Basic(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Phong(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Standard(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Physical(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Custom(_) => false,
        }
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