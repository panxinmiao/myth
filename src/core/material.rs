use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use glam::Vec4;
use bitflags::bitflags;

use crate::core::buffer::{BufferRef};
use crate::core::uniforms::{MeshBasicUniforms, MeshStandardUniforms};
use crate::core::assets::TextureHandle;

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
    }
}

// ============================================================================
// 具体材质定义 (Specific Materials)
// ============================================================================

// 1. MeshBasicMaterial
// ----------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub uniforms: MeshBasicUniforms,
    // 内部使用的 Uniform Buffer，自动管理
    pub uniform_buffer: BufferRef,
    
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

// 2. MeshStandardMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshStandardMaterial {
    pub uniforms: MeshStandardUniforms,
    pub uniform_buffer: BufferRef,
    
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

/// MeshBasicMaterial 的构建器
pub struct MeshBasicMaterialBuilder {
    // Specific Properties
    color: Vec4,
    map: Option<TextureHandle>,

    // Common Properties (默认值)
    name: Option<String>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    cull_mode: Option<wgpu::Face>,
    side: u32,
}

impl MeshBasicMaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Vec4::ONE,
            map: None,
            // Common Defaults
            name: None,
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn map(mut self, map: TextureHandle) -> Self { self.map = Some(map); self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(name.into()); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self { self.cull_mode = mode; self }
    pub fn side(mut self, side: u32) -> Self { self.side = side; self }

    /// 构建最终的 Material
    pub fn build(self) -> Material {
        let mut basic = MeshBasicMaterial::new(self.color);
        basic.map = self.map;

        Material {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
            name: self.name,
            data: MaterialData::Basic(basic), // 自动装箱
            transparent: self.transparent,
            opacity: self.opacity,
            depth_write: self.depth_write,
            depth_test: self.depth_test,
            cull_mode: self.cull_mode,
            side: self.side,
        }
    }
}

/// MeshStandardMaterial 的构建器
pub struct MeshStandardMaterialBuilder {
    // Specific
    color: Vec4,
    roughness: f32,
    metalness: f32,
    map: Option<TextureHandle>,
    normal_map: Option<TextureHandle>,
    roughness_map: Option<TextureHandle>,
    metalness_map: Option<TextureHandle>,
    emissive_map: Option<TextureHandle>,
    ao_map: Option<TextureHandle>,

    // Common
    name: Option<String>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    cull_mode: Option<wgpu::Face>,
    side: u32,
}

impl MeshStandardMaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Vec4::ONE,
            roughness: 0.5,
            metalness: 0.5,
            map: None, normal_map: None, roughness_map: None, metalness_map: None, emissive_map: None, ao_map: None,
            // Common Defaults
            name: None,
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn roughness(mut self, value: f32) -> Self { self.roughness = value; self }
    pub fn metalness(mut self, value: f32) -> Self { self.metalness = value; self }
    
    pub fn map(mut self, map: TextureHandle) -> Self { self.map = Some(map); self }
    pub fn normal_map(mut self, map: TextureHandle) -> Self { self.normal_map = Some(map); self }
    pub fn roughness_map(mut self, map: TextureHandle) -> Self { self.roughness_map = Some(map); self }
    pub fn metalness_map(mut self, map: TextureHandle) -> Self { self.metalness_map = Some(map); self }
    pub fn emissive_map(mut self, map: TextureHandle) -> Self { self.emissive_map = Some(map); self }
    pub fn ao_map(mut self, map: TextureHandle) -> Self { self.ao_map = Some(map); self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(name.into()); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self { self.cull_mode = mode; self }
    pub fn side(mut self, side: u32) -> Self { self.side = side; self }
    
    pub fn build(self) -> Material {
        let mut standard = MeshStandardMaterial::new(self.color);
        standard.uniforms.roughness = self.roughness;
        standard.uniforms.metalness = self.metalness;
        standard.map = self.map;
        standard.normal_map = self.normal_map;
        standard.roughness_map = self.roughness_map;
        standard.metalness_map = self.metalness_map;
        standard.emissive_map = self.emissive_map;
        standard.ao_map = self.ao_map;

        Material {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
            name: self.name,
            data: MaterialData::Standard(standard),
            transparent: self.transparent,
            opacity: self.opacity,
            depth_write: self.depth_write,
            depth_test: self.depth_test,
            cull_mode: self.cull_mode,
            side: self.side,
        }
    }
}

// ============================================================================
// 核心材质枚举 (Material Data Enum)
// ============================================================================
// 这里替代了之前的 Box<dyn MaterialProperty>，静态分发性能更好且易于使用

#[derive(Debug)]
pub enum MaterialData {
    Basic(MeshBasicMaterial),
    Standard(MeshStandardMaterial),
    // 可以在这里扩展更多类型，例如 PBR, CustomShader 等
}

impl MaterialData {
    pub fn shader_name(&self) -> &'static str {
        match self {
            Self::Basic(_) => "mesh_basic",
            Self::Standard(_) => "MeshStandard", // 注意：保持和你 shader文件名/entry point 一致
        }
    }

    pub fn flush_uniforms(&self) {
        match self {
            Self::Basic(m) => m.uniform_buffer.update(&[m.uniforms]),
            Self::Standard(m) => m.uniform_buffer.update(&[m.uniforms]),
        }
    }

    pub fn get_features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        match self {
            Self::Basic(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
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
    pub opacity: f32,
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
            opacity: 1.0,
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
    pub fn new_basic(color: Vec4) -> MeshBasicMaterial {
        MeshBasicMaterial::new(color)
    }

    // 辅助构造：Standard
    pub fn new_standard(color: Vec4) -> MeshStandardMaterial {
        MeshStandardMaterial::new(color)
    }

    // 代理方法：直接转发给内部数据
    pub fn shader_name(&self) -> &'static str { self.data.shader_name() }
    pub fn flush_uniforms(&self) { self.data.flush_uniforms() }
    pub fn get_features(&self) -> MaterialFeatures { self.data.get_features() }
}

// ============================================================================
// 语法糖：允许从 具体材质 直接转为 通用材质
// ============================================================================

impl From<MeshBasicMaterial> for Material {
    fn from(data: MeshBasicMaterial) -> Self {
        Material::new(MaterialData::Basic(data))
    }
}

impl From<MeshStandardMaterial> for Material {
    fn from(data: MeshStandardMaterial) -> Self {
        Material::new(MaterialData::Standard(data))
    }
}