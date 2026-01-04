use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use glam::{Vec2, Vec3, Vec4, Mat3, Mat4};
use crate::core::texture::Texture; // 引入 Texture
use wgpu::{CompareFunction, Face};

// ============================================================================
// 1. 底层核心：通用材质数据 (Engine Level)
// ============================================================================

/// 材质属性值的枚举
/// 涵盖了 Shader Uniform 可能用到的所有类型
#[derive(Debug, Clone)]
pub enum MaterialValue {
    Float(f32),
    Int(i32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4), // 颜色通常存储为 Vec4 (RGBA)
    Mat3(Mat3),
    Mat4(Mat4),

    // 直接持有纹理资源的共享引用
    // 使用 RwLock 是为了支持在运行时修改纹理属性（如 Filter, WrapMode）
    Texture(Arc<RwLock<Texture>>),
}

/// 通用材质结构体
/// 渲染器只与这个结构体打交道
#[derive(Debug, Clone)]
pub struct Material {
    // === 标识 ===
    pub id: Uuid,
    pub name: String,
    
    // === Shader 映射 ===
    // 渲染器根据这个名字去查找 WGSL 代码
    // 例如: "MeshBasic", "MeshStandard", "MeshPhysical"
    pub type_name: String, 

    // === 属性表 (Uniforms & Textures) ===
    pub properties: HashMap<String, MaterialValue>,
    
    // === 渲染状态 (Pipeline Config) ===
    pub transparent: bool,              // 是否开启混合
    pub depth_write: bool,              // 是否写入深度
    pub depth_compare: CompareFunction, // 深度测试函数
    pub cull_mode: Option<Face>,        // 剔除模式 (None=双面, Back=剔除背面)
    
    // === 状态控制 ===
    // 脏标记：当属性改变时 +1，通知 Renderer 更新 BindGroup
    pub version: u64,
}

impl Material {
    /// 创建一个基础材质容器
    pub fn new(type_name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: type_name.to_string(),
            type_name: type_name.to_string(),
            properties: HashMap::new(),
            
            // 默认渲染状态 (PBR 标准)
            transparent: false,
            depth_write: true,
            depth_compare: CompareFunction::Less,
            cull_mode: Some(Face::Back), // 默认单面渲染
            
            version: 0,
        }
    }

    // === 通用属性设置 ===

    pub fn set_uniform(&mut self, key: &str, value: MaterialValue) {
        self.properties.insert(key.to_string(), value);
        self.version += 1;
    }
    
    pub fn get_uniform(&self, key: &str) -> Option<&MaterialValue> {
        self.properties.get(key)
    }

    // === 常用快捷方法 ===
    
    pub fn set_color(&mut self, color: Vec4) {
        self.set_uniform("color", MaterialValue::Vec4(color));
    }
    
    pub fn set_map(&mut self, texture: Arc<RwLock<Texture>>) {
        self.set_uniform("map", MaterialValue::Texture(texture));
    }

    /// 设置为双面渲染
    pub fn set_double_sided(&mut self) {
        self.cull_mode = None;
    }
}

// ============================================================================
// 2. 中间层：材质定义 Trait (Bridge)
// ============================================================================

/// 任何实现了此 Trait 的结构体都可以被转换为通用 Material
pub trait MaterialDef {
    fn to_material(&self) -> Material;
}

// ============================================================================
// 3. 上层应用：强类型材质 (User Level)
// ============================================================================

// --- A. 基础材质 (MeshBasicMaterial) ---
// 不受光照影响，常用于 UI 或 Debug
#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub color: Vec4,
    pub map: Option<Arc<RwLock<Texture>>>,
    pub transparent: bool,
    pub wireframe: bool, // 如果支持线框模式，这会改变 type_name
}

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            map: None,
            transparent: false,
            wireframe: false,
        }
    }
}

impl MaterialDef for MeshBasicMaterial {
    fn to_material(&self) -> Material {
        let type_name = if self.wireframe { "LineBasic" } else { "MeshBasic" };
        let mut mat = Material::new(type_name);
        
        mat.set_uniform("color", MaterialValue::Vec4(self.color));
        if let Some(tex) = &self.map {
            mat.set_uniform("map", MaterialValue::Texture(tex.clone()));
        }
        
        mat.transparent = self.transparent;
        mat
    }
}

// --- B. 标准 PBR 材质 (MeshStandardMaterial) ---
// 你的数字人主要会用这个
#[derive(Debug, Clone)]
pub struct MeshStandardMaterial {
    pub color: Vec4,
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: Vec3,
    pub occlusion_strength: f32, // AO 强度
    pub normal_scale: Vec2,      // 法线缩放
    
    // 纹理贴图
    pub map: Option<Arc<RwLock<Texture>>>,           // Albedo / Diffuse
    pub normal_map: Option<Arc<RwLock<Texture>>>,    // Normal Map
    pub roughness_map: Option<Arc<RwLock<Texture>>>, // Roughness Map
    pub metalness_map: Option<Arc<RwLock<Texture>>>, // Metalness Map
    pub emissive_map: Option<Arc<RwLock<Texture>>>,  // Emissive Map
    pub occlusion_map: Option<Arc<RwLock<Texture>>>, // AO Map
    
    pub transparent: bool,
    pub double_sided: bool,
}

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            roughness: 0.5,
            metalness: 0.0, // 绝缘体默认
            emissive: Vec3::ZERO,
            occlusion_strength: 1.0,
            normal_scale: Vec2::ONE,
            
            map: None,
            normal_map: None,
            roughness_map: None,
            metalness_map: None,
            emissive_map: None,
            occlusion_map: None,
            
            transparent: false,
            double_sided: false,
        }
    }
}

impl MaterialDef for MeshStandardMaterial {
    fn to_material(&self) -> Material {
        let mut mat = Material::new("MeshStandard");
        
        // 基础属性
        mat.set_uniform("color", MaterialValue::Vec4(self.color));
        mat.set_uniform("roughness", MaterialValue::Float(self.roughness));
        mat.set_uniform("metalness", MaterialValue::Float(self.metalness));
        mat.set_uniform("emissive", MaterialValue::Vec3(self.emissive));
        mat.set_uniform("occlusionStrength", MaterialValue::Float(self.occlusion_strength));
        mat.set_uniform("normalScale", MaterialValue::Vec2(self.normal_scale));
        
        // 贴图
        if let Some(tex) = &self.map { mat.set_uniform("map", MaterialValue::Texture(tex.clone())); }
        if let Some(tex) = &self.normal_map { mat.set_uniform("normalMap", MaterialValue::Texture(tex.clone())); }
        if let Some(tex) = &self.roughness_map { mat.set_uniform("roughnessMap", MaterialValue::Texture(tex.clone())); }
        if let Some(tex) = &self.metalness_map { mat.set_uniform("metalnessMap", MaterialValue::Texture(tex.clone())); }
        if let Some(tex) = &self.emissive_map { mat.set_uniform("emissiveMap", MaterialValue::Texture(tex.clone())); }
        if let Some(tex) = &self.occlusion_map { mat.set_uniform("occlusionMap", MaterialValue::Texture(tex.clone())); }
        
        // 渲染状态
        mat.transparent = self.transparent;
        if self.double_sided {
            mat.cull_mode = None;
        }

        mat
    }
}