use uuid::Uuid;
use glam::{Vec4};
use std::any::Any;
use bitflags::bitflags;

use crate::core::buffer::{DataBuffer, BufferRef};
use crate::core::uniforms::{MeshBasicUniforms, MeshStandardUniforms};
use crate::core::Mut;

// Shader 编译选项 (用于 L2 Pipeline 缓存)
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


// === 1. 定义材质属性 Trait (扩展性的关键) ===
pub trait MaterialProperty: Send + Sync + std::fmt::Debug + Any {
    fn shader_name(&self) -> &'static str;

    /// [L0] 刷新 Uniform 数据到 CPU Buffer (极快，每帧可能调用)
    fn flush_uniforms(&self);

    /// [L2] 获取编译选项 (用于检测 Pipeline 是否需要重建)
    fn get_features(&self) -> MaterialFeatures;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}



// ============================================================================
// 具体材质实现 (MeshBasicMaterial)
// ============================================================================
#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub uniforms: MeshBasicUniforms,
    pub uniform_buffer: BufferRef,
    pub map: Option<Uuid>, 
}

impl MaterialProperty for MeshBasicMaterial {
    fn shader_name(&self) -> &'static str { "MeshBasic" }
    
    fn flush_uniforms(&self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn get_features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        features
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}


// ============================================================================
// 具体材质实现 (MeshStandardMaterial)
// ============================================================================

#[derive(Debug, Clone)]
pub struct MeshStandardMaterial {
    pub uniforms: MeshStandardUniforms,
    pub uniform_buffer: BufferRef,
    pub map: Option<Uuid>,
    pub normal_map: Option<Uuid>,
    pub roughness_map: Option<Uuid>,
    pub metalness_map: Option<Uuid>,
    pub emissive_map: Option<Uuid>,
    pub ao_map: Option<Uuid>,
}

impl MaterialProperty for MeshStandardMaterial {
    fn shader_name(&self) -> &'static str { "MeshStandard" }

    fn flush_uniforms(&self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn get_features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        if self.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
        if self.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
        if self.metalness_map.is_some() { features |= MaterialFeatures::USE_METALNESS_MAP; }
        if self.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
        if self.ao_map.is_some() { features |= MaterialFeatures::USE_AO_MAP; }
        
        features
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}





// === 3. Material 主结构体 ===

#[derive(Debug)]
pub struct Material {
    pub id: Uuid,
    pub version: u64,
    pub name: Option<String>,
    pub data: Box<dyn MaterialProperty>, 
    
    // 渲染状态
    pub transparent: bool,
    pub opacity: f32,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32,
}

impl Material {

    pub fn new_basic(color: Vec4) -> Self {
        let uniforms = MeshBasicUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(DataBuffer::new(
            &[uniforms], wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("MeshBasicUniforms")
        ));
        Self {
            id: Uuid::new_v4(), version: 1, name: Some("MeshBasic".into()),
            data: Box::new(MeshBasicMaterial { uniforms, uniform_buffer, map: None }),
            transparent: false, opacity: 1.0, depth_write: true, depth_test: true, cull_mode: Some(wgpu::Face::Back), side: 0,
        }
    }
    
    pub fn new_standard(color: Vec4) -> Self {
        let uniforms = MeshStandardUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(DataBuffer::new(
            &[uniforms], wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("MeshStandardUniforms")
        ));
        Self {
            id: Uuid::new_v4(), version: 1, name: Some("MeshStandard".into()),
            data: Box::new(MeshStandardMaterial { 
                uniforms, uniform_buffer, 
                map: None, normal_map: None, roughness_map: None, metalness_map: None, emissive_map: None, ao_map: None 
            }),
            transparent: false, opacity: 1.0, depth_write: true, depth_test: true, cull_mode: Some(wgpu::Face::Back), side: 0,
        }
    }

    /// 核心 API：获取可变数据引用
    /// 自动处理版本号增加
    pub fn data_mut<T: 'static>(&mut self) -> Option<Mut<'_, T>> {
        let data = self.data.as_any_mut().downcast_mut::<T>()?;
        Some(Mut {
            data,
            version: &mut self.version,
        })
    }

    /// 手动标记材质需要更新 (版本号自增)
    // pub fn needs_update(&mut self) {
    //     self.version = self.version.wrapping_add(1);
    // }

    /// 获取只读数据 (不变)
    pub fn data<T: 'static>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref::<T>()
    }

    // 代理方法
    pub fn shader_name(&self) -> &'static str { self.data.shader_name() }
    pub fn flush_uniforms(&self) { self.data.flush_uniforms() }
    pub fn get_features(&self) -> MaterialFeatures { self.data.get_features() }

}