use uuid::Uuid;
use glam::{Vec4};
use std::any::Any;
use wgpu::ShaderStages;
use bitflags::bitflags;

use crate::core::binding::{ResourceBuilder, define_texture_binding};
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
    fn wgsl_struct_def(&self) -> String;

    /// [L0] 刷新 Uniform 数据到 CPU Buffer (极快，每帧可能调用)
    fn flush_uniforms(&self);

    fn define_bindings(&self, builder: &mut ResourceBuilder);

    /// [L2] 获取编译选项 (用于检测 Pipeline 是否需要重建)
    fn get_defines(&self) -> MaterialFeatures;
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
    
    fn wgsl_struct_def(&self) -> String {
        MeshBasicUniforms::wgsl_struct_def("MaterialUniforms")
    }

    fn flush_uniforms(&self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn get_defines(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        features
    }

    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // 1. Uniform Buffer
        builder.add_uniform(
            "MaterialUniforms", 
            &self.uniform_buffer, 
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );

        // 2. Texture + Sampler
        if let Some(id) = self.map {
            define_texture_binding(builder, "map", Some(id));
        }

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

    fn wgsl_struct_def(&self) -> String {
        MeshStandardUniforms::wgsl_struct_def("MaterialUniforms")
    }

    fn flush_uniforms(&self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn get_defines(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        if self.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
        if self.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
        if self.metalness_map.is_some() { features |= MaterialFeatures::USE_METALNESS_MAP; }
        if self.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
        if self.ao_map.is_some() { features |= MaterialFeatures::USE_AO_MAP; }
        
        features
    }

    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // 1. Uniform Buffer
        builder.add_uniform(
            "MaterialUniforms", 
            &self.uniform_buffer, 
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );

        // 2. Maps
        // 辅助闭包：减少重复代码
        let mut add_tex = |name: &str, id: Option<Uuid>| {
            if let Some(tex_id) = id {
                builder.add_texture(name, Some(tex_id), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, ShaderStages::FRAGMENT);
                builder.add_sampler(name, Some(tex_id), wgpu::SamplerBindingType::Filtering, ShaderStages::FRAGMENT);
            }
        };

        add_tex("map", self.map);
        add_tex("normal_map", self.normal_map);
        add_tex("roughness_map", self.roughness_map);
        add_tex("metalness_map", self.metalness_map);
        add_tex("emissive_map", self.emissive_map);
        add_tex("ao_map", self.ao_map);
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
    pub fn wgsl_struct_def(&self) -> String { self.data.wgsl_struct_def() }
    pub fn flush_uniforms(&self) { self.data.flush_uniforms() }
    pub fn get_defines(&self) -> MaterialFeatures { self.data.get_defines() }
    pub fn define_bindings(&self, builder: &mut ResourceBuilder) { self.data.define_bindings(builder); }

}


// === 4. 实现 Bindable (带缓存逻辑) ===

// impl Bindable for Material {
//     // 注意：这里我们修改了签名，返回 Cow 或者 Clone 也可以
//     // 为了兼容之前的接口 (Vec, Vec)，我们这里先 Clone，但由于有缓存，
//     // generate_bindings 的重头戏（逻辑判断、Vec分配）只会发生一次。
//     // *更进一步优化*：可以让 Bindable 返回引用，但这需要改动 trait 定义。
//     // 这里演示“计算缓存”：
    
//     fn get_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'static>>) {
//         let mut cache = self.cache.lock().unwrap();
        
//         // 检查缓存有效性
//         if let Some((ver, ref descs, ref res)) = *cache {
//             if ver == self.version {
//                 // 命中缓存！直接 clone 结果 (Clone Vec 比 重新构建 Vec 快得多)
//                 return (descs.clone(), res.clone());
//             }
//         }

//         // 缓存失效，重新生成
//         let (descs, res) = self.data.generate_bindings();
        
//         // 更新缓存
//         *cache = Some((self.version, descs.clone(), res.clone()));
        
//         (descs, res)
//     }
// }