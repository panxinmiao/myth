// src/core/material.rs
use uuid::Uuid;
use glam::{Vec3, Vec4};
use bytemuck::{Pod, Zeroable};
use bitflags::bitflags;
use wgpu::ShaderStages;

// 引入 Binding 抽象
use crate::core::binding::{Bindable, BindingDescriptor, BindingResource, BindingType};

// ... (MeshBasicUniforms 和 MeshStandardUniforms 保持不变，省略以节省空间) ...
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct MeshBasicUniforms {
    pub color: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshStandardUniforms {
    pub color: Vec4,
    pub emissive: Vec3,
    pub roughness: f32,
    pub metalness: f32,
    pub _padding: [f32; 3],
}
impl Default for MeshStandardUniforms {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            emissive: Vec3::ZERO,
            roughness: 0.5,
            metalness: 0.0,
            _padding: [0.0; 3],
        }
    }
}

// ... (MaterialFeatures bitflags 保持不变，但稍后我们会用它来辅助判断) ...
bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MaterialFeatures: u32 {
        const USE_MAP           = 1 << 0;
        const USE_NORMAL_MAP    = 1 << 1;
        const USE_ROUGHNESS_MAP = 1 << 2;
        const USE_METALNESS_MAP = 1 << 3;
        const USE_EMISSIVE_MAP  = 1 << 4;
        const USE_AO_MAP        = 1 << 5;
    }
}

// ... (MeshBasicMaterial, MeshStandardMaterial, MaterialType 保持不变) ...
#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub uniforms: MeshBasicUniforms,
    pub map: Option<Uuid>, 
}

#[derive(Debug, Clone)]
pub struct MeshStandardMaterial {
    pub uniforms: MeshStandardUniforms,
    pub map: Option<Uuid>,
    pub normal_map: Option<Uuid>,
    pub roughness_map: Option<Uuid>,
    pub metalness_map: Option<Uuid>,
    pub emissive_map: Option<Uuid>,
    pub ao_map: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub enum MaterialType {
    Basic(MeshBasicMaterial),
    Standard(MeshStandardMaterial),
}

// ... (Material 结构体保持不变) ...
#[derive(Debug, Clone)]
pub struct Material {
    pub id: Uuid,
    pub version: u64,
    pub name: Option<String>,
    pub data: MaterialType,
    pub transparent: bool,
    pub opacity: f32,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32,
}

impl Material {
    // ... (new_basic, new_standard 构造函数保持不变) ...
    pub fn new_basic(color: Vec4) -> Self {
         Self {
            id: Uuid::new_v4(),
            version: 0,
            name: Some("MeshBasic".to_string()),
            data: MaterialType::Basic(MeshBasicMaterial {
                uniforms: MeshBasicUniforms { color },
                map: None,
            }),
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    pub fn new_standard() -> Self {
        Self {
            id: Uuid::new_v4(),
            version: 0,
            name: Some("MeshStandard".to_string()),
            data: MaterialType::Standard(MeshStandardMaterial {
                uniforms: MeshStandardUniforms::default(),
                map: None,
                normal_map: None,
                roughness_map: None,
                metalness_map: None,
                emissive_map: None,
                ao_map: None,
            }),
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    /// 根据当前绑定的资源计算 Features
    pub fn features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        match &self.data {
            MaterialType::Basic(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
            }
            MaterialType::Standard(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
                if m.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
            }
        }
        features
    }

    pub fn shader_name(&self) -> &'static str {
        match &self.data {
            MaterialType::Basic(_) => "MeshBasic",
            MaterialType::Standard(_) => "MeshStandard",
        }
    }
    
    pub fn wgsl_struct_def(&self) -> &'static str {
        // ... (保持原样) ...
         match &self.data {
            MaterialType::Basic(_) => r#"
                struct MaterialUniforms {
                    color: vec4<f32>,
                };
            "#,
            MaterialType::Standard(_) => r#"
                struct MaterialUniforms {
                    color: vec4<f32>,
                    emissive: vec3<f32>,
                    roughness: f32,
                    metalness: f32,
                };
            "#,
        }
    }
    
    // 辅助函数：将 Uniform 转为字节
    fn as_bytes(&self) -> &[u8] {
        match &self.data {
            MaterialType::Basic(m) => bytemuck::bytes_of(&m.uniforms),
            MaterialType::Standard(m) => bytemuck::bytes_of(&m.uniforms),
        }
    }

    pub fn needs_update(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

// === 核心实现：Bindable Trait ===
// 这里的逻辑取代了原本 Resource Manager 中硬编码的 TEXTURE_DEFINITIONS

impl Bindable for Material {

    fn get_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'_>>) {
       let mut bindings = Vec::new();
       let mut resources = Vec::new();
       // let features = self.features(); // 复用 features 逻辑来判断是否需要生成 Binding

       // 1. Binding 0: Uniform Buffer (所有材质都有)
       bindings.push(BindingDescriptor {
           name: "MaterialUniforms",
           index: 0,
           bind_type: BindingType::UniformBuffer,
           visibility: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
       });
       resources.push(BindingResource::Buffer(self.as_bytes()));

       // 2. 动态纹理 Bindings
       // 我们维护一个当前的 binding_index
       let mut current_index = 1;

       // 辅助闭包：减少重复代码
       let mut add_texture_slot = |name: &'static str, id: Option<Uuid>| {
           // Texture Slot
           bindings.push(BindingDescriptor {
               name, // 例如 "map" -> 对应 Shader 中的 t_map
               index: current_index,
               bind_type: BindingType::Texture { 
                   sample_type: wgpu::TextureSampleType::Float { filterable: true },
                   view_dimension: wgpu::TextureViewDimension::D2,
                   multisampled: false 
               },
               visibility: ShaderStages::FRAGMENT,
           });
           // Sampler Slot (紧跟 Texture)
           bindings.push(BindingDescriptor {
               name, // Sampler 复用名字，Shader 生成器会处理成 s_map
               index: current_index + 1,
               bind_type: BindingType::Sampler { 
                   type_: wgpu::SamplerBindingType::Filtering 
               },
               visibility: ShaderStages::FRAGMENT,
           });
           resources.push(BindingResource::Texture(id));
           resources.push(BindingResource::Sampler(id)); // 目前假设 Sampler 与 Texture 绑定
           current_index += 2;
       };

       match &self.data {
           MaterialType::Basic(m) => {
               // if features.contains(MaterialFeatures::USE_MAP) { add_texture_slot("map", m.map); }
               if m.map.is_some() {
                   add_texture_slot("map", m.map);
               }
           },
           MaterialType::Standard(m) => {
            //    if features.contains(MaterialFeatures::USE_MAP) { add_texture_slot("map", m.map); }
            //    if features.contains(MaterialFeatures::USE_NORMAL_MAP) { add_texture_slot("normal_map", m.normal_map); }
            //    if features.contains(MaterialFeatures::USE_ROUGHNESS_MAP) { add_texture_slot("roughness_map", m.roughness_map); }
            //    if features.contains(MaterialFeatures::USE_METALNESS_MAP) { add_texture_slot("metalness_map", m.metalness_map); }
            //    if features.contains(MaterialFeatures::USE_EMISSIVE_MAP) { add_texture_slot("emissive_map", m.emissive_map); }
            //    if features.contains(MaterialFeatures::USE_AO_MAP) { add_texture_slot("ao_map", m.ao_map); }
               if m.map.is_some() { add_texture_slot("map", m.map);}
               if m.normal_map.is_some() { add_texture_slot("normal_map", m.normal_map);}
               if m.roughness_map.is_some() { add_texture_slot("roughness_map", m.roughness_map);}
               if m.metalness_map.is_some() { add_texture_slot("metalness_map", m.metalness_map);}
               if m.emissive_map.is_some() { add_texture_slot("emissive_map", m.emissive_map);}
               if m.ao_map.is_some() { add_texture_slot("ao_map", m.ao_map);}    

           }
       }

       (bindings, resources)
    }

}