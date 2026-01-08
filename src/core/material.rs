use uuid::Uuid;
use glam::{Vec4};
use std::any::Any;

use crate::core::binding::{Bindable, BindingDescriptor, BindingResource, BindingType};
use crate::core::buffer::{DataBuffer, BufferRef};
use crate::core::uniforms::{MeshBasicUniforms, MeshStandardUniforms};
use wgpu::ShaderStages;


// ... (MaterialFeatures bitflags 保持不变，但稍后我们会用它来辅助判断) ...
// bitflags! {
//     #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
//     pub struct MaterialFeatures: u32 {
//         const USE_MAP           = 1 << 0;
//         const USE_NORMAL_MAP    = 1 << 1;
//         const USE_ROUGHNESS_MAP = 1 << 2;
//         const USE_METALNESS_MAP = 1 << 3;
//         const USE_EMISSIVE_MAP  = 1 << 4;
//         const USE_AO_MAP        = 1 << 5;
//     }
// }

// === 1. 定义材质属性 Trait (扩展性的关键) ===
pub trait MaterialProperty: Send + Sync + std::fmt::Debug + Any {
    /// 材质对应的 Shader 名称 (决定了 ShaderGenerator 用哪个模板)
    fn shader_name(&self) -> &'static str;

    /// 材质 Uniform 结构体的 WGSL 定义
    fn wgsl_struct_def(&self) -> String;

    /// 获取当前的 Uniform Buffer (用于绑定)
    fn get_uniform_buffer(&self) -> BufferRef;

    /// 生成绑定描述符和资源 (核心逻辑，将被 Material 缓存)
    fn generate_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'static>>);

    /// 刷新 Uniform 数据到 CPU Buffer
    fn flush_uniforms(&mut self);
    
    /// 用于向下转型 (Downcast)
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}



// ... (MeshBasicMaterial, MeshStandardMaterial, MaterialType 保持不变) ...
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

    fn get_uniform_buffer(&self) -> BufferRef {
        self.uniform_buffer.clone()
    }

    fn flush_uniforms(&mut self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn generate_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'static>>) {
        let mut bindings = Vec::new();
        let mut resources = Vec::new();

        // Binding 0: Uniforms
        bindings.push(BindingDescriptor {
            name: "MaterialUniforms",
            index: 0,
            bind_type: BindingType::UniformBuffer { dynamic: false, min_size: None },
            visibility: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
        });
        resources.push(BindingResource::Buffer { 
            buffer: self.get_uniform_buffer(), 
            offset: 0, 
            size: None 
        });

        // Textures
        if self.map.is_some() {
            add_texture_binding(&mut bindings, &mut resources, "map", self.map, 1);
        }

        (bindings, resources)
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}


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
    
    fn get_uniform_buffer(&self) -> BufferRef {
        self.uniform_buffer.clone()
    }

    fn flush_uniforms(&mut self) {
        self.uniform_buffer.write().update(&[self.uniforms]);
    }

    fn generate_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'static>>) {
        let mut bindings = Vec::new();
        let mut resources = Vec::new();

        // Binding 0
        bindings.push(BindingDescriptor {
            name: "MaterialUniforms",
            index: 0,
            bind_type: BindingType::UniformBuffer { dynamic: false, min_size: None },
            visibility: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
        });
        resources.push(BindingResource::Buffer { 
            buffer: self.get_uniform_buffer(), 
            offset: 0, 
            size: None 
        });

        // Textures
        let mut idx = 1;
        if self.map.is_some() { idx = add_texture_binding(&mut bindings, &mut resources, "map", self.map, idx); }
        if self.normal_map.is_some() { idx = add_texture_binding(&mut bindings, &mut resources, "normal_map", self.normal_map, idx); }
        if self.roughness_map.is_some() { idx = add_texture_binding(&mut bindings, &mut resources, "roughness_map", self.roughness_map, idx); }
        if self.metalness_map.is_some() { idx = add_texture_binding(&mut bindings, &mut resources, "metalness_map", self.metalness_map, idx); }
        if self.emissive_map.is_some() { idx = add_texture_binding(&mut bindings, &mut resources, "emissive_map", self.emissive_map, idx); }
        if self.ao_map.is_some() { add_texture_binding(&mut bindings, &mut resources, "ao_map", self.ao_map, idx); }

        (bindings, resources)
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

// 辅助函数：减少代码重复
fn add_texture_binding(
    bindings: &mut Vec<BindingDescriptor>, 
    resources: &mut Vec<BindingResource<'static>>, 
    name: &'static str, 
    id: Option<Uuid>,
    start_index: u32
) -> u32 {
    bindings.push(BindingDescriptor {
        name,
        index: start_index,
        bind_type: BindingType::Texture { 
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false 
        },
        visibility: ShaderStages::FRAGMENT,
    });
    bindings.push(BindingDescriptor {
        name, // sampler name convention handled by generator
        index: start_index + 1,
        bind_type: BindingType::Sampler { 
            type_: wgpu::SamplerBindingType::Filtering 
        },
        visibility: ShaderStages::FRAGMENT,
    });
    resources.push(BindingResource::Texture(id));
    resources.push(BindingResource::Sampler(id));
    start_index + 2
}


// === 3. Material 主结构体 (带缓存) ===

#[derive(Debug)]
pub struct Material {
    pub id: Uuid,
    pub version: u64,
    pub name: Option<String>,
    
    // 扩展性：使用 Box<dyn MaterialProperty> 替代 Enum
    pub data: Box<dyn MaterialProperty>, 
    
    pub transparent: bool,
    pub opacity: f32,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32,

    // === 性能核心：缓存 ===
    // 只有当 version 变化时才更新
    // 使用 interior mutability (RefCell) 因为 get_bindings 是 &self
    // 但在多线程环境下，Mutex 更安全（虽然 Renderer 此时可能是单线程访问）
    // 为了简化，我们假设 Renderer 拥有独占访问权，或者我们可以直接在 needs_update 时清空
    // 这里我们简单地把 cache 放在结构体里，并在 get_bindings 时 lazy update
    // 由于 trait 定义限制，我们通过方法返回 &Vec
    
    // 缓存: (Version, Bindings, Resources)
    cache: std::sync::Mutex<Option<(u64, Vec<BindingDescriptor>, Vec<BindingResource<'static>>)>>,
}

impl Material {
    // 构造函数示例
    pub fn new_basic(color: Vec4) -> Self {
        let uniforms = MeshBasicUniforms { color, ..Default::default() };
        let uniform_buffer = BufferRef::new(DataBuffer::new(
            &[uniforms],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("MeshBasicUniforms")
        ));

        Self {
            id: Uuid::new_v4(),
            version: 1, // Start with 1
            name: Some("MeshBasic".to_string()),
            data: Box::new(MeshBasicMaterial {
                uniforms,
                uniform_buffer,
                map: None,
            }),
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
            cache: std::sync::Mutex::new(None),
        }
    }
    
    // 省略 new_standard ... 逻辑类似

    /// 标记材质需要更新（Dirty Flag 的手动触发器）
    /// 用户修改属性后必须调用（或者通过 setter 自动调用）
    pub fn needs_update(&mut self) {
        self.version = self.version.wrapping_add(1);
        // 可以在这里清空缓存，也可以在 get 时 lazy check
    }

    /// 获取数据引用的泛型方法
    pub fn get_data<T: 'static>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref::<T>()
    }
    
    pub fn get_data_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.needs_update(); // 获取可变引用通常意味着要修改，自动标记 dirty
        self.data.as_any_mut().downcast_mut::<T>()
    }

    // 代理方法
    pub fn shader_name(&self) -> &'static str { self.data.shader_name() }
    pub fn wgsl_struct_def(&self) -> String { self.data.wgsl_struct_def() }
    pub fn flush_uniforms(&mut self) { self.data.flush_uniforms() }
}


// === 4. 实现 Bindable (带缓存逻辑) ===

impl Bindable for Material {
    // 注意：这里我们修改了签名，返回 Cow 或者 Clone 也可以
    // 为了兼容之前的接口 (Vec, Vec)，我们这里先 Clone，但由于有缓存，
    // generate_bindings 的重头戏（逻辑判断、Vec分配）只会发生一次。
    // *更进一步优化*：可以让 Bindable 返回引用，但这需要改动 trait 定义。
    // 这里演示“计算缓存”：
    
    fn get_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'static>>) {
        let mut cache = self.cache.lock().unwrap();
        
        // 检查缓存有效性
        if let Some((ver, ref descs, ref res)) = *cache {
            if ver == self.version {
                // 命中缓存！直接 clone 结果 (Clone Vec 比 重新构建 Vec 快得多)
                return (descs.clone(), res.clone());
            }
        }

        // 缓存失效，重新生成
        let (descs, res) = self.data.generate_bindings();
        
        // 更新缓存
        *cache = Some((self.version, descs.clone(), res.clone()));
        
        (descs, res)
    }
}