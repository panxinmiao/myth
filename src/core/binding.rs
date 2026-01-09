use uuid::Uuid;
use wgpu::{ShaderStages, wgc::device::resource};
use crate::core::buffer::BufferRef; // 引入新类型

/// 定义绑定的具体类型 (Schema)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// Uniform Buffer (通常用于材质参数，全局变量)
    UniformBuffer { 
        dynamic: bool,        // 是否开启动态偏移 (has_dynamic_offset)
        min_size: Option<u64> // 最小绑定大小 (min_binding_size)
    },
    
    /// Storage Buffer (只读/读写，用于骨骼矩阵、粒子等)
    StorageBuffer { read_only: bool },
    
    /// 纹理
    Texture {
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        multisampled: bool,
    },
    
    /// 采样器
    Sampler {
        type_: wgpu::SamplerBindingType,
    },
}


/// 实际的绑定资源数据 (用于生成 BindGroup)
/// Core 层只持有 ID 或 数据引用，不持有 GPU 句柄
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingResource {

    /// 持有 CPU Buffer 的引用 (统一了 Vertex/Index/Uniform/Storage)
    Buffer {
        buffer: BufferRef,
        offset: u64,        // 偏移量 (默认为 0)
        size: Option<u64>,  // 绑定窗口大小 (None 表示整个 Buffer)
    },
    
    /// 外部 Buffer ID (用于高级场景，暂保留)
    // BufferId(Uuid),

    /// 纹理 ID (可能为空，意味着需要使用缺省纹理)
    Texture(Option<Uuid>),
    
    /// 采样器 ID (通常跟随纹理，但也可以独立)
    Sampler(Option<Uuid>),

}

pub struct ResourceBuilder {
    // 生成 Layout 需要的信息
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    // 生成 BindGroup 需要的信息 (中间态)
    pub resources: Vec<BindingResource>,
    pub names: Vec<String>,
    // 自动维护 binding index
    pub next_binding_index: u32,
}

impl ResourceBuilder {
    pub fn new() -> Self {
        Self {
            layout_entries: Vec::new(),
            resources: Vec::new(),
            names: Vec::new(),
            next_binding_index: 0,
        }
    }

    /// 添加 Uniform Buffer
    pub fn add_uniform(&mut self, _name: &str, buffer: &BufferRef, visibility: ShaderStages) {
        // 1. Layout Entry
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None, // 或者根据 buffer.size 推断
            },
            count: None,
        });

        // 2. Resource Data (保存 BufferRef，稍后解析)
        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(), // Arc 克隆，开销很小
            offset: 0,
            size: None,
        });

        self.names.push(_name.to_string());
        self.next_binding_index += 1;
    }

    pub fn add_dynamic_uniform(&mut self, name: &str, buffer: &BufferRef, min_binding_size: u64, visibility: ShaderStages) {
        // 1. Layout Entry: 开启 has_dynamic_offset
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true, // <--- 关键点
                min_binding_size: std::num::NonZeroU64::new(min_binding_size),
            },
            count: None,
        });

        // 2. Resource: 绑定一个视窗 (Window)
        // 注意：对于 Dynamic Buffer，这里绑定的 size 通常是单个结构体的大小
        // 实际的偏移量会在 RenderPass set_bind_group 时传入
        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(),
            offset: 0,
            size: Some(min_binding_size),
        });

        self.names.push(name.to_string());
        self.next_binding_index += 1;
    }

    /// 添加 Texture (通过 UUID)
    pub fn add_texture(&mut self, _name: &str, texture_id: Option<Uuid>, sample_type: wgpu::TextureSampleType, view_dimension: wgpu::TextureViewDimension, visibility: ShaderStages) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: false,
            },
            count: None,
        });

        self.resources.push(BindingResource::Texture(texture_id));
        self.names.push(_name.to_string());
        self.next_binding_index += 1;
    }

    /// 添加 Sampler (通过 UUID)
    pub fn add_sampler(&mut self, _name: &str, texture_id: Option<Uuid>, sampler_type: wgpu::SamplerBindingType, visibility: ShaderStages) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Sampler(sampler_type),
            count: None,
        });

        self.resources.push(BindingResource::Sampler(texture_id));
        self.names.push(_name.to_string());
        self.next_binding_index += 1;
    }

    pub fn add_storage_buffer(&mut self, _name: &str, buffer: &BufferRef, read_only: bool, visibility: ShaderStages) {
        // 1. Layout Entry
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: if read_only {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None, // 或者根据 buffer.size 推断
            },
            count: None,
        });

        // 2. Resource Data (保存 BufferRef，稍后解析)
        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(), // Arc 克隆，开销很小
            offset: 0,
            size: None,
        });

        self.names.push(_name.to_string());
        self.next_binding_index += 1;
    }

    pub fn generate_wgsl(&self, group_index: u32) -> String {
        let mut code = String::new();

        for (i, entry) in self.layout_entries.iter().enumerate() {
            let name = &self.names[i];
            let resource = &self.resources[i];
            let binding_index = entry.binding; 

            let decl = match entry.ty {
                wgpu::BindingType::Buffer { ty, .. } => {
                    match ty {
                        wgpu::BufferBindingType::Uniform => {
                            // 简单的类型推断约定
                            let type_name = if name == "material" {
                                "MaterialUniforms"
                            } else if name == "mesh_model" || name == "DynamicModel" {
                                "DynamicModel" 
                            } else {
                                // 对于未知 Uniform，假设结构体名就是大写的变量名 (或者需要扩展 builder 传入类型名)
                                // 这里为了演示，暂时由外部保证名字匹配
                                "MaterialUniforms" 
                            };
                            format!("@group({}) @binding({}) var<uniform> {}: {};", group_index, binding_index, name, type_name)
                        },
                        wgpu::BufferBindingType::Storage { read_only } => {
                            let access = if read_only { "read" } else { "read_write" };
                            let type_str = if name == "morph_data" {
                                "array<f32>" 
                            } else if name.contains("bone") {
                                "array<mat4x4<f32>>"
                            } else {
                                "array<f32>"
                            };
                            format!("@group({}) @binding({}) var<storage, {}> {}: {};", group_index, binding_index, access, name, type_str)
                        },
                    }
                },
                wgpu::BindingType::Texture { sample_type, view_dimension, .. } => {
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { .. }) => "texture_2d<f32>",
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => "texture_depth_2d",
                        (wgpu::TextureViewDimension::Cube, wgpu::TextureSampleType::Float { .. }) => "texture_cube<f32>",
                        _ => "texture_2d<f32>", 
                    };
                    format!("@group({}) @binding({}) var t_{}: {};", group_index, binding_index, name, type_str)
                },
                wgpu::BindingType::Sampler(type_) => {
                    let type_str = match type_ {
                        wgpu::SamplerBindingType::Comparison => "sampler_comparison",
                        _ => "sampler",
                    };
                    format!("@group({}) @binding({}) var s_{}: {};", group_index, binding_index, name, type_str)
                },
                _ => String::new(),
            };

            code.push_str(&decl);
            code.push('\n');
        }
        code
    }
}

pub fn define_texture_binding(builder: &mut ResourceBuilder, name: &'static str, texture_id: Option<Uuid>) {
    builder.add_texture(
        name,
        texture_id,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2,
        ShaderStages::FRAGMENT,
    );
    builder.add_sampler(
        name,
        texture_id,
        wgpu::SamplerBindingType::Filtering,
        ShaderStages::FRAGMENT,
    );
}