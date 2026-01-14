use wgpu::ShaderStages;
use crate::resources::buffer::BufferRef;
use crate::resources::uniforms::WgslStruct;
use crate::render::resources::binding::BindingResource;
use crate::assets::TextureHandle;
use std::sync::atomic::{AtomicU64, Ordering};

type WgslStructGenerator = fn(&str) -> String;

// 临时 uniform 数据的 ID 生成器
static NEXT_TEMP_UNIFORM_ID: AtomicU64 = AtomicU64::new(1 << 61);

pub enum WgslStructName{
    Generator(WgslStructGenerator),
    Name(String),
}

pub struct ResourceBuilder<'a> {
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub resources: Vec<BindingResource<'a>>,
    pub names: Vec<String>,
    pub struct_generators: Vec<Option<WgslStructName>>,
    // 自动维护 binding index
    next_binding_index: u32,
}

impl<'a> ResourceBuilder<'a> {
    pub fn new() -> Self {
        Self {
            layout_entries: Vec::new(),
            resources: Vec::new(),
            names: Vec::new(),
            struct_generators: Vec::new(),
            next_binding_index: 0,
        }
    }

    pub fn add_uniform_with_struct<T: WgslStruct>(&mut self, name: &str, buffer: &BufferRef, visibility: ShaderStages) {
        self.add_uniform_buffer(name, buffer, visibility, Some(WgslStructName::Generator(T::wgsl_struct_def)));
    }

    pub fn add_uniform_buffer(&mut self, name: &str, buffer: &BufferRef, visibility: ShaderStages, struct_name: Option<WgslStructName>) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(),
            offset: 0,
            size: None,
        });

        self.names.push(name.to_string());
        self.struct_generators.push(struct_name); 
        self.next_binding_index += 1;
    }
    
    /// 【新】添加 Uniform 数据（直接使用数据引用，带struct生成器）
    pub fn add_uniform_with_generator<T: WgslStruct>(
        &mut self, 
        name: &str, 
        data: &'a [u8],
        visibility: ShaderStages
    ) {
        let temp_id = NEXT_TEMP_UNIFORM_ID.fetch_add(1, Ordering::Relaxed);
        
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self.resources.push(BindingResource::UniformSlot {
            slot_id: temp_id,
            data,
            label: "", // label不重要，只是为了满足类型
        });

        self.names.push(name.to_string());
        self.struct_generators.push(Some(WgslStructName::Generator(T::wgsl_struct_def)));
        self.next_binding_index += 1;
    }
    
    /// 【新】添加 Uniform 数据（直接使用数据引用）
    pub fn add_uniform(&mut self, name: &str, data: &'a [u8], label: &'a str, visibility: ShaderStages) {
        let temp_id = NEXT_TEMP_UNIFORM_ID.fetch_add(1, Ordering::Relaxed);
        
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self.resources.push(BindingResource::UniformSlot {
            slot_id: temp_id,
            data,
            label,
        });

        self.names.push(name.to_string());
        self.struct_generators.push(Some(WgslStructName::Name(label.to_string())));
        self.next_binding_index += 1;
    }

     /// 添加 Dynamic Uniform Buffer

    pub fn add_dynamic_uniform<T: WgslStruct>(&mut self, name: &str, buffer: &BufferRef, min_binding_size: u64, visibility: ShaderStages) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(min_binding_size),
            },
            count: None,
        });

        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(),
            offset: 0,
            size: Some(min_binding_size),
        });

        self.names.push(name.to_string());
        self.struct_generators.push(Some(WgslStructName::Generator(T::wgsl_struct_def)));
        self.next_binding_index += 1;
    }

    pub fn add_texture(&mut self, name: &str, texture: TextureHandle, sample_type: wgpu::TextureSampleType, view_dimension: wgpu::TextureViewDimension, visibility: ShaderStages) {
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

        self.resources.push(BindingResource::Texture(Some(texture)));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn add_sampler(&mut self, name: &str, texture: TextureHandle, sampler_type: wgpu::SamplerBindingType, visibility: ShaderStages) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Sampler(sampler_type),
            count: None,
        });

        self.resources.push(BindingResource::Sampler(Some(texture)));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn add_storage(&mut self, name: &str, buffer: &BufferRef, read_only: bool, visibility: ShaderStages, struct_name: Option<WgslStructName>) {
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

        self.names.push(name.to_string());
        self.struct_generators.push(struct_name);
        self.next_binding_index += 1;
    }

     /// 添加 Storage Buffer

    pub fn add_storage_with_struct<T: WgslStruct>(&mut self, name: &str, buffer: &BufferRef, read_only: bool, visibility: ShaderStages) {
        self.add_storage(name, buffer, read_only, visibility, Some(WgslStructName::Generator(T::wgsl_struct_def)));
    }

    pub fn generate_wgsl(&self, group_index: u32) -> String {
        let mut bindings_code = String::new();
        let mut struct_defs = String::new();

        for (i, entry) in self.layout_entries.iter().enumerate() {
            let name = &self.names[i];
            let binding_index = entry.binding; 

            // 如果有生成器，先生成结构体定义
            // 自动生成结构体名称：Struct_{变量名}
            let struct_type_name = if let Some(generator) = &self.struct_generators[i] {
                let stuct_name = match generator {
                    WgslStructName::Generator(generator) => {
                        let auto_struct_name = format!("Struct_{}", name);
                        struct_defs.push_str(&generator(&auto_struct_name));
                        struct_defs.push('\n');
                        auto_struct_name
                    },
                    WgslStructName::Name(name_str) => {
                        name_str.clone()
                    },
                };  

                Some(stuct_name)
            } else {
                None
            };

            // 生成 Binding 声明
            let decl = match entry.ty {
                wgpu::BindingType::Buffer { ty, .. } => {
                    match ty {
                        wgpu::BufferBindingType::Uniform => {
                            format!("@group({}) @binding({}) var<uniform> u_{}: {};", group_index, binding_index, name, struct_type_name.expect("need a struct name"))
                        },
                        wgpu::BufferBindingType::Storage { read_only } => {
                            let access = if read_only { "read" } else { "read_write" };

                            let struct_type_name = struct_type_name.expect("need a struct name");
                            format!("@group({}) @binding({}) var<storage, {}> st_{}: {};", group_index, binding_index, access, name,  format!("array<{}>", struct_type_name))
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

            bindings_code.push_str(&decl);
            bindings_code.push('\n');
        }
        format!("// --- Auto Generated Bindings (Group {}) ---\n{}\n{}\n", group_index, struct_defs, bindings_code)
    }
}


