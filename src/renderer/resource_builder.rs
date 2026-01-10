use wgpu::ShaderStages;
use crate::core::buffer::BufferRef;
use crate::core::uniforms::UniformBlock;
use crate::renderer::binding::{BindingResource};
use std::sync::{Arc, RwLock};
use crate::core::texture::Texture;

type WgslStructGenerator = fn(&str) -> String;

pub struct ResourceBuilder {
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub resources: Vec<BindingResource>,
    pub names: Vec<String>,
    pub struct_generators: Vec<Option<WgslStructGenerator>>,
    // 自动维护 binding index
    pub next_binding_index: u32,
}

impl ResourceBuilder {
    pub fn new() -> Self {
        Self {
            layout_entries: Vec::new(),
            resources: Vec::new(),
            names: Vec::new(),
            struct_generators: Vec::new(),
            next_binding_index: 0,
        }
    }

    /// 添加 Uniform Buffer
    pub fn add_uniform<T: UniformBlock>(&mut self, name: &str, buffer: &BufferRef, visibility: ShaderStages) {
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
        self.struct_generators.push(Some(T::wgsl_struct_def)); 
        self.next_binding_index += 1;
    }

    pub fn add_dynamic_uniform<T: UniformBlock>(&mut self, name: &str, buffer: &BufferRef, min_binding_size: u64, visibility: ShaderStages) {
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
        self.struct_generators.push(Some(T::wgsl_struct_def));
        self.next_binding_index += 1;
    }

    /// 添加 Texture (通过 UUID)
    pub fn add_texture(&mut self, name: &str, texture: Option<Arc<RwLock<Texture>>>, sample_type: wgpu::TextureSampleType, view_dimension: wgpu::TextureViewDimension, visibility: ShaderStages) {
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

        self.resources.push(BindingResource::Texture(texture));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    /// 添加 Sampler (通过 UUID)
    pub fn add_sampler(&mut self, name: &str, texture: Option<Arc<RwLock<Texture>>>, sampler_type: wgpu::SamplerBindingType, visibility: ShaderStages) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Sampler(sampler_type),
            count: None,
        });

        self.resources.push(BindingResource::Sampler(texture));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn _add_storage_buffer(&mut self, name: &str, buffer: &BufferRef, read_only: bool, visibility: ShaderStages) {
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
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn generate_wgsl(&self, group_index: u32) -> String {
        let mut bindings_code = String::new();
        let mut struct_defs = String::new();

        for (i, entry) in self.layout_entries.iter().enumerate() {
            let name = &self.names[i];
            let binding_index = entry.binding; 

            // 如果有生成器，先生成结构体定义
            // 自动生成结构体名称：Struct_{变量名}
            // 只要变量名唯一（在同一 Group 内肯定唯一），结构体名就唯一
            let struct_type_name = if let Some(generator) = self.struct_generators[i] {
                let auto_struct_name = format!("Struct_{}", name);
                struct_defs.push_str(&generator(&auto_struct_name));
                struct_defs.push('\n');
                auto_struct_name // 返回生成的结构体名用作变量类型
            } else {
                String::new() // 非 Uniform 或者是 Storage Array
            };

            // 生成 Binding 声明
            let decl = match entry.ty {
                wgpu::BindingType::Buffer { ty, .. } => {
                    match ty {
                        wgpu::BufferBindingType::Uniform => {
                            format!("@group({}) @binding({}) var<uniform> u_{}: {};", group_index, binding_index, name, struct_type_name)
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
                            format!("@group({}) @binding({}) var<storage, {}> st_{}: {};", group_index, binding_index, access, name, type_str)
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


