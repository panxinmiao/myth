//! 资源构建器
//!
//! 用于构建 `BindGroup` Layout 和收集绑定资源

use crate::renderer::core::binding::BindingResource;
use crate::resources::buffer::BufferRef;
use crate::resources::buffer::{CpuBuffer, GpuData};
use crate::resources::texture::{SamplerSource, TextureSource};
use crate::resources::uniforms::WgslStruct;
use wgpu::ShaderStages;

type WgslStructGenerator = fn(&str) -> String;

pub enum WgslStructName {
    Generator(WgslStructGenerator),
    Name(String),
}

pub struct ResourceBuilder<'a> {
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub resources: Vec<BindingResource<'a>>,
    pub names: Vec<String>,
    pub struct_generators: Vec<Option<WgslStructName>>,
    next_binding_index: u32,
}

impl Default for ResourceBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ResourceBuilder<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout_entries: Vec::new(),
            resources: Vec::new(),
            names: Vec::new(),
            struct_generators: Vec::new(),
            next_binding_index: 0,
        }
    }

    pub fn add_uniform_buffer(
        &mut self,
        name: &str,
        buffer: &BufferRef,
        data: Option<&'a [u8]>,
        visibility: ShaderStages,
        has_dynamic_offset: bool,
        min_binding_size: Option<std::num::NonZeroU64>,
        struct_name: Option<WgslStructName>,
    ) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size,
            },
            count: None,
        });

        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(),
            offset: 0,
            size: min_binding_size.as_ref().map(|s| s.get()),
            data,
        });

        self.names.push(name.to_string());
        self.struct_generators.push(struct_name);
        self.next_binding_index += 1;
    }

    pub fn add_uniform<T: WgslStruct + GpuData>(
        &mut self,
        name: &str,
        cpu_buffer: &'a CpuBuffer<T>,
        visibility: ShaderStages,
    ) {
        self.add_uniform_buffer(
            name,
            &cpu_buffer.handle(),
            None,
            visibility,
            false,
            None,
            Some(WgslStructName::Generator(T::wgsl_struct_def)),
        );
    }

    pub fn add_dynamic_uniform<T: WgslStruct>(
        &mut self,
        name: &str,
        buffer_ref: &BufferRef,
        data: Option<&'a [u8]>,
        min_binding_size: std::num::NonZeroU64,
        visibility: ShaderStages,
    ) {
        self.add_uniform_buffer(
            name,
            buffer_ref,
            data,
            visibility,
            true,
            Some(min_binding_size),
            Some(WgslStructName::Generator(T::wgsl_struct_def)),
        );
    }

    fn add_texture_internal(
        &mut self,
        name: &str,
        source: Option<TextureSource>,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        visibility: ShaderStages,
    ) {
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

        self.resources.push(BindingResource::Texture(source));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn add_texture(
        &mut self,
        name: &str,
        source: Option<impl Into<TextureSource>>,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        visibility: ShaderStages,
    ) {
        self.add_texture_internal(
            name,
            source.map(std::convert::Into::into),
            sample_type,
            view_dimension,
            visibility,
        );
    }

    pub fn add_sampler(
        &mut self,
        name: &str,
        source: Option<impl Into<SamplerSource>>,
        sampler_type: wgpu::SamplerBindingType,
        visibility: ShaderStages,
    ) {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.next_binding_index,
            visibility,
            ty: wgpu::BindingType::Sampler(sampler_type),
            count: None,
        });

        self.resources.push(BindingResource::Sampler(
            source.map(std::convert::Into::into),
        ));
        self.names.push(name.to_string());
        self.struct_generators.push(None);
        self.next_binding_index += 1;
    }

    pub fn add_storage_buffer(
        &mut self,
        name: &str,
        buffer: &BufferRef,
        data: Option<&'a [u8]>,
        read_only: bool,
        visibility: ShaderStages,
        struct_name: Option<WgslStructName>,
    ) {
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
                min_binding_size: None,
            },
            count: None,
        });

        self.resources.push(BindingResource::Buffer {
            buffer: buffer.clone(),
            offset: 0,
            size: None,
            data,
        });

        self.names.push(name.to_string());
        self.struct_generators.push(struct_name);
        self.next_binding_index += 1;
    }

    pub fn add_storage<T: WgslStruct>(
        &mut self,
        name: &str,
        buffer: &BufferRef,
        data: Option<&'a [u8]>,
        read_only: bool,
        visibility: ShaderStages,
    ) {
        self.add_storage_buffer(
            name,
            buffer,
            data,
            read_only,
            visibility,
            Some(WgslStructName::Generator(T::wgsl_struct_def)),
        );
    }

    #[must_use]
    pub fn generate_wgsl(&self, group_index: u32) -> String {
        let mut bindings_code = String::new();
        let mut struct_defs = String::new();

        for (i, entry) in self.layout_entries.iter().enumerate() {
            let name = &self.names[i];
            let binding_index = entry.binding;

            let struct_type_name = if let Some(generator) = &self.struct_generators[i] {
                let stuct_name = match generator {
                    WgslStructName::Generator(generator) => {
                        let auto_struct_name = format!("Struct_{name}");
                        struct_defs.push_str(&generator(&auto_struct_name));
                        struct_defs.push('\n');
                        auto_struct_name
                    }
                    WgslStructName::Name(name_str) => name_str.clone(),
                };
                Some(stuct_name)
            } else {
                None
            };

            let decl = match entry.ty {
                wgpu::BindingType::Buffer { ty, .. } => match ty {
                    wgpu::BufferBindingType::Uniform => {
                        format!(
                            "@group({}) @binding({}) var<uniform> u_{}: {};",
                            group_index,
                            binding_index,
                            name,
                            struct_type_name.expect("need a struct name")
                        )
                    }
                    wgpu::BufferBindingType::Storage { read_only } => {
                        let access = if read_only { "read" } else { "read_write" };
                        let struct_type_name = struct_type_name.expect("need a struct name");
                        format!(
                            "@group({}) @binding({}) var<storage, {}> st_{}: {};",
                            group_index,
                            binding_index,
                            access,
                            name,
                            format!("array<{}>", struct_type_name)
                        )
                    }
                },
                wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension,
                    ..
                } => {
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { .. }) => {
                            "texture_2d<f32>"
                        }
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => {
                            "texture_depth_2d"
                        }
                        (
                            wgpu::TextureViewDimension::Cube,
                            wgpu::TextureSampleType::Float { .. },
                        ) => "texture_cube<f32>",
                        _ => "texture_2d<f32>",
                    };
                    format!(
                        "@group({group_index}) @binding({binding_index}) var t_{name}: {type_str};"
                    )
                }
                wgpu::BindingType::Sampler(type_) => {
                    let type_str = match type_ {
                        wgpu::SamplerBindingType::Comparison => "sampler_comparison",
                        _ => "sampler",
                    };
                    format!(
                        "@group({group_index}) @binding({binding_index}) var s_{name}: {type_str};"
                    )
                }
                _ => String::new(),
            };

            bindings_code.push_str(&decl);
            bindings_code.push('\n');
        }
        format!(
            "// --- Auto Generated Bindings (Group {group_index}) ---\n{struct_defs}\n{bindings_code}\n"
        )
    }
}
