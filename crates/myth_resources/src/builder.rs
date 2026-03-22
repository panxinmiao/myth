//! Resource builder for collecting GPU binding descriptions.
//!
//! [`ResourceBuilder`] accumulates [`Binding`] entries that describe both the
//! CPU-side resource data and the layout metadata needed by the renderer.
//! Each [`Binding`] combines a [`BindingResource`] with a [`BindingDesc`],
//! shader visibility, and an optional WGSL struct name.
//!
//! A single texture [`Binding`] automatically expands into two GPU binding
//! slots — one for the texture view and one for the paired sampler — so
//! callers never need to manage sampler bindings manually.

use crate::binding::BindingResource;
use crate::buffer::BufferRef;
use crate::buffer::{CpuBuffer, GpuData};
use crate::texture::TextureSource;
use crate::uniforms::WgslStruct;
use wgpu::ShaderStages;

type WgslStructGenerator = fn(&str) -> String;

/// Identifies the WGSL struct type associated with a binding.
pub enum WgslStructName {
    /// Generates the struct definition at code-gen time.
    Generator(WgslStructGenerator),
    /// A pre-existing struct name (no generation needed).
    Name(String),
}

// ============================================================================
// Binding descriptor types
// ============================================================================

/// Type-specific layout descriptor for a collected binding.
///
/// Each variant carries the metadata needed to generate
/// [`wgpu::BindGroupLayoutEntry`] entries and WGSL declarations.
pub enum BindingDesc {
    /// Uniform or storage buffer.
    Buffer {
        ty: wgpu::BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: Option<std::num::NonZeroU64>,
    },
    /// Texture with auto-paired sampler.
    ///
    /// A single `Texture` binding produces **two** GPU binding slots:
    /// one for the texture view and one for the sampler. The sampler is
    /// resolved automatically from the [`Texture`](crate::Texture) asset's
    /// [`TextureSampler`](crate::TextureSampler) configuration.
    Texture {
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        /// Sampler binding type for the auto-paired sampler entry.
        sampler_type: wgpu::SamplerBindingType,
    },
}

/// A single resource binding collected by [`ResourceBuilder`].
///
/// Combines CPU-side resource data ([`BindingResource`]) with layout
/// metadata ([`BindingDesc`]) and shader stage visibility.
pub struct Binding<'a> {
    /// The actual resource data for bind group creation.
    pub resource: BindingResource<'a>,
    /// Shader stage visibility.
    pub visibility: ShaderStages,
    /// Binding name (used for WGSL variable naming: `t_name`, `s_name`, `u_name`).
    pub name: &'a str,
    /// Optional WGSL struct type for code generation.
    pub struct_name: Option<WgslStructName>,
    /// Type-specific layout descriptor.
    pub desc: BindingDesc,
}

// ============================================================================
// ResourceBuilder
// ============================================================================

/// Collects resource bindings for automatic layout and bind group generation.
///
/// After populating the builder via `add_*` methods, use
/// [`generate_layout_entries`](Self::generate_layout_entries) to produce
/// `wgpu::BindGroupLayoutEntry` arrays and
/// [`generate_wgsl`](Self::generate_wgsl) for WGSL declarations.
pub struct ResourceBuilder<'a> {
    pub bindings: Vec<Binding<'a>>,
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
            bindings: Vec::new(),
        }
    }

    pub fn add_uniform_buffer(
        &mut self,
        name: &'a str,
        buffer: &BufferRef,
        data: Option<&'a [u8]>,
        visibility: ShaderStages,
        has_dynamic_offset: bool,
        min_binding_size: Option<std::num::NonZeroU64>,
        struct_name: Option<WgslStructName>,
    ) {
        self.bindings.push(Binding {
            resource: BindingResource::Buffer {
                buffer: buffer.clone(),
                offset: 0,
                size: min_binding_size.as_ref().map(|s| s.get()),
                data,
            },
            visibility,
            name,
            struct_name,
            desc: BindingDesc::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size,
            },
        });
    }

    pub fn add_uniform<T: WgslStruct + GpuData>(
        &mut self,
        name: &'a str,
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
        name: &'a str,
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

    /// Adds a texture binding with an auto-paired sampler.
    ///
    /// This produces **two** GPU binding slots: the texture view at the
    /// current index and the sampler at the next index. The sampler
    /// binding type is inferred from `sample_type` (depth → comparison,
    /// otherwise → filtering).
    pub fn add_texture(
        &mut self,
        name: &'a str,
        source: Option<impl Into<TextureSource>>,
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        visibility: ShaderStages,
    ) {
        let sampler_type = if matches!(sample_type, wgpu::TextureSampleType::Depth) {
            wgpu::SamplerBindingType::Comparison
        } else {
            wgpu::SamplerBindingType::Filtering
        };
        self.bindings.push(Binding {
            resource: BindingResource::Texture(source.map(std::convert::Into::into)),
            visibility,
            name,
            struct_name: None,
            desc: BindingDesc::Texture {
                sample_type,
                view_dimension,
                sampler_type,
            },
        });
    }

    pub fn add_storage_buffer(
        &mut self,
        name: &'a str,
        buffer: &BufferRef,
        data: Option<&'a [u8]>,
        read_only: bool,
        visibility: ShaderStages,
        struct_name: Option<WgslStructName>,
    ) {
        self.bindings.push(Binding {
            resource: BindingResource::Buffer {
                buffer: buffer.clone(),
                offset: 0,
                size: None,
                data,
            },
            visibility,
            name,
            struct_name,
            desc: BindingDesc::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        });
    }

    pub fn add_storage<T: WgslStruct>(
        &mut self,
        name: &'a str,
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

    /// Generates [`wgpu::BindGroupLayoutEntry`] array from collected bindings.
    ///
    /// Each buffer binding produces one layout entry. Each texture binding
    /// produces two entries (texture view + auto-paired sampler).
    #[must_use]
    pub fn generate_layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = Vec::new();
        let mut idx = 0u32;
        for b in &self.bindings {
            match &b.desc {
                BindingDesc::Buffer {
                    ty,
                    has_dynamic_offset,
                    min_binding_size,
                } => {
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: idx,
                        visibility: b.visibility,
                        ty: wgpu::BindingType::Buffer {
                            ty: *ty,
                            has_dynamic_offset: *has_dynamic_offset,
                            min_binding_size: *min_binding_size,
                        },
                        count: None,
                    });
                    idx += 1;
                }
                BindingDesc::Texture {
                    sample_type,
                    view_dimension,
                    sampler_type,
                } => {
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: idx,
                        visibility: b.visibility,
                        ty: wgpu::BindingType::Texture {
                            sample_type: *sample_type,
                            view_dimension: *view_dimension,
                            multisampled: false,
                        },
                        count: None,
                    });
                    idx += 1;
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: idx,
                        visibility: b.visibility,
                        ty: wgpu::BindingType::Sampler(*sampler_type),
                        count: None,
                    });
                    idx += 1;
                }
            }
        }
        entries
    }

    /// Generates WGSL binding declarations and struct definitions.
    ///
    /// Buffer bindings produce a single `var<uniform>` or `var<storage>`
    /// declaration. Texture bindings produce both a `var t_` texture
    /// declaration and a `var s_` sampler declaration.
    #[must_use]
    pub fn generate_wgsl(&self, group_index: u32) -> String {
        let mut bindings_code = String::new();
        let mut struct_defs = String::new();
        let mut idx = 0u32;

        for b in &self.bindings {
            let name = b.name;

            let struct_type_name = if let Some(generator) = &b.struct_name {
                let sn = match generator {
                    WgslStructName::Generator(g) => {
                        let auto = format!("Struct_{name}");
                        struct_defs.push_str(&g(&auto));
                        struct_defs.push('\n');
                        auto
                    }
                    WgslStructName::Name(n) => n.clone(),
                };
                Some(sn)
            } else {
                None
            };

            match &b.desc {
                BindingDesc::Buffer { ty, .. } => {
                    let decl = match ty {
                        wgpu::BufferBindingType::Uniform => {
                            format!(
                                "@group({group_index}) @binding({idx}) var<uniform> u_{name}: {};",
                                struct_type_name.expect("buffer binding needs a struct name")
                            )
                        }
                        wgpu::BufferBindingType::Storage { read_only } => {
                            let access = if *read_only { "read" } else { "read_write" };
                            let stn =
                                struct_type_name.expect("storage binding needs a struct name");
                            format!(
                                "@group({group_index}) @binding({idx}) var<storage, {access}> st_{name}: array<{stn}>;"
                            )
                        }
                    };
                    bindings_code.push_str(&decl);
                    bindings_code.push('\n');
                    idx += 1;
                }
                BindingDesc::Texture {
                    sample_type,
                    view_dimension,
                    sampler_type,
                } => {
                    // Texture declaration
                    let type_str = match (view_dimension, sample_type) {
                        (wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Depth) => {
                            "texture_depth_2d"
                        }
                        (wgpu::TextureViewDimension::D2Array, wgpu::TextureSampleType::Depth) => {
                            "texture_depth_2d_array"
                        }
                        (
                            wgpu::TextureViewDimension::Cube,
                            wgpu::TextureSampleType::Float { .. },
                        ) => "texture_cube<f32>",
                        (
                            wgpu::TextureViewDimension::D2Array,
                            wgpu::TextureSampleType::Float { .. },
                        ) => "texture_2d_array<f32>",
                        _ => "texture_2d<f32>",
                    };
                    bindings_code.push_str(&format!(
                        "@group({group_index}) @binding({idx}) var t_{name}: {type_str};\n"
                    ));
                    idx += 1;

                    // Sampler declaration (auto-paired)
                    let sampler_type_str = match sampler_type {
                        wgpu::SamplerBindingType::Comparison => "sampler_comparison",
                        _ => "sampler",
                    };
                    bindings_code.push_str(&format!(
                        "@group({group_index}) @binding({idx}) var s_{name}: {sampler_type_str};\n"
                    ));
                    idx += 1;
                }
            }
        }
        format!(
            "// --- Auto Generated Bindings (Group {group_index}) ---\n{struct_defs}\n{bindings_code}\n"
        )
    }
}
