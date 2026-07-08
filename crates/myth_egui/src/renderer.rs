use std::{
    borrow::Cow,
    num::NonZeroU64,
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};

use bytemuck::Zeroable as _;
use egui::epaint::{self, Primitive, Vertex};
use rustc_hash::FxHashMap;
use wgpu::util::DeviceExt as _;

static CALLBACK_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

/// Information about the screen used for rendering.
pub struct ScreenDescriptor {
    /// Size of the target in physical pixels.
    pub size_in_pixels: [u32; 2],
    /// High-DPI scale factor: physical pixels per egui point.
    pub pixels_per_point: f32,
}

impl ScreenDescriptor {
    fn screen_size_in_points(&self) -> [f32; 2] {
        [
            self.size_in_pixels[0] as f32 / self.pixels_per_point,
            self.size_in_pixels[1] as f32 / self.pixels_per_point,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UniformBuffer {
    screen_size_in_points: [f32; 2],
    dithering: u32,
    predictable_texture_filtering: u32,
}

struct SlicedBuffer {
    buffer: wgpu::Buffer,
    slices: Vec<Range<usize>>,
    capacity: wgpu::BufferAddress,
}

/// Texture record owned by the Myth egui renderer.
pub struct Texture {
    pub texture: Option<wgpu::Texture>,
    pub bind_group: wgpu::BindGroup,
    pub options: Option<epaint::textures::TextureOptions>,
}

/// Renderer configuration.
#[derive(Clone, Copy, Debug)]
pub struct RendererOptions {
    pub msaa_samples: u32,
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    pub dithering: bool,
    pub predictable_texture_filtering: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            msaa_samples: 1,
            depth_stencil_format: None,
            dithering: true,
            predictable_texture_filtering: false,
        }
    }
}

/// A small egui renderer implemented directly on Myth's wgpu/RDG path.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    index_buffer: SlicedBuffer,
    vertex_buffer: SlicedBuffer,
    uniform_buffer: wgpu::Buffer,
    previous_uniform_buffer_content: UniformBuffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    placeholder_bind_group: wgpu::BindGroup,
    textures: FxHashMap<epaint::TextureId, Texture>,
    next_user_texture_id: u64,
    samplers: FxHashMap<epaint::textures::TextureOptions, wgpu::Sampler>,
    options: RendererOptions,
}

impl Renderer {
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        output_color_format: wgpu::TextureFormat,
        options: RendererOptions,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("myth_egui_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("egui.wgsl"))),
        });

        let initial_uniform = UniformBuffer {
            screen_size_in_points: [0.0, 0.0],
            dithering: u32::from(options.dithering),
            predictable_texture_filtering: u32::from(options.predictable_texture_filtering),
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("myth_egui_uniform_buffer"),
            contents: bytemuck::cast_slice(&[initial_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("myth_egui_uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<UniformBuffer>() as _,
                        ),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("myth_egui_uniform_bg"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("myth_egui_texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("myth_egui_pipeline_layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let depth_stencil = options
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

        let fragment_entry = if output_color_format.is_srgb() {
            log::warn!(
                "Detected an sRGB-aware framebuffer {output_color_format:?}. egui prefers gamma-space targets such as Rgba8Unorm or Bgra8Unorm"
            );
            "fs_main_linear_framebuffer"
        } else {
            "fs_main_gamma_framebuffer"
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("myth_egui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: Some("vs_main"),
                module: &module,
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 5 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Uint32
                    ],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::default(),
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::default(),
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: options.msaa_samples.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        const VERTEX_BUFFER_START_CAPACITY: wgpu::BufferAddress =
            (std::mem::size_of::<Vertex>() * 1024) as _;
        const INDEX_BUFFER_START_CAPACITY: wgpu::BufferAddress =
            (std::mem::size_of::<u32>() * 1024 * 3) as _;

        let placeholder_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("myth_egui_placeholder_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        let placeholder_view =
            placeholder_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("myth_egui_placeholder_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let placeholder_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("myth_egui_placeholder_bg"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&placeholder_sampler),
                },
            ],
        });

        Self {
            pipeline,
            vertex_buffer: SlicedBuffer {
                buffer: create_vertex_buffer(device, VERTEX_BUFFER_START_CAPACITY),
                slices: Vec::with_capacity(64),
                capacity: VERTEX_BUFFER_START_CAPACITY,
            },
            index_buffer: SlicedBuffer {
                buffer: create_index_buffer(device, INDEX_BUFFER_START_CAPACITY),
                slices: Vec::with_capacity(64),
                capacity: INDEX_BUFFER_START_CAPACITY,
            },
            uniform_buffer,
            previous_uniform_buffer_content: UniformBuffer::zeroed(),
            uniform_bind_group,
            texture_bind_group_layout,
            placeholder_bind_group,
            textures: FxHashMap::default(),
            next_user_texture_id: 0,
            samplers: FxHashMap::default(),
            options,
        }
    }

    pub fn render(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        let size_in_pixels = screen_descriptor.size_in_pixels;
        if size_in_pixels.contains(&0) {
            return;
        }

        let pixels_per_point = screen_descriptor.pixels_per_point;
        let mut index_buffer_slices = self.index_buffer.slices.iter();
        let mut vertex_buffer_slices = self.vertex_buffer.slices.iter();

        render_pass.set_viewport(
            0.0,
            0.0,
            size_in_pixels[0] as f32,
            size_in_pixels[1] as f32,
            0.0,
            1.0,
        );
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

        for epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in paint_jobs
        {
            let rect = ScissorRect::new(clip_rect, pixels_per_point, size_in_pixels);
            if rect.width == 0 || rect.height == 0 {
                if let Primitive::Mesh(_) = primitive {
                    index_buffer_slices
                        .next()
                        .expect("Call update_buffers before render");
                    vertex_buffer_slices
                        .next()
                        .expect("Call update_buffers before render");
                }
                continue;
            }

            render_pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);

            match primitive {
                Primitive::Mesh(mesh) => {
                    let index_buffer_slice = index_buffer_slices
                        .next()
                        .expect("Call update_buffers before render");
                    let vertex_buffer_slice = vertex_buffer_slices
                        .next()
                        .expect("Call update_buffers before render");

                    if let Some(Texture { bind_group, .. }) = self.textures.get(&mesh.texture_id) {
                        render_pass.set_bind_group(1, bind_group, &[]);
                        render_pass.set_index_buffer(
                            self.index_buffer.buffer.slice(
                                index_buffer_slice.start as u64..index_buffer_slice.end as u64,
                            ),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.set_vertex_buffer(
                            0,
                            self.vertex_buffer.buffer.slice(
                                vertex_buffer_slice.start as u64..vertex_buffer_slice.end as u64,
                            ),
                        );
                        render_pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
                    } else {
                        log::warn!("Missing egui texture: {:?}", mesh.texture_id);
                    }
                }
                Primitive::Callback(_) => {
                    if !CALLBACK_WARNING_EMITTED.swap(true, Ordering::Relaxed) {
                        log::warn!("egui paint callbacks are not implemented by myth_egui yet");
                    }
                }
            }
        }

        render_pass.set_scissor_rect(0, 0, size_in_pixels[0], size_in_pixels[1]);
    }

    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: epaint::TextureId,
        image_delta: &epaint::ImageDelta,
    ) {
        let width = image_delta.image.width() as u32;
        let height = image_delta.image.height() as u32;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let data_color32 = match &image_delta.image {
            epaint::ImageData::Color(image) => {
                assert_eq!(
                    width as usize * height as usize,
                    image.pixels.len(),
                    "Mismatch between egui texture size and texel count"
                );
                Cow::Borrowed(&image.pixels)
            }
        };
        let data_bytes: &[u8] = bytemuck::cast_slice(data_color32.as_slice());

        let label_string = format!("myth_egui_texid_{id:?}");
        let label = Some(label_string.as_str());

        let (texture, origin, bind_group) = if let Some(pos) = image_delta.pos {
            let Texture {
                texture,
                bind_group,
                options,
            } = self
                .textures
                .remove(&id)
                .expect("Tried to update an egui texture that has not been allocated");
            let texture = texture.expect("Tried to update a user-provided egui texture");
            let options = options.expect("Tried to update a user-provided egui texture");
            let origin = wgpu::Origin3d {
                x: pos[0] as u32,
                y: pos[1] as u32,
                z: 0,
            };

            let bind_group = (image_delta.options == options).then_some(bind_group);
            (texture, origin, bind_group)
        } else {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            });
            (texture, wgpu::Origin3d::ZERO, None)
        };

        let bind_group = bind_group.unwrap_or_else(|| {
            let sampler = self
                .samplers
                .entry(image_delta.options)
                .or_insert_with(|| create_sampler(image_delta.options, device));
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            data_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        self.textures.insert(
            id,
            Texture {
                texture: Some(texture),
                bind_group,
                options: Some(image_delta.options),
            },
        );
    }

    pub fn free_texture(&mut self, id: &epaint::TextureId) {
        if let Some(texture) = self.textures.remove(id).and_then(|t| t.texture) {
            texture.destroy();
        }
    }

    #[must_use]
    pub fn texture(&self, id: &epaint::TextureId) -> Option<&Texture> {
        self.textures.get(id)
    }

    pub fn register_external_texture(
        &mut self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> epaint::TextureId {
        let id = self.reserve_external_texture();
        self.update_external_texture(device, id, view, sampler);
        id
    }

    pub fn reserve_external_texture(&mut self) -> epaint::TextureId {
        let id = epaint::TextureId::User(self.next_user_texture_id);
        self.next_user_texture_id += 1;
        self.use_placeholder_texture(id);
        id
    }

    pub fn use_placeholder_texture(&mut self, id: epaint::TextureId) {
        self.insert_texture(
            id,
            Texture {
                texture: None,
                bind_group: self.placeholder_bind_group.clone(),
                options: None,
            },
        );
    }

    pub fn update_external_texture(
        &mut self,
        device: &wgpu::Device,
        id: epaint::TextureId,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let label_string = format!("myth_egui_external_{id:?}");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label_string.as_str()),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        self.insert_texture(
            id,
            Texture {
                texture: None,
                bind_group,
                options: None,
            },
        );
    }

    fn insert_texture(&mut self, id: epaint::TextureId, texture: Texture) {
        if let Some(old) = self.textures.insert(id, texture)
            && let Some(texture) = old.texture
        {
            texture.destroy();
        }
    }

    pub fn register_native_texture(
        &mut self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
    ) -> epaint::TextureId {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("myth_egui_native_texture_sampler"),
            mag_filter: texture_filter,
            min_filter: texture_filter,
            compare: None,
            ..Default::default()
        });
        self.register_external_texture(device, view, &sampler)
    }

    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        let uniform_buffer_content = UniformBuffer {
            screen_size_in_points: screen_descriptor.screen_size_in_points(),
            dithering: u32::from(self.options.dithering),
            predictable_texture_filtering: u32::from(self.options.predictable_texture_filtering),
        };
        if uniform_buffer_content != self.previous_uniform_buffer_content {
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform_buffer_content]),
            );
            self.previous_uniform_buffer_content = uniform_buffer_content;
        }

        let (vertex_count, index_count) =
            paint_jobs.iter().fold((0, 0), |acc, clipped_primitive| {
                match &clipped_primitive.primitive {
                    Primitive::Mesh(mesh) => {
                        (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len())
                    }
                    Primitive::Callback(_) => acc,
                }
            });

        self.index_buffer.slices.clear();
        if index_count > 0 {
            let required_index_buffer_size = (std::mem::size_of::<u32>() * index_count) as u64;
            if self.index_buffer.capacity < required_index_buffer_size {
                self.index_buffer.capacity =
                    (self.index_buffer.capacity * 2).max(required_index_buffer_size);
                self.index_buffer.buffer = create_index_buffer(device, self.index_buffer.capacity);
            }

            let Some(mut staging) = queue.write_buffer_with(
                &self.index_buffer.buffer,
                0,
                NonZeroU64::new(required_index_buffer_size).expect("non-zero index buffer size"),
            ) else {
                panic!(
                    "Failed to create staging buffer for egui index data. Required: {required_index_buffer_size}; capacity: {}",
                    self.index_buffer.capacity
                );
            };

            let mut index_offset = 0;
            for epaint::ClippedPrimitive { primitive, .. } in paint_jobs {
                if let Primitive::Mesh(mesh) = primitive {
                    let size = mesh.indices.len() * std::mem::size_of::<u32>();
                    let slice = index_offset..(size + index_offset);
                    staging
                        .slice(slice.clone())
                        .copy_from_slice(bytemuck::cast_slice(&mesh.indices));
                    self.index_buffer.slices.push(slice);
                    index_offset += size;
                }
            }
        }

        self.vertex_buffer.slices.clear();
        if vertex_count > 0 {
            let required_vertex_buffer_size = (std::mem::size_of::<Vertex>() * vertex_count) as u64;
            if self.vertex_buffer.capacity < required_vertex_buffer_size {
                self.vertex_buffer.capacity =
                    (self.vertex_buffer.capacity * 2).max(required_vertex_buffer_size);
                self.vertex_buffer.buffer =
                    create_vertex_buffer(device, self.vertex_buffer.capacity);
            }

            let Some(mut staging) = queue.write_buffer_with(
                &self.vertex_buffer.buffer,
                0,
                NonZeroU64::new(required_vertex_buffer_size).expect("non-zero vertex buffer size"),
            ) else {
                panic!(
                    "Failed to create staging buffer for egui vertex data. Required: {required_vertex_buffer_size}; capacity: {}",
                    self.vertex_buffer.capacity
                );
            };

            let mut vertex_offset = 0;
            for epaint::ClippedPrimitive { primitive, .. } in paint_jobs {
                if let Primitive::Mesh(mesh) = primitive {
                    let size = mesh.vertices.len() * std::mem::size_of::<Vertex>();
                    let slice = vertex_offset..(size + vertex_offset);
                    staging
                        .slice(slice.clone())
                        .copy_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    self.vertex_buffer.slices.push(slice);
                    vertex_offset += size;
                }
            }
        }
    }
}

fn create_sampler(
    options: epaint::textures::TextureOptions,
    device: &wgpu::Device,
) -> wgpu::Sampler {
    let mag_filter = match options.magnification {
        epaint::textures::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        epaint::textures::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let min_filter = match options.minification {
        epaint::textures::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        epaint::textures::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode = match options.wrap_mode {
        epaint::textures::TextureWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        epaint::textures::TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        epaint::textures::TextureWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };

    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("myth_egui_sampler"),
        mag_filter,
        min_filter,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..Default::default()
    })
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("myth_egui_vertex_buffer"),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        size,
        mapped_at_creation: false,
    })
}

fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("myth_egui_index_buffer"),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        size,
        mapped_at_creation: false,
    })
}

struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl ScissorRect {
    fn new(clip_rect: &epaint::Rect, pixels_per_point: f32, target_size: [u32; 2]) -> Self {
        let clip_min_x = (pixels_per_point * clip_rect.min.x).round() as i32;
        let clip_min_y = (pixels_per_point * clip_rect.min.y).round() as i32;
        let clip_max_x = (pixels_per_point * clip_rect.max.x).round() as i32;
        let clip_max_y = (pixels_per_point * clip_rect.max.y).round() as i32;

        let max_x = target_size[0] as i32;
        let max_y = target_size[1] as i32;
        let clip_min_x = clip_min_x.clamp(0, max_x) as u32;
        let clip_min_y = clip_min_y.clamp(0, max_y) as u32;
        let clip_max_x = clip_max_x.clamp(clip_min_x as i32, max_x) as u32;
        let clip_max_y = clip_max_y.clamp(clip_min_y as i32, max_y) as u32;

        Self {
            x: clip_min_x,
            y: clip_min_y,
            width: clip_max_x - clip_min_x,
            height: clip_max_y - clip_min_y,
        }
    }
}
