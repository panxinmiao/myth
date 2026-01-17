//! GPU 资源管理器
//!
//! 负责 GPU 端资源的创建、更新和管理

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::vec;
use std::borrow::Cow;

use slotmap::SecondaryMap;
use core::ops::Range;
use rustc_hash::FxHashMap;

use crate::resources::geometry::Geometry;
use crate::resources::texture::{Texture, TextureSampler};
use crate::scene::environment::Environment;
use crate::resources::buffer::BufferRef;
use crate::resources::image::{Image, ImageInner};
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};

use crate::renderer::core::binding::{BindingResource, Bindings};
use crate::renderer::core::builder::ResourceBuilder;
use crate::renderer::pipeline::vertex::GeneratedVertexLayout;
use crate::renderer::graph::RenderState;

static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(0);

pub fn generate_resource_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// GPU 资源包装器
// ============================================================================

pub struct GpuBuffer {
    pub id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub label: String,
    pub last_used_frame: u64,
    pub version: u64,
    pub last_uploaded_version: u64,
    shadow_data: Option<Vec<u8>>,
}

impl GpuBuffer {
    pub fn new(device: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });
        let id = generate_resource_id();

        Self {
            id,
            buffer,
            size: data.len() as u64,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
            shadow_data: None,
        }
    }

    pub fn enable_shadow_copy(&mut self) {
        if self.shadow_data.is_none() {
            self.shadow_data = Some(Vec::new());
        }
    }

    pub fn update_with_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        if let Some(prev) = &mut self.shadow_data {
            if prev == data {
                return false;
            }
            if prev.len() != data.len() {
                *prev = vec![0u8; data.len()];
            }
            prev.copy_from_slice(data);
        }
        self.write_to_gpu(device, queue, data)
    }

    pub fn update_with_version(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], new_version: u64) -> bool {
        if new_version <= self.version {
            return false;
        }
        self.version = new_version;
        self.write_to_gpu(device, queue, data)
    }

    fn write_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let new_size = data.len() as u64;
        if new_size > self.size {
            self.resize(device, new_size);
            queue.write_buffer(&self.buffer, 0, data);
            return true;
        }
        queue.write_buffer(&self.buffer, 0, data);
        false
    }

    fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        self.buffer.destroy();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.size = new_size;
        self.id = generate_resource_id();
    }
}

pub struct GpuTexture {
    pub view: wgpu::TextureView,
    pub image_id: u64,
    pub image_generation_id: u64,
    pub version: u64,
    pub image_data_version: u64,
    pub last_used_frame: u64,
}

impl GpuTexture {
    pub fn new(texture: &Texture, gpu_image: &GpuImage) -> Self {
        let view = gpu_image.texture.create_view(&wgpu::TextureViewDescriptor {
            label: texture.name(),
            format: Some(gpu_image.texture.format()),
            dimension: Some(texture.view_dimension),
            ..Default::default()
        });

        Self {
            view,
            image_id: gpu_image.id,
            image_generation_id: gpu_image.generation_id,
            version: texture.version(),
            image_data_version: gpu_image.version,
            last_used_frame: 0,
        }
    }
}

pub struct GpuImage {
    pub texture: wgpu::Texture,
    pub id: u64,
    pub version: u64,
    pub generation_id: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,
    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

impl GpuImage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner, mip_level_count: u32, usage: wgpu::TextureUsages) -> Self {
        use std::sync::atomic::Ordering;
        
        let width = image.width.load(Ordering::Relaxed);
        let height = image.height.load(Ordering::Relaxed);
        let depth = image.depth.load(Ordering::Relaxed);
        let desc = image.description.read().expect("Failed to read image descriptor");
        
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: image.label(),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: desc.dimension,
            format: desc.format,
            usage,
            view_formats: &[],
        });

        Self::upload_data(queue, &texture, image, width, height, depth, desc.format);

        let mipmaps_generated = mip_level_count <= 1;
        Self {
            texture,
            id: image.id,
            version: image.version.load(Ordering::Relaxed),
            generation_id: image.generation_id.load(Ordering::Relaxed),
            width,
            height,
            depth,
            format: desc.format,
            mip_level_count,
            usage,
            mipmaps_generated,
            last_used_frame: 0,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner) {
        use std::sync::atomic::Ordering;
        
        let gen_id = image.generation_id.load(Ordering::Relaxed);
        if self.generation_id != gen_id {
            *self = Self::new(device, queue, image, self.mip_level_count, self.usage);
            return;
        }

        let ver = image.version.load(Ordering::Relaxed);
        if self.version < ver {
            Self::upload_data(queue, &self.texture, image, self.width, self.height, self.depth, self.format);
            self.version = ver;
            if self.mip_level_count > 1 {
                self.mipmaps_generated = false;
            }
        }
    }

    fn upload_data(queue: &wgpu::Queue, texture: &wgpu::Texture, image: &ImageInner, src_width: u32, src_height: u32, src_depth: u32, src_format: wgpu::TextureFormat) {
        let data_guard = image.data.read().expect("Failed to read image data");
        if let Some(data) = &*data_guard {
            let block_size = src_format.block_copy_size(None).unwrap_or(4);
            let bytes_per_row = src_width * block_size;
            
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(src_height),
                },
                wgpu::Extent3d {
                    width: src_width,
                    height: src_height,
                    depth_or_array_layers: src_depth,
                }
            );
        }
    }
}

pub struct GpuGeometry {
    pub layout_info: Arc<GeneratedVertexLayout>,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,
    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>,
    pub version: u64,
    pub last_data_version: u64,
    pub last_used_frame: u64,
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    pub uniform_buffers: Vec<u64>,
    pub last_data_version: u64,
    pub last_binding_version: u64,
    pub last_layout_version: u64,
    pub last_used_frame: u64,
}

pub struct GpuEnvironment {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    pub last_uniform_version: u64,
    pub last_binding_version: u64,
    pub last_layout_version: u64,
    pub last_render_state_version: u64,
    pub last_used_frame: u64,
}

// ============================================================================
// Mipmap Generator
// ============================================================================

const BLIT_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var output : VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

@group(0) @binding(0) var t_diffuse : texture_2d<f32>;
@group(0) @binding(1) var s_diffuse : sampler;

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
"#;

pub struct MipmapGenerator {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    shader: wgpu::ShaderModule,
    pipelines: FxHashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT_WGSL)),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            layout,
            sampler,
            shader,
            pipelines: FxHashMap::default(),
        }
    }

    fn get_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Mipmap Pipeline {:?}", format)),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Mipmap Pipeline Layout"),
                    bind_group_layouts: &[&self.layout],
                    immediate_size: 0,
                })),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        }).clone()
    }

    pub fn generate(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, mip_count: u32) {
        if mip_count < 2 { return; }

        let format = texture.format();
        let pipeline = self.get_pipeline(device, format);
        let layer_count = texture.depth_or_array_layers();

        for layer in 0..layer_count {
            for i in 0..mip_count - 1 {
                let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Mipmap Src"),
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                });

                let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Mipmap Dst"),
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: i + 1,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Mipmap BG"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });

                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mipmap Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dst_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass.draw(0..3, 0..1);
            }
        }
    }
}

// ============================================================================
// Resource Manager
// ============================================================================

pub struct ResourceManager {
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame_index: u64,

    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    gpu_textures: SecondaryMap<TextureHandle, GpuTexture>,
    gpu_samplers: SecondaryMap<TextureHandle, wgpu::Sampler>,

    worlds: FxHashMap<u64, GpuEnvironment>,
    gpu_buffers: FxHashMap<u64, GpuBuffer>,
    gpu_images: FxHashMap<u64, GpuImage>,

    sampler_cache: FxHashMap<TextureSampler, wgpu::Sampler>,
    layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    dummy_texture: GpuTexture,
    dummy_sampler: wgpu::Sampler,
    mipmap_generator: MipmapGenerator,
}

impl ResourceManager {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let dummy_tex = Texture::new_2d(Some("dummy"), 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm);
        let dummy_gpu_image = GpuImage::new(&device, &queue, &dummy_tex.image, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let dummy_gpu_tex = GpuTexture::new(&dummy_tex, &dummy_gpu_image);

        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Dummy Sampler"),
            ..Default::default()
        });

        let mipmap_generator = MipmapGenerator::new(&device);

        Self {
            device,
            queue,
            frame_index: 0,
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_textures: SecondaryMap::new(),
            gpu_samplers: SecondaryMap::new(),
            worlds: FxHashMap::default(),
            gpu_buffers: FxHashMap::default(),
            gpu_images: FxHashMap::default(),
            layout_cache: FxHashMap::default(),
            sampler_cache: FxHashMap::default(),
            dummy_texture: dummy_gpu_tex,
            dummy_sampler,
            mipmap_generator,
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn write_buffer(&mut self, buffer_ref: &BufferRef, data: &[u8]) -> u64 {
        let id = buffer_ref.id();
        let gpu_buf = if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            gpu_buf
        } else {
            let gpu_buf = GpuBuffer::new(&self.device, data, buffer_ref.usage(), buffer_ref.label());
            self.gpu_buffers.insert(id, gpu_buf);
            self.gpu_buffers.get_mut(&id).unwrap()
        };

        if buffer_ref.version > gpu_buf.last_uploaded_version {
            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
            gpu_buf.last_uploaded_version = buffer_ref.version;
        }
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    pub fn prepare_attribute_buffer(&mut self, attr: &crate::resources::geometry::Attribute) -> u64 {
        let id = attr.buffer.id();

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            if attr.version > gpu_buf.last_uploaded_version
                && let Some(data) = &attr.data {
                    let bytes: &[u8] = data.as_ref();
                    self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                    gpu_buf.last_uploaded_version = attr.version;
                }
            gpu_buf.last_used_frame = self.frame_index;
            return gpu_buf.id;
        }

        if let Some(data) = &attr.data {
            let bytes: &[u8] = data.as_ref();
            let mut gpu_buf = GpuBuffer::new(&self.device, bytes, attr.buffer.usage(), attr.buffer.label());
            gpu_buf.last_uploaded_version = attr.version;
            gpu_buf.last_used_frame = self.frame_index;
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        } else {
            log::error!("Geometry attribute buffer {:?} missing CPU data!", attr.buffer.label());
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
                return gpu_buf.id;
            }
            let dummy_data = [0u8; 1];
            let gpu_buf = GpuBuffer::new(&self.device, &dummy_data, attr.buffer.usage(), Some("Dummy Fallback Buffer"));
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        }
    }

    pub fn prepare_uniform_slot_data(&mut self, slot_id: u64, data: &[u8], label: &str) -> u64 {
        let gpu_buf = self.gpu_buffers.entry(slot_id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(&self.device, data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some(label));
            buf.enable_shadow_copy();
            buf
        });
        gpu_buf.update_with_data(&self.device, &self.queue, data);
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    pub fn prepare_geometry(&mut self, assets: &AssetServer, handle: GeometryHandle) {
        let geometry = if let Some(geo) = assets.get_geometry(handle) { geo } else {
            log::warn!("Geometry {:?} not found in AssetServer.", handle);
            return;
        };

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            if geometry.structure_version() == gpu_geo.version && geometry.data_version() == gpu_geo.last_data_version {
                gpu_geo.last_used_frame = self.frame_index;
                return;
            }
        }

        let mut buffer_ids_changed = false;
        let needs_data_update = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.data_version() != gpu_geo.last_data_version
        } else { true };

        if needs_data_update {
            for attr in geometry.attributes().values() {
                let new_id = self.prepare_attribute_buffer(attr);
                if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                    if !gpu_geo.vertex_buffer_ids.contains(&new_id) {
                        buffer_ids_changed = true;
                    }
                }
            }
            if let Some(indices) = geometry.index_attribute() {
                let new_id = self.prepare_attribute_buffer(indices);
                if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                    if let Some((_, _, _, old_id)) = gpu_geo.index_buffer {
                        if old_id != new_id { buffer_ids_changed = true; }
                    }
                }
            }
        }

        let needs_rebuild = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.structure_version() > gpu_geo.version || buffer_ids_changed
        } else { true };

        if needs_rebuild {
            self.create_gpu_geometry(geometry, handle);
        } else if needs_data_update {
            if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
                gpu_geo.last_data_version = geometry.data_version();
            }
        }

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            gpu_geo.last_used_frame = self.frame_index;
        }
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry, handle: GeometryHandle) {
        let layout_info = Arc::new(crate::renderer::pipeline::vertex::generate_vertex_layout(geometry));

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();

        for layout_desc in &layout_info.buffers {
            let gpu_buf = self.gpu_buffers.get(&layout_desc.buffer.id()).expect("Vertex buffer should be prepared");
            vertex_buffers.push(gpu_buf.buffer.clone());
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = geometry.index_attribute() {
            let gpu_buf = self.gpu_buffers.get(&indices.buffer.id()).expect("Index buffer should be prepared");
            let format = match indices.format {
                wgpu::VertexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                wgpu::VertexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                _ => wgpu::IndexFormat::Uint16,
            };
            Some((gpu_buf.buffer.clone(), format, indices.count, gpu_buf.id))
        } else { None };

        let mut draw_range = geometry.draw_range.clone();
        if draw_range == (0..u32::MAX) {
            if let Some(attr) = geometry.attributes().get("position") {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else if let Some(attr) = geometry.attributes().values().next() {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else {
                draw_range = 0..0;
            }
        }

        let gpu_geo = GpuGeometry {
            layout_info,
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            draw_range,
            instance_range: 0..1,
            version: geometry.structure_version(),
            last_data_version: geometry.data_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_geometries.insert(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.gpu_geometries.get(handle)
    }

    pub fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let Some(material) = assets.get_material(handle) else {
            log::warn!("Material {:?} not found in AssetServer.", handle);
            return;
        };

        let uniform_ver = material.data.uniform_version();
        let binding_ver = material.data.binding_version();
        let layout_ver = material.data.layout_version();

        if !self.gpu_materials.contains_key(handle) {
            self.build_full_material(assets, handle, material);
        }

        let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");

        if layout_ver != gpu_mat.last_layout_version {
            self.build_full_material(assets, handle, material);
            return;
        }

        if binding_ver != gpu_mat.last_binding_version {
            self.rebuild_bind_group(assets, handle, material);
            return;
        }

        if uniform_ver != gpu_mat.last_data_version {
            self.update_material_uniforms(handle, material);
        }

        let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
        gpu_mat.last_used_frame = self.frame_index;
    }

    fn prepare_binding_resources(&mut self, assets: &AssetServer, resources: &[BindingResource]) -> Vec<u64> {
        let mut uniform_buffers = Vec::new();

        for resource in resources {
            match resource {
                BindingResource::Buffer { buffer: buffer_ref, offset: _, size: _, data } => {
                    let id = buffer_ref.id();
                    if let Some(bytes) = data {
                        let gpu_buf = self.gpu_buffers.entry(id).or_insert_with(|| {
                            let mut buf = GpuBuffer::new(&self.device, bytes, buffer_ref.usage, buffer_ref.label());
                            buf.last_uploaded_version = buffer_ref.version;
                            buf
                        });

                        if buffer_ref.version > gpu_buf.last_uploaded_version {
                            if bytes.len() as u64 > gpu_buf.size {
                                log::debug!("Recreating buffer {:?} due to size increase.", buffer_ref.label());
                                *gpu_buf = GpuBuffer::new(&self.device, bytes, buffer_ref.usage, buffer_ref.label());
                            } else {
                                self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                            }
                            gpu_buf.last_uploaded_version = buffer_ref.version;
                        }
                        gpu_buf.last_used_frame = self.frame_index;
                    } else {
                        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
                            gpu_buf.last_used_frame = self.frame_index;
                        } else {
                            panic!("ResourceManager: Trying to bind buffer {:?} (ID: {}) but it is not initialized!", buffer_ref.label(), id);
                        }
                    }
                    uniform_buffers.push(id);
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(handle) = handle_opt {
                        self.prepare_texture(assets, *handle);
                    }
                },
                _ => {}
            }
        }
        uniform_buffers
    }

    fn build_full_material(&mut self, assets: &AssetServer, handle: MaterialHandle, material: &crate::resources::material::Material) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        let uniform_buffers = self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GpuMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            uniform_buffers,
            last_data_version: material.data.uniform_version(),
            last_binding_version: material.data.binding_version(),
            last_layout_version: material.data.layout_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_materials.insert(handle, gpu_mat);
        self.gpu_materials.get(handle).expect("Just inserted")
    }

    fn rebuild_bind_group(&mut self, assets: &AssetServer, handle: MaterialHandle, material: &crate::resources::material::Material) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        let uniform_buffers = self.prepare_binding_resources(assets, &builder.resources);
        let layout = {
            let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");
            gpu_mat.layout.clone()
        };

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);

        {
            let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
            gpu_mat.bind_group = bind_group;
            gpu_mat.bind_group_id = bg_id;
            gpu_mat.uniform_buffers = uniform_buffers;
            gpu_mat.last_binding_version = material.data.binding_version();
            gpu_mat.last_used_frame = self.frame_index;
        }

        self.gpu_materials.get(handle).expect("gpu material should exist.")
    }

    fn update_material_uniforms(&mut self, handle: MaterialHandle, material: &crate::resources::material::Material) {
        use crate::resources::material::MaterialData;

        match &material.data {
            MaterialData::Basic(m) => {
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                    && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                        self.queue.write_buffer(&gpu_buf.buffer, 0, m.uniforms.as_bytes());
                    }
            },
            MaterialData::Phong(m) => {
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                    && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                        self.queue.write_buffer(&gpu_buf.buffer, 0, m.uniforms.as_bytes());
                    }
            },
            MaterialData::Standard(m) => {
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                    && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                        self.queue.write_buffer(&gpu_buf.buffer, 0, m.uniforms.as_bytes());
                    }
            },
        }

        if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
            gpu_mat.last_data_version = material.data.uniform_version();
        }
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }

    pub fn get_or_create_layout(&mut self, entries: &[wgpu::BindGroupLayoutEntry]) -> (wgpu::BindGroupLayout, u64) {
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        let id = generate_resource_id();
        self.layout_cache.insert(entries.to_vec(), (layout.clone(), id));
        (layout, id)
    }

    pub fn create_bind_group(&self, layout: &wgpu::BindGroupLayout, resources: &[BindingResource]) -> (wgpu::BindGroup, u64) {
        let mut entries = Vec::new();

        for (i, resource_data) in resources.iter().enumerate() {
            let binding_resource = match resource_data {
                BindingResource::Buffer { buffer, data: _, offset, size } => {
                    let cpu_id = buffer.id();
                    let gpu_buf = self.gpu_buffers.get(&cpu_id).expect("Buffer should be prepared");
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new),
                    })
                },
                BindingResource::Texture(handle_opt) => {
                    let gpu_tex = if let Some(handle) = handle_opt {
                        self.gpu_textures.get(*handle).unwrap_or(&self.dummy_texture)
                    } else { &self.dummy_texture };
                    wgpu::BindingResource::TextureView(&gpu_tex.view)
                },
                BindingResource::Sampler(handle_opt) => {
                    let sampler = if let Some(handle) = handle_opt {
                        self.gpu_samplers.get(*handle).unwrap_or(&self.dummy_sampler)
                    } else { &self.dummy_sampler };
                    wgpu::BindingResource::Sampler(sampler)
                },
                BindingResource::_Phantom(_) => unreachable!("_Phantom should never be used"),
            };

            entries.push(wgpu::BindGroupEntry { binding: i as u32, resource: binding_resource });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auto BindGroup"),
            layout,
            entries: &entries,
        });

        (bind_group, generate_resource_id())
    }

    fn prepare_image(&mut self, image: &Image, required_mip_count: u32, required_usage: wgpu::TextureUsages) {
        let id = image.id();
        let mut needs_recreate = false;

        if let Some(gpu_img) = self.gpu_images.get(&id) {
            if gpu_img.mip_level_count < required_mip_count || !gpu_img.usage.contains(required_usage) {
                needs_recreate = true;
            }
        } else {
            needs_recreate = true;
        }

        if needs_recreate {
            self.gpu_images.remove(&id);
            let mut gpu_img = GpuImage::new(&self.device, &self.queue, image, required_mip_count, required_usage);
            gpu_img.last_used_frame = self.frame_index;
            self.gpu_images.insert(id, gpu_img);
        } else if let Some(gpu_img) = self.gpu_images.get_mut(&id) {
            gpu_img.update(&self.device, &self.queue, image);
            gpu_img.last_used_frame = self.frame_index;
        }
    }

    pub fn prepare_texture(&mut self, assets: &AssetServer, handle: TextureHandle) {
        let Some(texture_asset) = assets.get_texture(handle) else {
            log::warn!("Texture asset not found for handle: {:?}", handle);
            return;
        };

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            let tex_ver_match = gpu_tex.version == texture_asset.version();
            let img_id_match = gpu_tex.image_id == texture_asset.image.id();
            let img_gen_match = gpu_tex.image_generation_id == texture_asset.image.generation_id();
            let img_data_match = gpu_tex.image_data_version == texture_asset.image.version();

            if tex_ver_match && img_id_match && img_gen_match && img_data_match {
                gpu_tex.last_used_frame = self.frame_index;
                if let Some(gpu_img) = self.gpu_images.get_mut(&gpu_tex.image_id) {
                    gpu_img.last_used_frame = self.frame_index;
                }
                return;
            }
        }

        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let image_mips = 1;
        let generated_mips = if texture_asset.generate_mipmaps { texture_asset.mip_level_count() } else { 1 };
        let final_mip_count = std::cmp::max(image_mips, generated_mips);

        if final_mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        self.prepare_image(&texture_asset.image, final_mip_count, usage);

        let image_id = texture_asset.image.id();
        let gpu_image = self.gpu_images.get(&image_id).expect("GpuImage should be ready");

        if texture_asset.generate_mipmaps && !gpu_image.mipmaps_generated {
            let gpu_img_mut = self.gpu_images.get_mut(&image_id).unwrap();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Mipmap Gen") });
            self.mipmap_generator.generate(&self.device, &mut encoder, &gpu_img_mut.texture, gpu_img_mut.mip_level_count);
            self.queue.submit(Some(encoder.finish()));
            gpu_img_mut.mipmaps_generated = true;
        }

        let gpu_image = self.gpu_images.get(&image_id).unwrap();

        let mut needs_update_texture = false;
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            let config_changed = gpu_tex.version != texture_asset.version();
            let image_recreated = gpu_tex.image_generation_id != gpu_image.generation_id;
            let image_swapped = gpu_tex.image_id != image_id;

            if config_changed || image_recreated || image_swapped {
                needs_update_texture = true;
            }
            gpu_tex.last_used_frame = self.frame_index;
        } else {
            needs_update_texture = true;
        }

        if needs_update_texture {
            let gpu_tex = GpuTexture::new(texture_asset, gpu_image);
            self.gpu_textures.insert(handle, gpu_tex);

            let sampler = self.get_or_create_sampler(&texture_asset.sampler, texture_asset.name());
            self.gpu_samplers.insert(handle, sampler);
        }

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            gpu_tex.last_used_frame = self.frame_index;
        }
    }

    fn get_or_create_sampler(&mut self, config: &TextureSampler, label: Option<&str>) -> wgpu::Sampler {
        if let Some(sampler) = self.sampler_cache.get(config) {
            return sampler.clone();
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
            address_mode_u: config.address_mode_u,
            address_mode_v: config.address_mode_v,
            address_mode_w: config.address_mode_w,
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            mipmap_filter: config.mipmap_filter,
            compare: config.compare,
            anisotropy_clamp: config.anisotropy_clamp,
            ..Default::default()
        });

        self.sampler_cache.insert(*config, sampler.clone());
        sampler
    }

    pub fn prepare_global(&mut self, assets: &AssetServer, env: &Environment, render_state: &RenderState) {
        let world_id = Self::compose_env_render_state_id(render_state.id, env.id);

        if let Some(gpu_env) = self.worlds.get_mut(&world_id) {
            let uniform_match = gpu_env.last_uniform_version == env.uniforms().buffer.version;
            let binding_match = gpu_env.last_binding_version == env.binding_version();
            let layout_match = gpu_env.last_layout_version == env.layout_version();
            let render_state_match = render_state.uniforms().buffer.version == gpu_env.last_render_state_version;

            if uniform_match && binding_match && layout_match && render_state_match {
                gpu_env.last_used_frame = self.frame_index;
                return;
            }
        }

        let mut builder = ResourceBuilder::new();
        render_state.define_bindings(&mut builder);
        env.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);

        let needs_new_bind_group = if let Some(gpu_env) = self.worlds.get(&world_id) {
            gpu_env.layout_id != layout_id || gpu_env.last_binding_version != env.binding_version()
        } else { true };

        if !needs_new_bind_group {
            if let Some(gpu_env) = self.worlds.get_mut(&world_id) {
                gpu_env.last_uniform_version = env.uniforms().buffer.version;
                gpu_env.last_render_state_version = render_state.uniforms().buffer.version;
                gpu_env.last_used_frame = self.frame_index;
            }
            return;
        }

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(0);

        let gpu_world = GpuEnvironment {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            last_uniform_version: env.uniforms().buffer.version,
            last_binding_version: env.binding_version(),
            last_layout_version: env.layout_version(),
            last_render_state_version: render_state.uniforms().buffer.version,
            last_used_frame: self.frame_index,
        };
        self.worlds.insert(world_id, gpu_world);
    }

    fn compose_env_render_state_id(render_state_id: u32, env_id: u32) -> u64 {
        ((render_state_id as u64) << 32) | (env_id as u64)
    }

    pub fn get_world(&self, render_state_id: u32, env_id: u32) -> Option<&GpuEnvironment> {
        let world_id = Self::compose_env_render_state_id(render_state_id, env_id);
        self.worlds.get(&world_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames { return; }
        let cutoff = self.frame_index - ttl_frames;

        self.gpu_geometries.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_textures.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_samplers.retain(|k, _| self.gpu_textures.contains_key(k));
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.worlds.retain(|_, v| v.last_used_frame >= cutoff);
    }
}
