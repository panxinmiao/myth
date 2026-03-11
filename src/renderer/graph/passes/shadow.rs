//! RDG Shadow Render Pass
//!
//! Renders shadow depth maps for all shadow-casting lights using a 2D texture
//! array. Each cascade / light gets its own layer. VP matrices are uploaded to
//! a per-layer dynamic uniform buffer managed by this pass.
//!
//! # RDG Integration
//!
//! The shadow map is a **transient RDG texture** — its lifetime is fully managed
//! by the render graph compiler. Consumer passes (Opaque, Transparent, etc.)
//! declare explicit `read_texture` dependencies to establish DAG ordering and
//! enable automatic memory aliasing with other non-overlapping intermediates.
//!
//! # Data Flow
//!
//! ```text
//! RenderLists::active_views (shadow views)     ← built by extract_and_prepare
//! RenderLists::shadow_queues (per-view cmds)   ← built by culling::cull_and_sort
//!         │
//!         ▼
//!  ┌──────────────────────────────────────┐
//!  │   ShadowPass                         │
//!  │  setup:   create_texture (RDG)       │
//!  │  prepare: resolve physical → views   │
//!  │  execute: draw depth per layer       │
//!  └──────────────────────────────────────┘
//!         │
//!         ▼  (TextureNodeId via explicit wiring)
//!  Consumer passes (Opaque / Transparent)
//!         read_texture(shadow_id) → DAG edge
//! ```

use glam::Mat4;

use crate::renderer::core::view::ViewTarget;
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassBuilder, PassNode, PrepareContext, RenderGraph,
    TextureDesc, TextureNodeId,
};
use crate::renderer::graph::frame::ShadowLightInstance;
use crate::renderer::graph::passes::draw::submit_draw_commands;

/// Shadow rendering feature.
///
/// Manages a dynamic uniform buffer for per-layer VP matrices and produces an
/// ephemeral [`ShadowPassNode`] each frame via [`Self::add_to_graph`].
///
/// The shadow map texture is a **transient RDG resource** — its physical
/// memory is allocated by the graph compiler and can be aliased with other
/// non-overlapping intermediates. If no shadow-casting lights are active,
/// no GPU memory is allocated at all.
pub struct ShadowFeature {
    /// Per-light VP matrix buffer (dynamic uniform, stride-aligned).
    uniform_buffer: wgpu::Buffer,
    /// Current buffer capacity in layers.
    uniform_capacity: u32,
    /// Aligned stride between consecutive VP matrices.
    uniform_stride: u32,
    /// Bind group referencing `uniform_buffer`.
    bind_group: wgpu::BindGroup,
    /// Bind group layout for shadow light uniforms.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Populated during extract — one entry per shadow layer.
    shadow_lights: Vec<ShadowLightInstance>,
    /// Total shadow layers required this frame (set during extract).
    total_layers: u32,
    /// Maximum shadow map resolution across all shadow views this frame.
    shadow_resolution: u32,
}

impl ShadowFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = align_to(std::mem::size_of::<Mat4>() as u32, min_alignment);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Light BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow VP Buffer"),
            size: u64::from(stride),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                }),
            }],
        });

        Self {
            uniform_buffer,
            uniform_capacity: 1,
            uniform_stride: stride,
            bind_group,
            bind_group_layout,
            shadow_lights: Vec::with_capacity(16),
            total_layers: 0,
            shadow_resolution: 1,
        }
    }

    /// Grow the uniform buffer (and recreate the bind group) if the current
    /// capacity is insufficient.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required_count: u32) {
        if required_count <= self.uniform_capacity {
            return;
        }

        let mut capacity = self.uniform_capacity.max(1);
        while capacity < required_count {
            capacity = capacity.saturating_mul(2);
        }

        self.uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow VP Buffer"),
            size: u64::from(self.uniform_stride) * u64::from(capacity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.recreate_bind_group(device);
        self.uniform_capacity = capacity;
    }

    fn recreate_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light BG"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                }),
            }],
        });
    }
}

impl ShadowFeature {
    /// Extract shadow light data and prepare GPU resources.
    ///
    /// Uploads per-layer VP matrices and builds the shadow light instance list.
    /// The physical shadow texture will be allocated by the RDG transient pool
    /// after graph compilation — no `ResourceManager` allocation is performed.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.shadow_lights.clear();
        self.total_layers = 0;
        self.shadow_resolution = 1;

        let total_layers = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .count() as u32;

        if total_layers == 0 {
            return;
        }

        self.total_layers = total_layers;
        self.shadow_resolution = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .map(|v| v.viewport_size.0)
            .max()
            .unwrap_or(1);

        // Ensure buffer capacity and recreate bind group
        self.ensure_capacity(ctx.device, total_layers);
        self.recreate_bind_group(ctx.device);

        // Build uniform data + shadow light instances
        let mut uniform_data = vec![0u8; self.uniform_stride as usize * total_layers as usize];

        for view in &ctx.render_lists.active_views {
            let ViewTarget::ShadowLight {
                light_id,
                layer_index,
            } = view.target
            else {
                continue;
            };

            let offset = layer_index as usize * self.uniform_stride as usize;
            let bytes = bytemuck::bytes_of(&view.view_projection);
            uniform_data[offset..offset + bytes.len()].copy_from_slice(bytes);

            self.shadow_lights.push(ShadowLightInstance {
                light_id,
                layer_index,
                light_buffer_index: view.light_buffer_index,
                light_view_projection: view.view_projection,
            });
        }

        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, &uniform_data);
    }

    /// Create an ephemeral [`ShadowPassNode`] and add it to the render graph.
    ///
    /// Registers a transient `Depth32Float` 2D-array texture in the RDG and
    /// returns its [`TextureNodeId`] so that consumer passes (Opaque,
    /// Transparent, etc.) can wire explicit read dependencies.
    ///
    /// Returns `None` if no shadow layers are required this frame (the pass
    /// is not added to the graph at all, and no GPU memory is allocated).
    pub fn add_to_graph(&self, graph: &mut RenderGraph) -> Option<TextureNodeId> {
        if self.total_layers == 0 || self.shadow_lights.is_empty() {
            return None;
        }

        let desc = TextureDesc::new(
            self.shadow_resolution,
            self.shadow_resolution,
            self.total_layers,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let shadow_array_id = graph.register_resource("Shadow_Array_Map", desc, false);

        let node = ShadowPassNode {
            bind_group: self.bind_group.clone(),
            shadow_lights: self.shadow_lights.clone(),
            uniform_stride: self.uniform_stride,
            shadow_array_id,
            shadow_layer_views: Vec::with_capacity(self.shadow_lights.len()),
        };
        graph.add_pass(Box::new(node));

        Some(shadow_array_id)
    }
}

// ─── Shadow Pass Node ─────────────────────────────────────────────────────────

/// Ephemeral per-frame shadow render pass node.
///
/// Carries the RDG texture node ID for the shadow array and cloned light
/// metadata. Layer texture views are created in the [`prepare`] phase after
/// the RDG compiler has allocated the physical GPU texture.
pub struct ShadowPassNode {
    bind_group: wgpu::BindGroup,
    shadow_lights: Vec<ShadowLightInstance>,
    uniform_stride: u32,
    /// RDG transient texture node for the Depth32Float 2D-array.
    shadow_array_id: TextureNodeId,
    /// Per-layer texture views, populated in [`Self::prepare`].
    shadow_layer_views: Vec<wgpu::TextureView>,
}

impl PassNode for ShadowPassNode {
    fn name(&self) -> &'static str {
        "Shadow_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.declare_output(self.shadow_array_id);
    }

    /// Resolve the physical shadow texture and create per-layer D2 views.
    ///
    /// Called after the RDG compiler has allocated transient GPU memory.
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.shadow_layer_views.clear();

        if self.shadow_lights.is_empty() {
            return;
        }

        let texture = ctx.views.get_texture(self.shadow_array_id);

        for shadow_light in &self.shadow_lights {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Shadow_Layer_View"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: shadow_light.layer_index,
                array_layer_count: Some(1),
                ..Default::default()
            });
            self.shadow_layer_views.push(view);
        }
    }

    /// Render shadow depth maps — one render pass per shadow layer.
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.shadow_lights.is_empty() {
            return;
        }

        for (i, shadow_light) in self.shadow_lights.iter().enumerate() {
            let Some(layer_view) = self.shadow_layer_views.get(i) else {
                continue;
            };

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let raw_pass = encoder.begin_render_pass(&pass_desc);
            let mut pass = raw_pass;

            let dynamic_offset = shadow_light.layer_index * self.uniform_stride;
            pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);

            let Some(commands) = ctx
                .baked_lists
                .shadow_queues
                .get(&(shadow_light.light_id, shadow_light.layer_index))
            else {
                continue;
            };

            submit_draw_commands(&mut pass, commands);
        }
    }
}

#[inline]
fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}
