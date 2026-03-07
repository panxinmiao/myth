//! RDG Shadow Render Pass
//!
//! Renders shadow depth maps for all shadow-casting lights using a 2D texture
//! array. Each cascade / light gets its own layer. VP matrices are uploaded to
//! a per-layer dynamic uniform buffer managed by this pass.
//!
//! # RDG Integration
//!
//! The pass is a **side-effect** node — it writes to the external shadow map
//! texture array managed by [`ResourceManager`] and does not participate in
//! the transient-resource dependency graph.
//!
//! # Data Flow
//!
//! ```text
//! RenderLists::active_views (shadow views)     ← built by extract_and_prepare
//! RenderLists::shadow_queues (per-view cmds)   ← built by culling::cull_and_sort
//!         │
//!         ▼
//!  ┌──────────────────────┐
//!  │   RdgShadowPass      │
//!  │  prepare: upload VP   │
//!  │  execute: draw depth  │
//!  └──────────────────────┘
//!         │
//!         ▼
//!  Shadow 2D Array Texture  → consumed by global bind group (Group 0)
//! ```

use glam::Mat4;

use crate::renderer::core::view::ViewTarget;
use crate::renderer::graph::frame::ShadowLightInstance;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{ExtractContext, RdgExecuteContext};
use crate::renderer::graph::rdg::draw::submit_draw_commands;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;

/// Shadow rendering feature.
///
/// Manages a dynamic uniform buffer for per-layer VP matrices.
/// Produces an ephemeral [`ShadowPassNode`] each frame via [`Self::add_to_graph`].
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
    /// Populated during prepare — one entry per shadow layer.
    shadow_lights: Vec<ShadowLightInstance>,
    /// Pre-created shadow layer texture views, parallel to `shadow_lights`.
    shadow_layer_views: Vec<wgpu::TextureView>,
}

impl ShadowFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = align_to(std::mem::size_of::<Mat4>() as u32, min_alignment);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG Shadow Light BGL"),
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
            label: Some("RDG Shadow VP Buffer"),
            size: u64::from(stride),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RDG Shadow Light BG"),
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
            shadow_layer_views: Vec::with_capacity(16),
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
            label: Some("RDG Shadow VP Buffer"),
            size: u64::from(self.uniform_stride) * u64::from(capacity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.recreate_bind_group(device);
        self.uniform_capacity = capacity;
    }

    fn recreate_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RDG Shadow Light BG"),
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
    /// Uploads per-layer VP matrices, builds the shadow light instance list,
    /// and pre-creates per-layer texture views so the execute phase never
    /// touches the resource manager.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.shadow_lights.clear();
        self.shadow_layer_views.clear();

        let total_layers = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .count() as u32;

        if total_layers == 0 {
            return;
        }

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

        // Pre-create per-layer texture views for the execute phase.
        for shadow_light in &self.shadow_lights {
            if let Some(view) = ctx
                .resource_manager
                .create_shadow_2d_layer_view(shadow_light.layer_index)
            {
                self.shadow_layer_views.push(view);
            }
        }
    }

    /// Create an ephemeral [`ShadowPassNode`] and add it to the render graph.
    pub fn add_to_graph(&self, rdg: &mut RenderGraph) {
        let node = ShadowPassNode {
            bind_group: self.bind_group.clone(),
            shadow_lights: self.shadow_lights.clone(),
            uniform_stride: self.uniform_stride,
            shadow_layer_views: self.shadow_layer_views.clone(),
        };
        rdg.add_pass(Box::new(node));
    }
}

// ─── Shadow Pass Node ─────────────────────────────────────────────────────────

/// Ephemeral per-frame shadow render pass node.
///
/// Carries pre-created layer texture views and cloned light metadata.
/// The execute phase iterates shadow layers and submits pre-baked
/// [`DrawCommand`]s without any resource-manager lookups.
pub struct ShadowPassNode {
    bind_group: wgpu::BindGroup,
    shadow_lights: Vec<ShadowLightInstance>,
    uniform_stride: u32,
    /// Pre-created per-layer texture views, parallel to `shadow_lights`.
    shadow_layer_views: Vec<wgpu::TextureView>,
}

impl PassNode for ShadowPassNode {
    fn name(&self) -> &'static str {
        "RDG_Shadow_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.mark_side_effect();
    }

    /// Render shadow depth maps — one render pass per shadow layer.
    ///
    /// Uses pre-baked [`DrawCommand`]s from [`BakedRenderLists::shadow_queues`]
    /// and pre-created layer texture views, eliminating all resource-manager
    /// lookups during the execute phase.
    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.shadow_lights.is_empty() {
            return;
        }

        for (i, shadow_light) in self.shadow_lights.iter().enumerate() {
            let Some(layer_view) = self.shadow_layer_views.get(i) else {
                continue;
            };

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("RDG Shadow Depth"),
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
