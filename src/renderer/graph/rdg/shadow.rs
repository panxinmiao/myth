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
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;

/// RDG Shadow Render Pass.
///
/// Manages a dynamic uniform buffer for per-layer VP matrices and issues
/// one depth-only render pass per shadow view.
pub struct RdgShadowPass {
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
}

impl RdgShadowPass {
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

impl PassNode for RdgShadowPass {
    fn name(&self) -> &'static str {
        "RDG_Shadow_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Shadow maps are external resources managed by ResourceManager.
        // Mark as side-effect so the pass is never culled.
        builder.mark_side_effect();
    }

    /// Upload per-layer VP matrices and build the shadow light instance list.
    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.shadow_lights.clear();

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
    }

    /// Render shadow depth maps — one render pass per shadow layer.
    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.shadow_lights.is_empty() {
            return;
        }

        for shadow_light in &self.shadow_lights {
            let Some(layer_view) = ctx
                .resource_manager
                .create_shadow_2d_layer_view(shadow_light.layer_index)
            else {
                continue;
            };

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("RDG Shadow Depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
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

            let mut pass = encoder.begin_render_pass(&pass_desc);
            let dynamic_offset = shadow_light.layer_index * self.uniform_stride;
            pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);

            let Some(commands) = ctx
                .render_lists
                .shadow_queues
                .get(&(shadow_light.light_id, shadow_light.layer_index))
            else {
                continue;
            };

            for cmd in commands {
                let pipeline = ctx.pipeline_cache.get_render_pipeline(cmd.pipeline_id);
                pass.set_pipeline(pipeline);

                if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                    pass.set_bind_group(1, &gpu_material.bind_group, &[]);
                }

                pass.set_bind_group(2, &cmd.object_bind_group.bind_group, &[cmd.dynamic_offset]);

                if let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) {
                    for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                        pass.set_vertex_buffer(slot as u32, buffer.slice(..));
                    }

                    if let Some((index_buffer, index_format, count, _)) = &gpu_geometry.index_buffer
                    {
                        pass.set_index_buffer(index_buffer.slice(..), *index_format);
                        pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                    } else {
                        pass.draw(
                            gpu_geometry.draw_range.clone(),
                            gpu_geometry.instance_range.clone(),
                        );
                    }
                }
            }
        }
    }
}

#[inline]
fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}
