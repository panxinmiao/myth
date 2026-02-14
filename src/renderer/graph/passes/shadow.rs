use glam::{Mat4, Vec4};

use crate::renderer::core::view::ViewTarget;
use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::RenderContext;
use crate::renderer::graph::shadow_utils::MAX_CASCADES;

pub struct ShadowPass {
    light_uniform_buffer: wgpu::Buffer,
    light_uniform_capacity: u32,
    light_uniform_stride: u32,
    light_bind_group: wgpu::BindGroup,
}

impl ShadowPass {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = align_to(std::mem::size_of::<Mat4>() as u32, min_alignment);

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Light BindGroup Layout"),
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

        let light_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light Uniform Buffer"),
            size: u64::from(stride),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light BindGroup"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &light_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                }),
            }],
        });

        Self {
            light_uniform_buffer,
            light_uniform_capacity: 1,
            light_uniform_stride: stride,
            light_bind_group,
        }
    }

    fn recreate_light_bind_group(
        &mut self,
        device: &wgpu::Device,
        light_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        self.light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Light BindGroup"),
            layout: light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.light_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                }),
            }],
        });
    }

    fn ensure_light_uniform_capacity(
        &mut self,
        device: &wgpu::Device,
        light_bind_group_layout: &wgpu::BindGroupLayout,
        required_count: u32,
    ) {
        if required_count <= self.light_uniform_capacity {
            return;
        }

        let mut capacity = self.light_uniform_capacity.max(1);
        while capacity < required_count {
            capacity = capacity.saturating_mul(2);
        }

        self.light_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Light Uniform Buffer"),
            size: u64::from(self.light_uniform_stride) * u64::from(capacity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.recreate_light_bind_group(device, light_bind_group_layout);

        self.light_uniform_capacity = capacity;
    }
}

impl RenderNode for ShadowPass {
    fn name(&self) -> &'static str {
        "Shadow Pass"
    }

    /// Reads pre-computed [`RenderView`]s from `render_lists.active_views`,
    /// allocates GPU resources, and uploads VP matrices + light storage data.
    ///
    /// All matrix computation is done upstream by `SceneCullPass` (via `shadow_utils`).
    fn prepare(&mut self, ctx: &mut RenderContext) {
        let shadow_layout_entries = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
            },
            count: None,
        }];
        let (shadow_global_layout, _) = ctx
            .resource_manager
            .get_or_create_layout(&shadow_layout_entries);

        // ============================================================
        // 1. Count shadow views and determine resource sizes
        // ============================================================
        let total_layers = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .count() as u32;

        let max_map_size = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .map(|v| v.viewport_size.0)
            .max()
            .unwrap_or(1);

        // ============================================================
        // 2. Ensure GPU resources
        // ============================================================
        ctx.resource_manager
            .ensure_shadow_maps(total_layers, max_map_size);
        self.ensure_light_uniform_capacity(
            &ctx.wgpu_ctx.device,
            &shadow_global_layout,
            total_layers.max(1),
        );
        self.recreate_light_bind_group(&ctx.wgpu_ctx.device, &shadow_global_layout);

        // ============================================================
        // 3. Reset light storage buffer shadow fields
        // ============================================================
        {
            let mut light_storage = ctx.scene.light_storage_buffer.write();
            for light in light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.shadow_matrices.0 = [Mat4::IDENTITY; 4];
                light.cascade_count = 0;
                light.cascade_splits = Vec4::ZERO;
            }
        }

        ctx.render_lists.shadow_lights.clear();

        if total_layers == 0 {
            ctx.resource_manager
                .ensure_buffer(&ctx.scene.light_storage_buffer);
            return;
        }

        // ============================================================
        // 4. Write VP matrices to per-layer uniform buffer and
        //    create ShadowLightInstance entries
        // ============================================================
        let mut shadow_uniform_data =
            vec![0u8; (self.light_uniform_stride as usize) * (total_layers as usize)];

        for view in &ctx.render_lists.active_views {
            let ViewTarget::ShadowLight {
                light_id,
                layer_index,
            } = view.target
            else {
                continue;
            };

            // Write VP matrix to the uniform buffer at this layer's slot
            let offset = layer_index as usize * self.light_uniform_stride as usize;
            let bytes = bytemuck::bytes_of(&view.view_projection);
            shadow_uniform_data[offset..offset + bytes.len()].copy_from_slice(bytes);

            // Create ShadowLightInstance for the run phase
            ctx.render_lists.shadow_lights.push(
                crate::renderer::graph::frame::ShadowLightInstance {
                    light_id,
                    layer_index,
                    light_buffer_index: view.light_buffer_index,
                    light_view_projection: view.view_projection,
                },
            );
        }

        ctx.wgpu_ctx
            .queue
            .write_buffer(&self.light_uniform_buffer, 0, &shadow_uniform_data);

        // ============================================================
        // 5. Update light storage buffer with per-light shadow metadata
        //    (aggregated from per-view data)
        // ============================================================
        {
            let mut light_storage = ctx.scene.light_storage_buffer.write();

            for (light_buffer_index, light) in ctx.extracted_scene.lights.iter().enumerate() {
                if !light.cast_shadows {
                    continue;
                }

                let shadow_cfg = light.shadow.clone().unwrap_or_default();

                // Collect all views belonging to this light
                let mut base_layer = u32::MAX;
                let mut cascade_count = 0u32;
                let mut cascade_matrices = [Mat4::IDENTITY; MAX_CASCADES as usize];
                let mut cascade_splits_arr = [0.0f32; MAX_CASCADES as usize];

                for view in &ctx.render_lists.active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };

                    if light_id != light.id {
                        continue;
                    }

                    if layer_index < base_layer {
                        base_layer = layer_index;
                    }

                    // Cascade index within this light (relative to base_layer)
                    // We compute it after finding base_layer, so do a second pass below.
                    cascade_count += 1;
                }

                if cascade_count == 0 {
                    continue;
                }

                // Second pass: fill matrices and splits relative to base_layer
                for view in &ctx.render_lists.active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };

                    if light_id != light.id {
                        continue;
                    }

                    let cascade_idx = (layer_index - base_layer) as usize;
                    if cascade_idx < MAX_CASCADES as usize {
                        cascade_matrices[cascade_idx] = view.view_projection;
                        if let Some(split) = view.csm_split {
                            cascade_splits_arr[cascade_idx] = split;
                        }
                    }
                }

                if let Some(gpu_light) = light_storage.get_mut(light_buffer_index) {
                    gpu_light.shadow_layer_index = base_layer as i32;
                    gpu_light.shadow_matrices.0 = cascade_matrices;
                    gpu_light.cascade_count = cascade_count;
                    gpu_light.cascade_splits = Vec4::new(
                        cascade_splits_arr[0],
                        cascade_splits_arr[1.min(cascade_count as usize - 1)],
                        cascade_splits_arr[2.min(cascade_count as usize - 1)],
                        cascade_splits_arr[3.min(cascade_count as usize - 1)],
                    );
                    gpu_light.shadow_bias = shadow_cfg.bias;
                    gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                }
            }
        }

        ctx.resource_manager
            .ensure_buffer(&ctx.scene.light_storage_buffer);
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        if ctx.render_lists.shadow_lights.is_empty() {
            return;
        }

        for shadow_light in &ctx.render_lists.shadow_lights {
            let Some(layer_view) = ctx
                .resource_manager
                .create_shadow_2d_layer_view(shadow_light.layer_index)
            else {
                continue;
            };

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth Pass"),
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
            let dynamic_offset = shadow_light.layer_index * self.light_uniform_stride;
            pass.set_bind_group(0, &self.light_bind_group, &[dynamic_offset]);

            // Look up per-view command queue using (light_id, layer_index)
            let Some(commands) = ctx
                .render_lists
                .shadow_queues
                .get(&(shadow_light.light_id, shadow_light.layer_index))
            else {
                continue;
            };

            for cmd in commands {
                pass.set_pipeline(&cmd.pipeline);

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

impl Default for ShadowPass {
    fn default() -> Self {
        panic!("ShadowPass::default is unavailable, use ShadowPass::new(device)");
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}
