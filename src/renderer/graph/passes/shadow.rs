use glam::{Mat4, Vec3};

use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::RenderContext;
use crate::scene::light::LightKind;

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

        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    fn build_light_vp(light_kind: &LightKind, position: Vec3, direction: Vec3, camera_pos: Vec3) -> Mat4 {
        let safe_dir = if direction.length_squared() > 1e-6 {
            direction.normalize()
        } else {
            -Vec3::Z
        };

        let up = if safe_dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };

        match light_kind {
            LightKind::Directional(_) => {
                let _ = camera_pos;
                let center = Vec3::ZERO;
                let eye = center - safe_dir * 50.0;
                let view = Mat4::look_at_rh(eye, center, up);
                let proj = Mat4::orthographic_rh(-30.0, 30.0, -30.0, 30.0, 0.1, 150.0);
                proj * view
            }
            LightKind::Spot(spot) => {
                let view = Mat4::look_at_rh(position, position + safe_dir, up);
                let fov = (spot.outer_cone * 2.0).clamp(0.1, std::f32::consts::PI - 0.01);
                let far = spot.range.max(1.0);
                let proj = Mat4::perspective_rh(fov, 1.0, 0.1, far);
                proj * view
            }
            LightKind::Point(_) => Mat4::IDENTITY,
        }
    }
}

impl RenderNode for ShadowPass {
    fn name(&self) -> &'static str {
        "Shadow Pass"
    }

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

        let mut active_shadow_lights = Vec::with_capacity(ctx.extracted_scene.lights.len());
        let mut max_map_size = 1u32;

        for (light_index, light) in ctx.extracted_scene.lights.iter().enumerate() {
            if !light.cast_shadows {
                continue;
            }

            if !matches!(light.kind, LightKind::Directional(_) | LightKind::Spot(_)) {
                continue;
            }

            let Some(queue) = ctx.render_lists.shadow_queues.get(&light.id) else {
                continue;
            };
            if queue.is_empty() {
                continue;
            }

            max_map_size = max_map_size.max(light.shadow.as_ref().map_or(1024, |shadow| shadow.map_size));
            active_shadow_lights.push((light_index, light.clone()));
        }

        let required_count = active_shadow_lights.len() as u32;
        ctx.resource_manager
            .ensure_shadow_maps(required_count, max_map_size);
        self.ensure_light_uniform_capacity(
            &ctx.wgpu_ctx.device,
            &shadow_global_layout,
            required_count.max(1),
        );
        self.recreate_light_bind_group(&ctx.wgpu_ctx.device, &shadow_global_layout);

        ctx.render_lists.shadow_lights.clear();

        {
            let mut light_storage = ctx.scene.light_storage_buffer.write();
            for light in light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.shadow_matrix = Mat4::IDENTITY;
            }

            if required_count != 0 {
                let mut shadow_matrices =
                    vec![0u8; (self.light_uniform_stride as usize) * (required_count as usize)];
                let camera_pos = ctx.camera.position.to_array().into();

                for (layer_index, (light_buffer_index, light)) in
                    active_shadow_lights.iter().enumerate()
                {
                    let light_vp = Self::build_light_vp(
                        &light.kind,
                        light.position,
                        light.direction,
                        camera_pos,
                    );
                    let layer_offset = layer_index * self.light_uniform_stride as usize;
                    let bytes = bytemuck::bytes_of(&light_vp);
                    shadow_matrices[layer_offset..layer_offset + bytes.len()].copy_from_slice(bytes);

                    if let Some(gpu_light) = light_storage.get_mut(*light_buffer_index) {
                        gpu_light.shadow_layer_index = layer_index as i32;
                        gpu_light.shadow_matrix = light_vp;
                    }

                    ctx.render_lists.shadow_lights.push(
                        crate::renderer::graph::frame::ShadowLightInstance {
                            light_id: light.id,
                            layer_index: layer_index as u32,
                            light_buffer_index: *light_buffer_index,
                            light_view_projection: light_vp,
                        },
                    );
                }

                ctx.wgpu_ctx
                    .queue
                    .write_buffer(&self.light_uniform_buffer, 0, &shadow_matrices);
            }
        }

        ctx.resource_manager.ensure_buffer(&ctx.scene.light_storage_buffer);
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

            let Some(commands) = ctx.render_lists.shadow_queues.get(&shadow_light.light_id) else {
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

                    if let Some((index_buffer, index_format, count, _)) = &gpu_geometry.index_buffer {
                        pass.set_index_buffer(index_buffer.slice(..), *index_format);
                        pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                    } else {
                        pass.draw(gpu_geometry.draw_range.clone(), gpu_geometry.instance_range.clone());
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
