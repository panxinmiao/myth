use glam::{Mat4, Vec3, Vec4};

use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::RenderContext;
use crate::scene::camera::RenderCamera;
use crate::scene::light::LightKind;

/// Maximum number of cascades per directional light.
const MAX_CASCADES: u32 = 4;

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

    // ========================================================================
    // Spot Light VP Matrix
    // ========================================================================

    fn build_spot_vp(position: Vec3, direction: Vec3, spot: &crate::scene::light::SpotLight) -> Mat4 {
        let safe_dir = if direction.length_squared() > 1e-6 {
            direction.normalize()
        } else {
            -Vec3::Z
        };
        let up = if safe_dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let view = Mat4::look_at_rh(position, position + safe_dir, up);
        let fov = (spot.outer_cone * 2.0).clamp(0.1, std::f32::consts::PI - 0.01);
        let far = spot.range.max(1.0);
        let proj = Mat4::perspective_rh(fov, 1.0, 0.1, far);
        proj * view
    }

    // ========================================================================
    // CSM: Cascade Split Computation (Practical Split Scheme)
    // ========================================================================

    /// Computes cascade split distances using the Practical Split Scheme.
    ///
    /// `lambda` blends between uniform (0.0) and logarithmic (1.0) distribution.
    /// Returns an array of far distances for each cascade (in view space).
    fn compute_cascade_splits(
        cascade_count: u32,
        near: f32,
        far: f32,
        lambda: f32,
    ) -> [f32; MAX_CASCADES as usize] {
        let mut splits = [0.0f32; MAX_CASCADES as usize];
        let n = cascade_count.min(MAX_CASCADES) as usize;

        for i in 0..n {
            let p = (i + 1) as f32 / n as f32;
            let log_split = near * (far / near).powf(p);
            let uni_split = near + (far - near) * p;
            splits[i] = lambda * log_split + (1.0 - lambda) * uni_split;
        }

        // Ensure the last split reaches the far plane
        if n > 0 {
            splits[n - 1] = far;
        }

        splits
    }

    // ========================================================================
    // CSM: Frustum Corners in World Space
    // ========================================================================

    /// Computes the 8 frustum corners of a view-space frustum slice in world space.
    fn compute_frustum_corners_world(
        camera: &RenderCamera,
        slice_near: f32,
        slice_far: f32,
    ) -> [Vec3; 8] {
        // Extract fov and aspect from the projection matrix.
        // For perspective_infinite_reverse_rh: proj[1][1] = 1/tan(fov/2),
        //                                     proj[0][0] = proj[1][1]/aspect
        let proj = camera.projection_matrix;
        let tan_half_fov = 1.0 / proj.y_axis.y;
        let aspect = proj.y_axis.y / proj.x_axis.x;

        let h_near = tan_half_fov * slice_near;
        let w_near = h_near * aspect;
        let h_far = tan_half_fov * slice_far;
        let w_far = h_far * aspect;

        // Corners in view space (RH: -Z is forward)
        let corners_view = [
            // Near face (z = -slice_near)
            Vec3::new(-w_near, -h_near, -slice_near),
            Vec3::new( w_near, -h_near, -slice_near),
            Vec3::new( w_near,  h_near, -slice_near),
            Vec3::new(-w_near,  h_near, -slice_near),
            // Far face (z = -slice_far)
            Vec3::new(-w_far, -h_far, -slice_far),
            Vec3::new( w_far, -h_far, -slice_far),
            Vec3::new( w_far,  h_far, -slice_far),
            Vec3::new(-w_far,  h_far, -slice_far),
        ];

        // Transform to world space using inverse view matrix
        let inv_view = camera.view_matrix.inverse();
        let mut corners_world = [Vec3::ZERO; 8];
        for (i, c) in corners_view.iter().enumerate() {
            corners_world[i] = inv_view.transform_point3(*c);
        }
        corners_world
    }

    // ========================================================================
    // CSM: Build Cascade VP Matrix
    // ========================================================================

    /// Builds an orthographic VP matrix for one cascade.
    ///
    /// Calculates the light-space AABB of the frustum slice,
    /// applies texel alignment to prevent shimmer when the camera moves.
    fn build_cascade_vp(
        light_direction: Vec3,
        frustum_corners: &[Vec3; 8],
        shadow_map_size: u32,
        caster_extension: f32,
    ) -> Mat4 {
        let safe_dir = if light_direction.length_squared() > 1e-6 {
            light_direction.normalize()
        } else {
            -Vec3::Z
        };

        // Compute frustum center
        let mut center = Vec3::ZERO;
        for c in frustum_corners {
            center += *c;
        }
        center /= 8.0;

        let up = if safe_dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let light_view = Mat4::look_at_rh(center - safe_dir, center, up);

        // Compute light-space AABB of frustum corners
        let mut ls_min = Vec3::splat(f32::MAX);
        let mut ls_max = Vec3::splat(f32::MIN);
        for c in frustum_corners {
            let ls = light_view.transform_point3(*c);
            ls_min = ls_min.min(ls);
            ls_max = ls_max.max(ls);
        }

        // Expand Z to include potential casters between camera and light.
        // In RH light view, ls_max.z is near (towards light), ls_min.z is far.
        let base_z_range = (ls_max.z - ls_min.z).max(1.0);
        let near_extension = caster_extension.max(base_z_range);
        let far_extension = base_z_range.max(50.0);
        ls_max.z += near_extension;
        ls_min.z -= far_extension;

        // Texel alignment: snap the ortho bounds to texel grid to prevent shimmer
        let world_units_per_texel_x = (ls_max.x - ls_min.x) / shadow_map_size as f32;
        let world_units_per_texel_y = (ls_max.y - ls_min.y) / shadow_map_size as f32;

        if world_units_per_texel_x > 0.0 {
            ls_min.x = (ls_min.x / world_units_per_texel_x).floor() * world_units_per_texel_x;
            ls_max.x = (ls_max.x / world_units_per_texel_x).ceil() * world_units_per_texel_x;
        }
        if world_units_per_texel_y > 0.0 {
            ls_min.y = (ls_min.y / world_units_per_texel_y).floor() * world_units_per_texel_y;
            ls_max.y = (ls_max.y / world_units_per_texel_y).ceil() * world_units_per_texel_y;
        }

        let proj = Mat4::orthographic_rh(
            ls_min.x, ls_max.x,
            ls_min.y, ls_max.y,
            -ls_max.z, -ls_min.z, // glam orthographic_rh: near/far are positive distances
        );

        proj * light_view
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

        // ============================================================
        // 1. Collect active shadow lights
        // ============================================================
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
            max_map_size = max_map_size.max(
                light.shadow.as_ref().map_or(1024, |s| s.map_size),
            );
            active_shadow_lights.push((light_index, light.clone()));
        }

        // ============================================================
        // 2. Count total shadow layers (Directional uses N cascades, Spot uses 1)
        // ============================================================
        let mut total_layers = 0u32;
        let mut light_layer_assignments: smallvec::SmallVec<[(u32, u32); 8]> = smallvec::SmallVec::new(); // (base_layer, cascade_count)

        for (_idx, light) in &active_shadow_lights {
            let cascade_count = match &light.kind {
                LightKind::Directional(_) => {
                    let cfg = light.shadow.as_ref().map_or(4, |s| s.cascade_count);
                    cfg.clamp(1, MAX_CASCADES)
                }
                _ => 1,
            };
            light_layer_assignments.push((total_layers, cascade_count));
            total_layers += cascade_count;
        }

        // ============================================================
        // 3. Ensure GPU resources
        // ============================================================
        ctx.resource_manager
            .ensure_shadow_maps(total_layers, max_map_size);
        self.ensure_light_uniform_capacity(
            &ctx.wgpu_ctx.device,
            &shadow_global_layout,
            total_layers.max(1),
        );
        self.recreate_light_bind_group(&ctx.wgpu_ctx.device, &shadow_global_layout);

        ctx.render_lists.shadow_lights.clear();

        // Estimate caster coverage from current shadow casters (world-space bounds).
        // Used to adaptively extend directional cascade near plane toward the light.
        let scene_caster_extent = {
            let geo_guard = ctx.assets.geometries.read_lock();
            let camera_pos: Vec3 = ctx.camera.position.to_array().into();
            let mut max_distance = 0.0f32;

            for item in &ctx.extracted_scene.shadow_caster_items {
                let Some(geometry) = geo_guard.map.get(item.geometry) else {
                    continue;
                };

                let (center_ws, radius_ws) = if let Some(bs) = geometry.bounding_sphere.read().as_ref() {
                    let sx = item.world_matrix.x_axis.truncate().length();
                    let sy = item.world_matrix.y_axis.truncate().length();
                    let sz = item.world_matrix.z_axis.truncate().length();
                    let uniform_scale = sx.max(sy).max(sz);
                    (
                        item.world_matrix.transform_point3(bs.center),
                        bs.radius * uniform_scale,
                    )
                } else if let Some(bb) = geometry.bounding_box.read().as_ref() {
                    let sx = item.world_matrix.x_axis.truncate().length();
                    let sy = item.world_matrix.y_axis.truncate().length();
                    let sz = item.world_matrix.z_axis.truncate().length();
                    let half = bb.size() * 0.5;
                    let scaled_half = Vec3::new(half.x * sx, half.y * sy, half.z * sz);
                    (
                        item.world_matrix.transform_point3(bb.center()),
                        scaled_half.length(),
                    )
                } else {
                    (item.world_matrix.w_axis.truncate(), 0.0)
                };

                let distance = camera_pos.distance(center_ws) + radius_ws;
                max_distance = max_distance.max(distance);
            }

            max_distance.max(50.0)
        };

        // ============================================================
        // 4. Compute VP matrices and write to GPU
        // ============================================================
        {
            let mut light_storage = ctx.scene.light_storage_buffer.write();
            for light in light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.shadow_matrices.0 = [Mat4::IDENTITY; 4];
                light.cascade_count = 0;
                light.cascade_splits = Vec4::ZERO;
            }

            if total_layers != 0 {
                let mut shadow_uniform_data =
                    vec![0u8; (self.light_uniform_stride as usize) * (total_layers as usize)];

                for (i, (light_buffer_index, light)) in active_shadow_lights.iter().enumerate() {
                    let (base_layer, cascade_count) = light_layer_assignments[i];
                    let shadow_cfg = light.shadow.clone().unwrap_or_default();
                    let map_size = shadow_cfg.map_size;

                    match &light.kind {
                        LightKind::Directional(_) => {
                            // CSM: compute cascade splits and VP matrices
                            let cam_near = ctx.camera.near.max(0.1);
                            let cam_far = if ctx.camera.far.is_finite() {
                                ctx.camera.far
                            } else {
                                shadow_cfg.max_shadow_distance
                            };
                            let shadow_far = shadow_cfg.max_shadow_distance.min(cam_far);
                            let caster_extension = scene_caster_extent.max(shadow_cfg.max_shadow_distance);

                            let splits = Self::compute_cascade_splits(
                                cascade_count,
                                cam_near,
                                shadow_far,
                                shadow_cfg.cascade_split_lambda,
                            );

                            let mut cascade_matrices = [Mat4::IDENTITY; MAX_CASCADES as usize];
                            let mut prev_split = cam_near;

                            for c in 0..cascade_count as usize {
                                let slice_near = prev_split;
                                let slice_far = splits[c];
                                prev_split = slice_far;

                                let corners = Self::compute_frustum_corners_world(
                                    ctx.camera,
                                    slice_near,
                                    slice_far,
                                );

                                let vp = Self::build_cascade_vp(
                                    light.direction,
                                    &corners,
                                    map_size,
                                    caster_extension,
                                );

                                cascade_matrices[c] = vp;

                                // Write to per-layer uniform buffer for the shadow depth pass
                                let layer_idx = (base_layer + c as u32) as usize;
                                let offset = layer_idx * self.light_uniform_stride as usize;
                                let bytes = bytemuck::bytes_of(&vp);
                                shadow_uniform_data[offset..offset + bytes.len()]
                                    .copy_from_slice(bytes);

                                // Each cascade layer gets a ShadowLightInstance
                                ctx.render_lists.shadow_lights.push(
                                    crate::renderer::graph::frame::ShadowLightInstance {
                                        light_id: light.id,
                                        layer_index: base_layer + c as u32,
                                        light_buffer_index: *light_buffer_index,
                                        light_view_projection: vp,
                                    },
                                );
                            }

                            // Write to light storage buffer
                            if let Some(gpu_light) = light_storage.get_mut(*light_buffer_index) {
                                gpu_light.shadow_layer_index = base_layer as i32;
                                gpu_light.shadow_matrices.0 = cascade_matrices;
                                gpu_light.cascade_count = cascade_count;
                                gpu_light.cascade_splits = Vec4::new(
                                    splits[0],
                                    splits[1.min(cascade_count as usize - 1)],
                                    splits[2.min(cascade_count as usize - 1)],
                                    splits[3.min(cascade_count as usize - 1)],
                                );
                                gpu_light.shadow_bias = shadow_cfg.bias;
                                gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                            }
                        }
                        LightKind::Spot(spot) => {
                            let vp = Self::build_spot_vp(light.position, light.direction, spot);

                            let layer_idx = base_layer as usize;
                            let offset = layer_idx * self.light_uniform_stride as usize;
                            let bytes = bytemuck::bytes_of(&vp);
                            shadow_uniform_data[offset..offset + bytes.len()]
                                .copy_from_slice(bytes);

                            if let Some(gpu_light) = light_storage.get_mut(*light_buffer_index) {
                                gpu_light.shadow_layer_index = base_layer as i32;
                                gpu_light.shadow_matrices.0[0] = vp;
                                gpu_light.cascade_count = 1;
                                gpu_light.shadow_bias = shadow_cfg.bias;
                                gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                            }

                            ctx.render_lists.shadow_lights.push(
                                crate::renderer::graph::frame::ShadowLightInstance {
                                    light_id: light.id,
                                    layer_index: base_layer,
                                    light_buffer_index: *light_buffer_index,
                                    light_view_projection: vp,
                                },
                            );
                        }
                        LightKind::Point(_) => {}
                    }
                }

                ctx.wgpu_ctx
                    .queue
                    .write_buffer(&self.light_uniform_buffer, 0, &shadow_uniform_data);
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
