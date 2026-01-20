//! BindGroup 相关操作
//!
//! 包括 Object BindGroup (Group 2)、骨骼管理和全局绑定 (Group 0)

use std::sync::atomic::{AtomicU32, Ordering};

use glam::Mat4;
use wgpu::ShaderStages;

use crate::Mesh;
use crate::assets::{AssetServer, TextureHandle};
use crate::resources::buffer::CpuBuffer;
use crate::resources::geometry::Geometry;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::scene::SkeletonKey;
use crate::scene::skeleton::{Skeleton};
use crate::scene::Scene;

use crate::renderer::core::binding::{BindingResource, Bindings};
use crate::renderer::core::builder::{ResourceBuilder, WgslStructName};
use crate::renderer::graph::RenderState;

use super::{
    ResourceManager, GpuBuffer, GpuGlobalState, 
    ObjectBindGroupKey, ObjectBindingData,
    generate_gpu_resource_id, ModelBufferAllocator,
};

static NEXT_GLOBAL_STATE_ID: AtomicU32 = AtomicU32::new(0);

impl ResourceManager {
    // ========================================================================
    // 骨骼管理
    // ========================================================================

    /// 更新骨骼数据到 GPU
    pub fn prepare_skeleton(&mut self, _skeleton_id: SkeletonKey, skeleton: &Skeleton) {
        // 每帧强制上传 joint matrices 到 GPU
        let buffer_ref = skeleton.joint_matrices.handle();

        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            buffer_ref,
            bytemuck::cast_slice(skeleton.joint_matrices.as_slice())
        );
    }

    /// 获取骨骼 Buffer
    pub fn get_skeleton_buffer(&self, skeleton_id: SkeletonKey) -> Option<&CpuBuffer<Vec<Mat4>>> {
        self.skeleton_buffers.get(&skeleton_id)
    }

    // ========================================================================
    // 统一的 prepare_mesh 入口
    // ========================================================================

    /// 准备 Mesh 的基础资源
    /// 
    /// 采用 "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" 模式
    pub fn prepare_mesh(&mut self, assets: &AssetServer, mesh: &mut Mesh, skeleton: Option<&Skeleton>) -> Option<ObjectBindingData> {
        // === Ensure 阶段: 确保所有资源已上传 ===
        mesh.update_morph_uniforms();
        let morph_result = self.ensure_buffer(&mesh.morph_uniforms);
        self.prepare_geometry(assets, mesh.geometry);
        let mat_prep_result = self.prepare_material(assets, mesh.material)?;
        
        let geometry = assets.get_geometry(mesh.geometry)?;
        
        // === Collect 阶段: 收集所有资源 ID ===
        let mut current_ids = super::ResourceIdSet::with_capacity(4);
        current_ids.push(self.model_allocator.buffer_id());
        current_ids.push(morph_result.resource_id);
        current_ids.push_optional(skeleton.map(|s| s.joint_matrices.handle().id));
        current_ids.push(mat_prep_result.layout_id);
        
        // === Check 阶段: 快速指纹比较 ===
        if mesh.render_cache.fingerprint_matches(&current_ids) {
            if let Some(cached_id) = mesh.render_cache.bind_group_id {
                if let Some(data) = self.get_cached_bind_group(cached_id) {
                    return Some(data.clone());
                }
            }
        }
        
        // === Rebind 阶段: 指纹不匹配，重建 BindGroup ===
        let cache_key = current_ids.hash_value();
        
        // 检查全局缓存
        if let Some(binding_data) = self.object_bind_group_cache.get(&cache_key) {
            mesh.render_cache.bind_group_id = Some(binding_data.bind_group_id);
            mesh.render_cache.resource_ids = current_ids;
            mesh.render_cache.geometry_id = Some(mesh.geometry);
            mesh.render_cache.material_id = Some(mesh.material);
            mesh.render_cache.pipeline_id = None;
            return Some(binding_data.clone());
        }
        
        // 创建新 BindGroup
        let binding_data = self.create_object_bind_group_internal(assets, geometry, mesh, skeleton, cache_key);
        mesh.render_cache.bind_group_id = Some(binding_data.bind_group_id);
        mesh.render_cache.resource_ids = current_ids;
        mesh.render_cache.geometry_id = Some(mesh.geometry);
        mesh.render_cache.material_id = Some(mesh.material);
        mesh.render_cache.pipeline_id = None;
        Some(binding_data)
    }

    fn create_object_bind_group_internal(
        &mut self,
        assets: &AssetServer,
        geometry: &Geometry,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        cache_key: ObjectBindGroupKey,
    ) -> ObjectBindingData {
        let min_binding_size = ModelBufferAllocator::uniform_stride();
        let model_buffer_ref = self.model_allocator.cpu_buffer().handle().clone();

        let mut builder = ResourceBuilder::new();
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model", 
            &model_buffer_ref, 
            None, 
            min_binding_size, 
            ShaderStages::VERTEX
        );
        mesh.define_bindings(&mut builder);
        geometry.define_bindings(&mut builder);
        
        if let Some(skeleton) = &skeleton {
            builder.add_storage_buffer(
                "skins", 
                skeleton.joint_matrices.handle(), 
                None, 
                true, 
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into()))
            );
        }

        let binding_wgsl = builder.generate_wgsl(2);
        let layout_entries = builder.layout_entries.clone();
        let resources = std::mem::take(&mut builder.resources);
        
        let (layout, _) = self.get_or_create_layout(&layout_entries);
        self.prepare_binding_resources(assets, &resources);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &resources);

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl: binding_wgsl.into(),
        };

        self.object_bind_group_cache.insert(cache_key, data.clone());
        self.bind_group_id_lookup.insert(bind_group_id, data.clone());
        data
    }

    // ========================================================================
    // BindGroup 通用操作
    // ========================================================================

    pub(crate) fn prepare_binding_resources(&mut self, assets: &AssetServer, resources: &[BindingResource]) {
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
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(handle) = handle_opt {
                        self.prepare_texture(assets, *handle);
                    }
                },
                _ => {}
            }
        }
    }

    pub fn get_or_create_layout(&mut self, entries: &[wgpu::BindGroupLayoutEntry]) -> (wgpu::BindGroupLayout, u64) {
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        let id = generate_gpu_resource_id();
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
                    let view = if let Some(handle) = handle_opt {
                        if *handle == TextureHandle::dummy_env_map() {
                            &self.dummy_env_image.default_view
                        } else if let Some(binding) = self.texture_bindings.get(*handle) {
                            self.gpu_images.values()
                                .find(|img| img.id == binding.image_id)
                                .map(|img| &img.default_view)
                                .unwrap_or(&self.dummy_image.default_view)
                        } else {
                            &self.dummy_image.default_view
                        }
                    } else { &self.dummy_image.default_view };
                    wgpu::BindingResource::TextureView(view)
                },
                BindingResource::Sampler(handle_opt) => {
                    // 从 TextureBinding 获取 sampler_id，然后从 sampler_id_lookup 快速查找
                    let sampler = if let Some(handle) = handle_opt {
                        if let Some(binding) = self.texture_bindings.get(*handle) {
                            self.sampler_id_lookup
                                .get(&binding.sampler_id)
                                .unwrap_or(&self.dummy_sampler.sampler)
                        } else {
                            &self.dummy_sampler.sampler
                        }
                    } else { &self.dummy_sampler.sampler };
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

        (bind_group, generate_gpu_resource_id())
    }

    // ========================================================================
    // 全局绑定 (Group 0)
    // ========================================================================

    /// 准备全局绑定资源
    /// 
    /// 采用 "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" 模式
    pub fn prepare_global(
        &mut self, 
        assets: &AssetServer, 
        scene: &Scene,
        render_state: &RenderState
    ) -> u32 {
        // === Ensure 阶段: 确保所有 Buffer 已上传，获取物理资源 ID ===
        let camera_result = self.ensure_buffer(render_state.uniforms());
        let env_result = self.ensure_buffer(&scene.uniforms_buffer);
        let light_result = self.ensure_buffer(&scene.light_storage_buffer);

        // 环境贴图 ID
        let env_map_id = scene.environment.env_map.map(|h| {
            self.prepare_texture(assets, h);
            self.texture_bindings.get(h).map(|b| b.image_id).unwrap_or(0)
        }).unwrap_or(0);
        
        // === Collect 阶段: 收集所有资源 ID ===
        let mut current_ids = super::ResourceIdSet::with_capacity(4);
        current_ids.push(camera_result.resource_id);
        current_ids.push(env_result.resource_id);
        current_ids.push(light_result.resource_id);
        current_ids.push(env_map_id);
        
        // 使用 (render_state.id, light_buffer_id) 组合作为缓存键，支持多场景并发渲染
        let state_id = Self::compute_global_state_key(render_state.id, scene.id);
        
        // === Check 阶段: 快速指纹比较 ===
        if let Some(gpu_state) = self.global_states.get_mut(&state_id) {
            if gpu_state.resource_ids == current_ids {
                gpu_state.last_used_frame = self.frame_index;
                return gpu_state.id;
            }
        }
        
        // === Rebind 阶段: 指纹不匹配，重建 BindGroup ===
        self.create_global_state(assets, state_id, render_state, scene, current_ids)
    }
    
    #[inline]
    fn compute_global_state_key(render_state_id: u32, scene_id: u32) -> u64 {
        ((scene_id as u64) << 32) | (render_state_id as u64)
    }
    
    fn create_global_state(
        &mut self,
        assets: &AssetServer,
        state_id: u64,
        render_state: &RenderState,
        scene: &Scene,
        resource_ids: super::ResourceIdSet,
    ) -> u32 {
        let mut builder = ResourceBuilder::new();
        render_state.define_bindings(&mut builder);
        scene.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder.resources);
        
        let new_id = if let Some(existing) = self.global_states.get(&state_id) {
            existing.id
        } else {
            NEXT_GLOBAL_STATE_ID.fetch_add(1, Ordering::Relaxed)
        };
        
        let gpu_state = GpuGlobalState {
            id: new_id,
            bind_group,
            bind_group_id,
            layout,
            layout_id,
            binding_wgsl: builder.generate_wgsl(0),
            resource_ids,
            last_used_frame: self.frame_index,
        };
        
        self.global_states.insert(state_id, gpu_state);
        new_id
    }

    /// 获取全局状态
    pub fn get_global_state(&self, render_state_id: u32, scene_id: u32) -> Option<&GpuGlobalState> {
        let state_id = Self::compute_global_state_key(render_state_id, scene_id);
        self.global_states.get(&state_id)
    }
    
    // 根据 render_state_id 和 scene 获取全局状态
    // pub fn get_global_state_for_scene(&self, render_state_id: u32, scene: &Scene) -> Option<&GpuGlobalState> {
    //     let state_id = Self::compute_global_state_key(render_state_id, scene.light_storage_buffer.handle().id);
    //     self.global_states.get(&state_id)
    // }
}
