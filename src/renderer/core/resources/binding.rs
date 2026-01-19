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
    generate_resource_id, ModelBufferAllocator,
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

    /// 准备 Mesh 的基础资源（几何体、材质、Morph Uniform Buffer）
    pub fn prepare_mesh(&mut self, assets: &AssetServer, mesh: &mut Mesh, skeleton: Option<&Skeleton>) -> Option<ObjectBindingData> {

        // 1. 数据同步 (Morph) - 必须每帧检查
        mesh.update_morph_uniforms();
        self.write_buffer(mesh.morph_uniforms.handle(), mesh.morph_uniforms.as_bytes());

        // 2. 准备子资源
        self.prepare_geometry(assets, mesh.geometry);
        self.prepare_material(assets, mesh.material);


        // 3. 获取用于校验的 Version 信息
        let geometry = assets.get_geometry(mesh.geometry)?;
        let material = assets.get_material(mesh.material)?;

        let geo_version = geometry.structure_version();
        let mat_version = material.binding_version();
        let current_model_buffer_id = self.model_allocator.buffer_id();


        let skeleton_buffer_id = skeleton.as_ref().map(|skel| skel.joint_matrices.handle().id);

        // =========================================================
        // [Fast Path] 快速路径：校验缓存
        // =========================================================
        if let Some(cached_bind_group_id) = mesh.render_cache.bind_group_id {
            if mesh.render_cache.is_valid(
                mesh.geometry, 
                geo_version, 
                mesh.material, 
                mat_version, 
                current_model_buffer_id,
                skeleton_buffer_id
            ) {
                if let Some(data) =  self.get_cached_bind_group(cached_bind_group_id){
                    return Some(data.clone());
                }
            }
        }

        // ======- 重建 BindGroup 路径 -======

        let model_buffer_id = self.model_allocator.buffer_id();
        let morph_buffer_id = Some(mesh.morph_uniforms.handle().id);

        let key = ObjectBindGroupKey {
            model_buffer_id,
            skeleton_buffer_id,
            morph_buffer_id,
        };


        // --- 3. 缓存命中检查 ---

        mesh.render_cache.geometry_id = Some(mesh.geometry);
        mesh.render_cache.geometry_version = geo_version;
        mesh.render_cache.material_id = Some(mesh.material);
        mesh.render_cache.material_version = mat_version;
        mesh.render_cache.model_buffer_id = current_model_buffer_id;
        mesh.render_cache.skeleton_id = skeleton_buffer_id;
    
        // Pipeline ID 也应该在这里被清理，因为 BindGroup 变了，Pipeline 可能也需要变
        mesh.render_cache.pipeline_id = None;
        
        // 检查全局缓存中是否已有对应的 BindGroup
        if let Some(binding_data) = self.object_bind_group_cache.get(&key) {
            mesh.render_cache.bind_group_id = Some(binding_data.bind_group_id);
            return Some(binding_data.clone());
        }

        // --- 4. 创建新 BindGroup (缓存未命中) ---
        let binding_data = self.create_object_bind_group_internal(assets, geometry, mesh, skeleton, key);
        mesh.render_cache.bind_group_id = Some(binding_data.bind_group_id);
        Some(binding_data)

    }


    /// 内部方法：创建 Object BindGroup 并写入缓存
    fn create_object_bind_group_internal(
        &mut self,
        assets: &AssetServer,
        geometry: &Geometry,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        key: ObjectBindGroupKey,
    ) -> ObjectBindingData {
        
        let min_binding_size = ModelBufferAllocator::uniform_stride();
        let model_buffer_ref = self.model_allocator.cpu_buffer().handle().clone();

        let mut builder = ResourceBuilder::new();

        // 1. Model Uniform
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model", 
            &model_buffer_ref, 
            None, 
            min_binding_size, 
            ShaderStages::VERTEX
        );

        // 2. Mesh Bindings (Morph)
        mesh.define_bindings(&mut builder);

        // 3. Geometry Bindings
        geometry.define_bindings(&mut builder);

        // 4. Skeleton Bindings
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
        
        let (layout, _layout_id) = self.get_or_create_layout(&layout_entries);
        
        // 确保所有依赖的 GPU 资源已就绪 (Double check)
        self.prepare_binding_resources(assets, &resources);
        
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &resources);

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl: binding_wgsl.into(),
        };

        // 写入全局缓存
        self.object_bind_group_cache.insert(key, data.clone());
        self.bind_group_id_lookup.insert(bind_group_id, data.clone());
        data
    }

    // ========================================================================
    // BindGroup 通用操作
    // ========================================================================

    pub(crate) fn prepare_binding_resources(&mut self, assets: &AssetServer, resources: &[BindingResource]) -> Vec<u64> {
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
                        if *handle == TextureHandle::dummy_env_map() {
                            &self.dummy_env_texture
                        } else {
                            self.gpu_textures.get(*handle).unwrap_or(&self.dummy_texture)
                        }
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

    // ========================================================================
    // 全局绑定 (Group 0)
    // ========================================================================

    /// 准备全局绑定资源
    /// 
    /// 从 Scene 收集 Camera、Lights、Environment 数据，创建 BindGroup 0
    /// 通过 Buffer 版本号追踪变化，避免每帧重复创建
    pub fn prepare_global(
        &mut self, 
        assets: &AssetServer, 
        scene: &Scene,
        render_state: &RenderState
    ) -> u32 {
        // 使用 render_state.id 作为缓存键
        let state_id = render_state.id as u64;
        
        // 获取当前版本信息
        let render_state_version = render_state.uniforms().version();
        let env_uniforms_version = scene.environment_uniforms().version();
        let light_storage_version = scene.light_storage().version();
        let env_map = scene.environment.env_map;
        
        // 检查是否需要更新
        if let Some(gpu_state) = self.global_states.get_mut(&state_id) {
            let render_state_match = render_state_version == gpu_state.last_render_state_version;
            let env_uniforms_match = env_uniforms_version == gpu_state.last_env_uniforms_version;
            let light_storage_match = light_storage_version == gpu_state.last_light_storage_version;
            let env_map_match = env_map == gpu_state.last_env_map;
            
            if render_state_match && env_uniforms_match && light_storage_match && env_map_match {
                gpu_state.last_used_frame = self.frame_index;
                return gpu_state.id;
            }
        }
        
        // 创建或更新 GpuGlobalState
        self.create_or_update_global_state(
            assets,
            state_id,
            render_state,
            scene,
        )
    }
    
    fn create_or_update_global_state(
        &mut self,
        assets: &AssetServer,
        state_id: u64,
        render_state: &RenderState,
        scene: &Scene,
    ) -> u32 {
        // 准备环境贴图
        if let Some(env_map) = scene.environment.env_map {
            self.prepare_texture(assets, env_map);
        }
        
        // 构建 BindGroup
        let mut builder = ResourceBuilder::new();
        
        // Binding 0: RenderState Uniforms (Camera)
        render_state.define_bindings(&mut builder);
        
        // Binding 1-4: Scene Bindings (Environment Uniforms, Lights, EnvMap)
        scene.define_bindings(&mut builder);
        
        // 准备资源并创建 BindGroup
        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(0);
        
        // 获取或创建新 ID
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
            binding_wgsl,
            last_render_state_version: render_state.uniforms().version(),
            last_env_uniforms_version: scene.environment_uniforms().version(),
            last_light_storage_version: scene.light_storage().version(),
            last_env_map: scene.environment.env_map,
            last_used_frame: self.frame_index,
        };
        
        self.global_states.insert(state_id, gpu_state);
        new_id
    }

    /// 获取全局状态
    pub fn get_global_state(&self, render_state_id: u32) -> Option<&GpuGlobalState> {
        self.global_states.get(&(render_state_id as u64))
    }
}
