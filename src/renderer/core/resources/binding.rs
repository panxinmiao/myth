//! BindGroup 相关操作
//!
//! 包括 Object BindGroup (Group 2)、骨骼管理和全局绑定 (Group 0)

use std::sync::atomic::{AtomicU32, Ordering};

use glam::Mat4;
use wgpu::ShaderStages;

use crate::Mesh;
use crate::assets::{AssetServer, TextureHandle};
use crate::resources::buffer::{BufferRef, CpuBuffer};
use crate::resources::geometry::Geometry;
use crate::resources::uniforms::{DynamicModelUniforms, WgslStruct};
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
    /// 从 Scene 收集 Camera、Lights、Environment 数据，管理 BindGroup 0
    /// 
    /// # 缓存策略
    /// 
    /// 1. **结构指纹匹配** → 复用现有 BindGroup，只上传数据变化
    /// 2. **结构指纹不匹配** → 重建 BindGroup
    /// 
    /// 结构指纹包括：
    /// - RenderState Buffer ID
    /// - GlobalResources 结构版本（Light Buffer 扩容）
    /// - Environment Map Handle
    /// 
    /// # 性能说明
    /// - 数据变化（灯光移动、相机变换）只触发 `write_buffer`
    /// - 结构变化（Buffer 扩容、贴图切换）才触发 BindGroup 重建
    pub fn prepare_global(
        &mut self, 
        assets: &AssetServer, 
        scene: &Scene,
        render_state: &RenderState
    ) -> u32 {
        // 1. 从 Scene 同步全局资源数据到 GlobalResources
        self.global_resources.sync_from_scene(scene);
        
        // 2. 计算当前结构指纹
        let render_state_buffer_id = render_state.uniforms().handle().id;
        let global_structure_version = self.global_resources.structure_version();
        let env_map = scene.environment.env_map;
        
        // 使用 render_state.id 作为缓存键
        let state_id = render_state.id as u64;
        
        // 3. 检查是否有缓存的 GpuGlobalState，并判断是否需要重建
        let cached_info = self.global_states.get(&state_id).map(|gpu_state| {
            let structure_match = 
                gpu_state.render_state_buffer_id == render_state_buffer_id &&
                gpu_state.global_structure_version == global_structure_version &&
                gpu_state.env_map == env_map;
            
            (
                gpu_state.id,
                structure_match,
                gpu_state.last_render_state_data_version,
                gpu_state.last_global_data_version,
                gpu_state.env_buffer_id,
                gpu_state.light_buffer_id,
            )
        });
        
        if let Some((id, structure_match, last_rs_ver, last_global_ver, env_buf_id, light_buf_id)) = cached_info {
            if structure_match {
                self.upload_global_data(
                    render_state, 
                    state_id,
                    last_rs_ver, 
                    last_global_ver, 
                    env_buf_id, 
                    light_buf_id
                );
                
                if let Some(gpu_state) = self.global_states.get_mut(&state_id) {
                    gpu_state.last_used_frame = self.frame_index;
                }
                return id;
            }
        }

        self.create_global_state(assets, state_id, render_state, scene)
    }
    
    /// 仅上传数据变化（不重建 BindGroup）
    fn upload_global_data(
        &mut self, 
        render_state: &RenderState,
        state_id: u64,
        last_render_state_data_version: u64,
        last_global_data_version: u64,
        env_buffer_id: u64,
        light_buffer_id: u64,
    ) {
        let frame_index = self.frame_index;
        
        // 上传 RenderState 数据（Camera）
        let render_state_data_version = render_state.uniforms().version();
        if render_state_data_version != last_render_state_data_version {
            self.write_buffer(render_state.uniforms().handle(), render_state.uniforms().as_bytes());
            if let Some(gpu_state) = self.global_states.get_mut(&state_id) {
                gpu_state.last_render_state_data_version = render_state_data_version;
            }
        }
        
        // 上传 GlobalResources 数据（Light/Env）
        let global_data_version = self.global_resources.data_version();
        if global_data_version != last_global_data_version {
            // 上传 Environment Uniforms
            let env_bytes = self.global_resources.environment_uniforms_bytes();
            if let Some(gpu_buffer) = self.gpu_buffers.get_mut(&env_buffer_id) {
                self.queue.write_buffer(&gpu_buffer.buffer, 0, env_bytes);
                gpu_buffer.last_used_frame = frame_index;
            }
            
            // 上传 Light Storage
            let light_bytes = self.global_resources.light_data_bytes();
            if let Some(gpu_buffer) = self.gpu_buffers.get_mut(&light_buffer_id) {
                self.queue.write_buffer(&gpu_buffer.buffer, 0, light_bytes);
                gpu_buffer.last_used_frame = frame_index;
            }
            
            if let Some(gpu_state) = self.global_states.get_mut(&state_id) {
                gpu_state.last_global_data_version = global_data_version;
            }
        }
    }
    
    /// 创建新的 GpuGlobalState（重建 BindGroup）
    fn create_global_state(
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
        
        // 1. 确保 Buffer 存在并上传数据
        let (env_buffer_ref, env_buffer_id) = self.ensure_global_env_buffer();
        let (light_buffer_ref, light_buffer_id) = self.ensure_global_light_buffer();
        
        // 2. 构建 BindGroup

        let mut builder = ResourceBuilder::new();
        // Binding 0: RenderState Uniforms (Camera)
        render_state.define_bindings(&mut builder);

        
        // Binding 1: Environment Uniforms
        builder.add_uniform_buffer(
            "environment",
            &env_buffer_ref,
            None,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
            false,
            None,
            Some(crate::renderer::core::builder::WgslStructName::Generator(
                crate::resources::uniforms::EnvironmentUniforms::wgsl_struct_def
            ))
        );
        
        // Binding 2: Light Storage Buffer
        builder.add_storage_buffer(
            "lights",
            &light_buffer_ref,
            None,
            true,
            wgpu::ShaderStages::FRAGMENT,
            Some(crate::renderer::core::builder::WgslStructName::Generator(
                crate::resources::uniforms::GpuLightStorage::wgsl_struct_def
            ))
        );
        
        // Binding 3-4: Environment Map (Cube) and Sampler
        scene.define_bindings(&mut builder);

                    // 3. 准备资源并创建 BindGroup
        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder.resources);
        
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
            binding_wgsl: builder.generate_wgsl(0),
            // 结构指纹
            render_state_buffer_id: render_state.uniforms().handle().id,
            global_structure_version: self.global_resources.structure_version(),
            env_map: scene.environment.env_map,
            env_buffer_id,
            light_buffer_id,
            // 数据版本
            last_render_state_data_version: render_state.uniforms().version(),
            last_global_data_version: self.global_resources.data_version(),
            last_used_frame: self.frame_index,
        };
        
        self.global_states.insert(state_id, gpu_state);
        new_id
    }

    /// 获取全局状态
    pub fn get_global_state(&self, render_state_id: u32) -> Option<&GpuGlobalState> {
        self.global_states.get(&(render_state_id as u64))
    }
    
    // ========================================================================
    // 全局资源 Buffer 管理
    // ========================================================================
    
    /// 确保 Environment Uniform Buffer 存在并返回 (BufferRef, buffer_id)
    fn ensure_global_env_buffer(&mut self) -> (BufferRef, u64) {
        const ENV_BUFFER_ID: u64 = u64::MAX - 1;
        
        let env_bytes = self.global_resources.environment_uniforms_bytes();
        
        if !self.gpu_buffers.contains_key(&ENV_BUFFER_ID) {
            let gpu_buffer = GpuBuffer::new(
                &self.device,
                env_bytes,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("Global Environment Uniforms")
            );
            self.gpu_buffers.insert(ENV_BUFFER_ID, gpu_buffer);
        } else {
            let gpu_buffer = self.gpu_buffers.get_mut(&ENV_BUFFER_ID).unwrap();
            self.queue.write_buffer(&gpu_buffer.buffer, 0, env_bytes);
            gpu_buffer.last_used_frame = self.frame_index;
        }
        
        let buffer_ref = BufferRef::with_fixed_id(
            ENV_BUFFER_ID,
            env_bytes.len(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            self.global_resources.data_version(),
            Some("Global Environment Uniforms")
        );
        
        (buffer_ref, ENV_BUFFER_ID)
    }
    
    /// 确保 Light Storage Buffer 存在并返回 (BufferRef, buffer_id)
    /// 
    /// 当需要扩容时，会销毁旧 Buffer 并创建新的
    fn ensure_global_light_buffer(&mut self) -> (BufferRef, u64) {
        const LIGHT_BUFFER_ID: u64 = u64::MAX - 2;
        
        let light_bytes = self.global_resources.light_data_bytes();
        let required_size = light_bytes.len() as u64;
        
        let need_recreate = if let Some(gpu_buffer) = self.gpu_buffers.get(&LIGHT_BUFFER_ID) {
            gpu_buffer.size < required_size
        } else {
            true
        };
        
        if need_recreate {
            // 创建新的 GPU Buffer（可能需要扩容）
            let gpu_buffer = GpuBuffer::new(
                &self.device,
                light_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Global Light Storage")
            );
            self.gpu_buffers.insert(LIGHT_BUFFER_ID, gpu_buffer);
        } else {
            let gpu_buffer = self.gpu_buffers.get_mut(&LIGHT_BUFFER_ID).unwrap();
            self.queue.write_buffer(&gpu_buffer.buffer, 0, light_bytes);
            gpu_buffer.last_used_frame = self.frame_index;
        }
        
        let buffer_ref = BufferRef::with_fixed_id(
            LIGHT_BUFFER_ID,
            light_bytes.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            self.global_resources.data_version(),
            Some("Global Light Storage")
        );
        
        (buffer_ref, LIGHT_BUFFER_ID)
    }
}
