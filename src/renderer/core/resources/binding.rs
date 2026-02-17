//! `BindGroup` 相关操作
//!
//! 包括 Object `BindGroup` (Group 2)、骨骼管理和全局绑定 (Group 0)

use std::sync::atomic::{AtomicU32, Ordering};

use wgpu::ShaderStages;

use crate::Mesh;
use crate::assets::{AssetServer, TextureHandle};
use crate::resources::geometry::Geometry;
use crate::resources::texture::{SamplerSource, TextureSource};
use crate::resources::uniforms::DynamicModelUniforms;
use crate::resources::uniforms::WgslStruct;
use crate::scene::Scene;
use crate::scene::skeleton::Skeleton;

use crate::renderer::core::binding::{BindingResource, Bindings};
use crate::renderer::core::builder::{ResourceBuilder, WgslStructName};
use crate::renderer::graph::RenderState;

use super::{
    BindGroupContext, GpuBuffer, GpuGlobalState, ModelBufferAllocator, ObjectBindGroupKey,
    ResourceManager, generate_gpu_resource_id,
};

static NEXT_GLOBAL_STATE_ID: AtomicU32 = AtomicU32::new(0);

impl ResourceManager {
    // ========================================================================
    // 骨骼管理
    // ========================================================================

    /// 更新骨骼数据到 GPU
    pub fn prepare_skeleton(&mut self, skeleton: &Skeleton) {
        // 每帧强制上传 joint matrices 到 GPU
        let buffer_ref = skeleton.joint_matrices.handle();

        let buffer_guard = skeleton.joint_matrices.read();

        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            &buffer_ref,
            bytemuck::cast_slice(buffer_guard.as_slice()),
        );
    }

    /// 注册一个内部生成的纹理（如 Render Target）
    ///
    /// 这种纹理不需要从 CPU 上传，没有版本控制，由调用者保证其生命周期。
    /// 通常在 `RenderPass` 执行前调用。
    ///
    /// 适用于：Pass 内部私有的资源，Pass 自己持有并维护 id 的稳定性。
    /// 性能最高，无哈希查找。
    pub fn register_internal_texture_direct(&mut self, id: u64, view: wgpu::TextureView) {
        self.internal_resources.insert(id, view);
    }

    /// 适用于：跨 Pass 共享的资源（如 "`SceneColor`"）。
    /// 内部维护 Name -> ID 的映射。
    pub fn register_internal_texture_by_name(
        &mut self,
        name: &str,
        view: wgpu::TextureView,
    ) -> u64 {
        // 1. 查找或创建 ID (只在第一次遇到该名字时会有 String 分配)
        let id = *self
            .internal_name_lookup
            .entry(name.to_string())
            .or_insert_with(generate_gpu_resource_id);

        // 2. 注册
        self.register_internal_texture_direct(id, view);

        id
    }

    pub fn register_internal_texture(&mut self, view: wgpu::TextureView) -> u64 {
        let id = generate_gpu_resource_id();
        self.internal_resources.insert(id, view);

        id
    }

    pub fn release_internal_texture(&mut self, id: u64) {
        self.internal_resources.remove(&id);
        log::debug!("Released internal texture: {id}");
    }

    /// 统一获取 `TextureView` 的辅助方法
    ///
    /// 优先查找 Asset 转换的纹理，其次查找注册的内部纹理，最后返回 Dummy
    pub fn get_texture_view<'a>(&'a self, source: &TextureSource) -> &'a wgpu::TextureView {
        match source {
            TextureSource::Asset(handle) => {
                // 特殊处理 Dummy Env Map
                if *handle == TextureHandle::dummy_env_map() {
                    return &self.dummy_env_image.default_view;
                }

                // 查找 Asset 对应的 GPU 资源
                if let Some(binding) = self.texture_bindings.get(*handle)
                    && let Some(img) = self.gpu_images.get(&binding.cpu_image_id)
                {
                    return &img.default_view;
                }

                // Fallback
                &self.dummy_image.default_view
            }
            TextureSource::Attachment(id, _) => {
                // 直接查找内部资源表
                self.internal_resources
                    .get(id)
                    .unwrap_or(&self.dummy_image.default_view)
            }
        }
    }

    // ========================================================================
    // 统一的 prepare_mesh 入口
    // ========================================================================

    /// 准备 Mesh 的基础资源
    ///
    /// 采用 "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" 模式
    pub fn prepare_mesh(
        &mut self,
        assets: &AssetServer,
        mesh: &mut Mesh,
        skeleton: Option<&Skeleton>,
    ) -> Option<BindGroupContext> {
        // === Ensure 阶段: 确保所有资源已上传 ===
        // 如果 Allocator 本帧发生了扩容，ID 会改变，必须在此处注册新 ID

        if self.model_allocator.last_ensure_frame != self.frame_index {
            // self.ensure_buffer(self.model_allocator.cpu_buffer());
            let buffer_ref = self.model_allocator.buffer_handle();

            {
                let cpu_buffer = self.model_allocator.cpu_buffer();
                let guard = cpu_buffer.read();
                let data = guard.as_slice();
                Self::write_buffer_internal(
                    &self.device,
                    &self.queue,
                    &mut self.gpu_buffers,
                    self.frame_index,
                    &buffer_ref,
                    bytemuck::cast_slice(data),
                );
            }

            self.model_allocator.last_ensure_frame = self.frame_index;
        }

        mesh.update_morph_uniforms();
        let morph_result = self.ensure_buffer(&mesh.morph_uniforms);
        self.prepare_geometry(assets, mesh.geometry);
        self.prepare_material(assets, mesh.material);

        let geometry = assets.geometries.get(mesh.geometry)?;

        // === Collect 阶段: 收集所有资源 ID ===
        let mut current_ids = super::ResourceIdSet::with_capacity(4);
        current_ids.push(self.model_allocator.buffer_id());
        current_ids.push(morph_result.resource_id);
        current_ids.push_optional(skeleton.map(|s| s.joint_matrices.handle().id));

        let cache_key = current_ids.hash_value();

        // 检查全局缓存
        if let Some(binding_data) = self.object_bind_group_cache.get(&cache_key) {
            return Some(binding_data.clone());
        }

        // 创建新 GpuObject
        let binding_data =
            self.create_object_bind_group_internal(assets, &geometry, mesh, skeleton, cache_key);
        Some(binding_data)
    }

    fn create_object_bind_group_internal(
        &mut self,
        assets: &AssetServer,
        geometry: &Geometry,
        mesh: &Mesh,
        skeleton: Option<&Skeleton>,
        cache_key: ObjectBindGroupKey,
    ) -> BindGroupContext {
        let min_binding_size = ModelBufferAllocator::uniform_stride();
        let model_buffer_ref = self.model_allocator.cpu_buffer().handle().clone();

        let mut builder = ResourceBuilder::new();
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model",
            &model_buffer_ref,
            None,
            min_binding_size,
            ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        );
        mesh.define_bindings(&mut builder);
        geometry.define_bindings(&mut builder);

        if let Some(skeleton) = &skeleton {
            builder.add_storage_buffer(
                "skins",
                &skeleton.joint_matrices.handle(),
                None,
                true,
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into())),
            );
        }

        let binding_wgsl = builder.generate_wgsl(2);
        let layout_entries = builder.layout_entries.clone();

        let (layout, layout_id) = self.get_or_create_layout(&layout_entries);
        self.prepare_binding_resources(assets, &builder.resources);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder);

        let data = BindGroupContext {
            layout,
            layout_id,
            bind_group,
            bind_group_id,
            binding_wgsl: binding_wgsl.into(),
        };

        self.object_bind_group_cache.insert(cache_key, data.clone());
        self.bind_group_id_lookup
            .insert(bind_group_id, data.clone());
        data
    }

    // ========================================================================
    // BindGroup 通用操作
    // ========================================================================

    pub(crate) fn prepare_binding_resources(
        &mut self,
        assets: &AssetServer,
        resources: &[BindingResource],
    ) {
        for resource in resources {
            match resource {
                BindingResource::Buffer {
                    buffer: buffer_ref,
                    offset: _,
                    size: _,
                    data,
                } => {
                    let id = buffer_ref.id();
                    if let Some(bytes) = data {
                        let gpu_buf = self.gpu_buffers.entry(id).or_insert_with(|| {
                            let mut buf = GpuBuffer::new(
                                &self.device,
                                bytes,
                                buffer_ref.usage,
                                buffer_ref.label(),
                            );
                            buf.last_uploaded_version = buffer_ref.version;
                            buf
                        });

                        if buffer_ref.version > gpu_buf.last_uploaded_version {
                            if bytes.len() as u64 > gpu_buf.size {
                                log::debug!(
                                    "Recreating buffer {:?} due to size increase.",
                                    buffer_ref.label()
                                );
                                *gpu_buf = GpuBuffer::new(
                                    &self.device,
                                    bytes,
                                    buffer_ref.usage,
                                    buffer_ref.label(),
                                );
                            } else {
                                self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                            }
                            gpu_buf.last_uploaded_version = buffer_ref.version;
                        }
                        gpu_buf.last_used_frame = self.frame_index;
                    } else if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
                        gpu_buf.last_used_frame = self.frame_index;
                    } else {
                        panic!(
                            "ResourceManager: Trying to bind buffer {:?} (ID: {}) but it is not initialized!",
                            buffer_ref.label(),
                            id
                        );
                    }
                }
                BindingResource::Texture(Some(source)) => {
                    match source {
                        // 只有 Asset 类型的纹理需要 Prepare (上传/更新)
                        TextureSource::Asset(handle) => {
                            self.prepare_texture(assets, *handle);
                        }
                        // Attachment 类型是 GPU 内部生成的，无需 CPU->GPU 上传
                        TextureSource::Attachment(_, _) => {
                            // Do nothing
                        }
                    }
                }

                BindingResource::Sampler(Some(source)) => {
                    match source {
                        SamplerSource::FromTexture(_handle) => {
                            // 应该在 prepare_texture 阶段已经准备好了
                        }
                        SamplerSource::Asset(handle) => {
                            self.prepare_sampler(assets, *handle);
                        }
                        SamplerSource::Default => {
                            // Do nothing
                        }
                    }
                }
                BindingResource::Texture(None)
                | BindingResource::Sampler(None)
                | BindingResource::_Phantom(_) => {}
            }
        }
    }

    pub fn get_or_create_layout(
        &mut self,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::BindGroupLayout, u64) {
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cached BindGroupLayout"),
                entries,
            });

        let id = generate_gpu_resource_id();
        self.layout_cache
            .insert(entries.to_vec(), (layout.clone(), id));
        (layout, id)
    }

    #[allow(clippy::too_many_lines)]
    pub fn create_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        builder: &ResourceBuilder,
    ) -> (wgpu::BindGroup, u64) {
        let mut entries = Vec::new();

        let resources = &builder.resources;
        let layout_entries = &builder.layout_entries;

        for (i, resource_data) in resources.iter().enumerate() {
            let binding_resource = match resource_data {
                BindingResource::Buffer {
                    buffer,
                    data: _,
                    offset,
                    size,
                } => {
                    let cpu_id = buffer.id();
                    let gpu_buf = self
                        .gpu_buffers
                        .get(&cpu_id)
                        .expect("Buffer should be prepared");
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new),
                    })
                }
                BindingResource::Texture(source_opt) => {
                    let view = if let Some(source) = source_opt {
                        self.get_texture_view(source)
                    } else {
                        match layout_entries[i].ty {
                            wgpu::BindingType::Texture {
                                view_dimension,
                                sample_type: _,
                                multisampled: _,
                            } => {
                                match view_dimension {
                                    wgpu::TextureViewDimension::D2 => {
                                        &self.dummy_image.default_view
                                    }
                                    wgpu::TextureViewDimension::D2Array => {
                                        &self.dummy_shadow_map.default_view
                                    }
                                    wgpu::TextureViewDimension::Cube => {
                                        &self.dummy_env_image.default_view
                                    }
                                    // Todo: 支持更多维度
                                    _ => &self.dummy_image.default_view,
                                }
                            }

                            // wgpu::BindingType::StorageTexture { .. } => &self.dummy_storage_image.default_view,
                            _ => unreachable!("Unexpected binding type for Texture without source"),
                        }
                    };
                    wgpu::BindingResource::TextureView(view)
                }
                BindingResource::Sampler(source_opt) => {
                    // 从 TextureBinding 获取 sampler_id，然后从 sampler_id_lookup 快速查找
                    let sampler = if let Some(source) = source_opt {
                        match source {
                            // 情况 1: 跟随 Texture Asset (默认)
                            SamplerSource::FromTexture(handle) => {
                                if let Some(binding) = self.texture_bindings.get(*handle) {
                                    self.sampler_id_lookup
                                        .get(&binding.sampler_id)
                                        .unwrap_or(&self.dummy_sampler.sampler)
                                } else {
                                    &self.dummy_sampler.sampler
                                }
                            }
                            // 情况 2: 显式 Sampler Asset
                            SamplerSource::Asset(handle) => {
                                // 这里需要从 sampler_bindings 查找 GPU ID
                                if let Some(id) = self.sampler_bindings.get(*handle) {
                                    self.sampler_id_lookup
                                        .get(id)
                                        .unwrap_or(&self.dummy_sampler.sampler)
                                } else {
                                    // 如果尚未 prepare，理论上这不应该发生（前提是 prepare 阶段做好了），
                                    // 但为了安全回退到 dummy
                                    &self.dummy_sampler.sampler
                                }
                            }
                            // 情况 3: 默认采样器 (用于 Render Target)
                            SamplerSource::Default => {
                                if matches!(
                                    layout_entries[i].ty,
                                    wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Comparison
                                    )
                                ) {
                                    &self.shadow_compare_sampler.sampler
                                } else {
                                    &self.dummy_sampler.sampler
                                }
                            }
                        }
                    } else if matches!(
                        layout_entries[i].ty,
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
                    ) {
                        &self.shadow_compare_sampler.sampler
                    } else {
                        &self.dummy_sampler.sampler
                    };
                    wgpu::BindingResource::Sampler(sampler)
                }
                BindingResource::_Phantom(_) => unreachable!("_Phantom should never be used"),
            };

            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding_resource,
            });
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
        render_state: &RenderState,
    ) -> u32 {
        // === Ensure: upload all buffers, obtain physical resource IDs ===
        let camera_result = self.ensure_buffer(render_state.uniforms());
        let env_result = self.ensure_buffer(&scene.uniforms_buffer);
        let light_result = self.ensure_buffer(&scene.light_storage_buffer);
        let scene_uniform_result = self.ensure_buffer(&scene.uniforms_buffer);

        // Resolve environment texture IDs from GpuEnvironment cache.
        // resolve_gpu_environment runs before prepare_global and always creates
        // cache entries, so a miss here should not happen in normal operation.
        let (processed_env_map_id, pmrem_map_id) =
            if let Some(source) = scene.environment.source_env_map {
                if let Some(gpu_env) = self.environment_map_cache.get(&source) {
                    (gpu_env.cube_view_id, gpu_env.pmrem_view_id)
                } else {
                    log::warn!("GpuEnvironment cache miss in prepare_global");
                    (self.dummy_env_image.id, self.dummy_env_image.id)
                }
            } else {
                (self.dummy_env_image.id, self.dummy_env_image.id)
            };

        let brdf_lut_id = self.brdf_lut_view_id.unwrap_or(self.dummy_image.id);
        let shadow_2d_id = self.shadow_2d_array_id.unwrap_or(self.dummy_shadow_map.id);

        // === Collect: gather all resource IDs ===
        let mut current_ids = super::ResourceIdSet::with_capacity(9);
        current_ids.push(camera_result.resource_id);
        current_ids.push(env_result.resource_id);
        current_ids.push(light_result.resource_id);
        current_ids.push(scene_uniform_result.resource_id);
        current_ids.push(processed_env_map_id);
        current_ids.push(pmrem_map_id);
        current_ids.push(brdf_lut_id);
        current_ids.push(shadow_2d_id);

        let state_id = Self::compute_global_state_key(render_state.id, scene.id);

        // === Check: fast fingerprint comparison ===
        if let Some(gpu_state) = self.global_states.get_mut(&state_id)
            && gpu_state.resource_ids.matches_slice(current_ids.as_slice())
        {
            gpu_state.last_used_frame = self.frame_index;
            return gpu_state.id;
        }

        // === Rebind: fingerprint mismatch, rebuild BindGroup ===
        self.create_global_state(assets, state_id, render_state, scene, current_ids)
    }

    #[inline]
    fn compute_global_state_key(render_state_id: u32, scene_id: u32) -> u64 {
        (u64::from(scene_id) << 32) | u64::from(render_state_id)
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

        // Build scene bindings (environment uniforms, lights, env textures)
        self.define_global_scene_bindings(&mut builder, scene);

        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &builder);

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

    /// Build the scene-level global bindings (Group 0, after `RenderState`).
    ///
    /// This replaces the old `Scene::define_bindings`, resolving environment
    /// textures from `ResourceManager`'s caches instead of `Environment`.
    fn define_global_scene_bindings<'a>(
        &self,
        builder: &mut ResourceBuilder<'a>,
        scene: &'a Scene,
    ) {
        use crate::renderer::core::builder::WgslStructName;
        use crate::resources::uniforms::{EnvironmentUniforms, GpuLightStorage};

        // Environment Uniforms
        builder.add_uniform_buffer(
            "environment",
            &scene.uniforms_buffer.handle(),
            None,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
            false,
            None,
            Some(WgslStructName::Generator(
                EnvironmentUniforms::wgsl_struct_def,
            )),
        );

        // Light Storage Buffer
        builder.add_storage_buffer(
            "lights",
            &scene.light_storage_buffer.handle(),
            None,
            true,
            wgpu::ShaderStages::FRAGMENT,
            Some(WgslStructName::Generator(GpuLightStorage::wgsl_struct_def)),
        );

        // Resolve env_map from GpuEnvironment cache
        let env_map_source = scene
            .environment
            .source_env_map
            .and_then(|src| self.environment_map_cache.get(&src))
            .map_or_else(
                || TextureHandle::dummy_env_map().into(),
                |gpu_env| {
                    TextureSource::Attachment(
                        gpu_env.cube_view_id,
                        wgpu::TextureViewDimension::Cube,
                    )
                },
            );

        builder.add_texture(
            "env_map",
            Some(env_map_source),
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::ShaderStages::FRAGMENT,
        );
        builder.add_sampler(
            "env_map",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT,
        );

        // Resolve pmrem_map from GpuEnvironment cache
        let pmrem_source = scene
            .environment
            .source_env_map
            .and_then(|src| self.environment_map_cache.get(&src))
            .map(|gpu_env| {
                TextureSource::Attachment(gpu_env.pmrem_view_id, wgpu::TextureViewDimension::Cube)
            });

        builder.add_texture(
            "pmrem_map",
            pmrem_source,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::ShaderStages::FRAGMENT,
        );
        builder.add_sampler(
            "pmrem_map",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT,
        );

        // Resolve brdf_lut from ResourceManager
        let brdf_lut_source = self
            .brdf_lut_view_id
            .map(|id| TextureSource::Attachment(id, wgpu::TextureViewDimension::D2));

        builder.add_texture(
            "brdf_lut",
            brdf_lut_source,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            wgpu::ShaderStages::FRAGMENT,
        );
        builder.add_sampler(
            "brdf_lut",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT,
        );

        let shadow_2d_source = self
            .shadow_2d_array_id
            .map(|id| TextureSource::Attachment(id, wgpu::TextureViewDimension::D2Array));

        builder.add_texture(
            "shadow_map_2d_array",
            shadow_2d_source,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
            wgpu::ShaderStages::FRAGMENT,
        );
        builder.add_sampler(
            "shadow_map_compare",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Comparison,
            wgpu::ShaderStages::FRAGMENT,
        );
    }

    pub fn get_global_state(&self, render_state_id: u32, scene_id: u32) -> Option<&GpuGlobalState> {
        let state_id = Self::compute_global_state_key(render_state_id, scene_id);
        self.global_states.get(&state_id)
    }
}
