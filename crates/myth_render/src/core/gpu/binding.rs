//! `BindGroup` operations
//!
//! Includes Object `BindGroup` (Group 2), skeleton management, and global bindings (Group 0)

use std::sync::atomic::{AtomicU32, Ordering};

use wgpu::ShaderStages;

use myth_assets::{AssetServer, TextureHandle};
use myth_resources::Mesh;
use myth_resources::geometry::Geometry;
use myth_resources::texture::TextureSource;
use myth_resources::uniforms::DynamicModelUniforms;
use myth_resources::uniforms::WgslStruct;
use myth_scene::Scene;
use myth_scene::skeleton::Skeleton;

use crate::core::binding::Bindings;
use crate::graph::RenderState;
use myth_resources::builder::{Binding, BindingDesc};
use myth_resources::{BindingResource, ResourceBuilder, WgslStructName};

use super::{
    BindGroupContext, GpuBuffer, GpuGlobalState, ModelBufferAllocator, ObjectBindGroupKey,
    ResourceManager, generate_gpu_resource_id,
};

static NEXT_GLOBAL_STATE_ID: AtomicU32 = AtomicU32::new(0);

impl ResourceManager {
    // ========================================================================
    // Skeleton management
    // ========================================================================

    /// Upload skeleton data (current and previous frame joints) to GPU.
    pub fn prepare_skeleton(&mut self, skeleton: &Skeleton) {
        let buffer_ref = skeleton.joint_matrices.handle();
        let buffer_guard = skeleton.joint_matrices.read();
        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            &mut self.buffer_index,
            self.frame_index,
            &buffer_ref,
            bytemuck::cast_slice(buffer_guard.as_slice()),
        );

        let prev_buffer_ref = skeleton.prev_joint_matrices.handle();
        let prev_buffer_guard = skeleton.prev_joint_matrices.read();
        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            &mut self.buffer_index,
            self.frame_index,
            &prev_buffer_ref,
            bytemuck::cast_slice(prev_buffer_guard.as_slice()),
        );
    }

    /// Register an internally generated texture (e.g. Render Target)
    ///
    /// These textures do not need CPU upload, have no version control,
    /// and their lifetime is managed by the caller.
    /// Typically called before a `RenderPass` executes.
    ///
    /// Suitable for: Pass-private resources where the Pass itself holds and maintains ID stability.
    /// Highest performance, no hash lookup.
    pub fn register_internal_texture_direct(&mut self, id: u64, view: wgpu::TextureView) {
        self.internal_resources.insert(id, view);
    }

    /// Suitable for: Cross-pass shared resources (e.g. "`SceneColor`").
    /// Internally maintains a Name -> ID mapping.
    pub fn register_internal_texture_by_name(
        &mut self,
        name: &str,
        view: wgpu::TextureView,
    ) -> u64 {
        // 1. Look up or create ID (String allocation only on first encounter of the name)
        let id = *self
            .internal_name_lookup
            .entry(name.to_string())
            .or_insert_with(generate_gpu_resource_id);

        // 2. Register
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

    /// Unified helper method for retrieving `TextureView`
    ///
    /// Prioritizes Asset-converted textures, then registered internal textures, finally returns Dummy
    pub fn get_texture_view<'a>(&'a self, source: &TextureSource) -> &'a wgpu::TextureView {
        match source {
            TextureSource::Asset(handle) => {
                // Special handling for Dummy Env Map
                if *handle == TextureHandle::dummy_env_map() {
                    return &self.system_textures.black_cube;
                }

                // Look up GPU resource corresponding to the Asset
                if let Some(binding) = self.texture_bindings.get(*handle)
                    && let Some(img) = self.gpu_images.get(binding.image_handle)
                {
                    return &img.default_view;
                }

                // Fallback
                &self.system_textures.black_2d
            }
            TextureSource::Attachment(id, _) => {
                // Directly look up the internal resource table
                self.internal_resources
                    .get(id)
                    .unwrap_or(&self.system_textures.black_2d)
            }
        }
    }

    // ========================================================================
    // Unified prepare_mesh entry point
    // ========================================================================

    /// Prepare basic resources for a Mesh
    ///
    /// Uses an "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" pattern
    pub fn prepare_mesh(
        &mut self,
        assets: &AssetServer,
        mesh: &mut Mesh,
        skeleton: Option<&Skeleton>,
    ) -> Option<BindGroupContext> {
        // === Ensure phase: ensure all resources are uploaded ===
        // If the Allocator expanded this frame, IDs will change and must be registered here
        mesh.update_morph_uniforms();
        let (_, morph_result) = self.ensure_buffer(&mesh.morph_uniforms);
        self.prepare_geometry(assets, mesh.geometry);
        self.prepare_material(assets, mesh.material);

        let geometry = assets.geometries.get(mesh.geometry)?;

        // === Collect phase: gather all resource IDs ===
        let mut current_ids = super::ResourceIdSet::with_capacity(6);
        current_ids.push(self.model_allocator.buffer_handle().id());
        current_ids.push(morph_result.resource_id);
        current_ids.push_optional(skeleton.map(|s| s.joint_matrices.handle().id));
        current_ids.push_optional(skeleton.map(|s| s.prev_joint_matrices.handle().id));

        let cache_key = current_ids.hash_value();

        // Check global cache
        if let Some(binding_data) = self.object_bind_group_cache.get(&cache_key) {
            return Some(binding_data.clone());
        }

        // Create new GpuObject
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

        let model_buffer_ref = self.model_allocator.buffer_handle();

        let mut builder = ResourceBuilder::new();
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model",
            model_buffer_ref,
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
            builder.add_storage_buffer(
                "prev_skins",
                &skeleton.prev_joint_matrices.handle(),
                None,
                true,
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into())),
            );
        }

        let binding_wgsl = builder.generate_wgsl(2);
        let layout_entries = builder.generate_layout_entries();

        let (layout, layout_id) = self.get_or_create_layout(&layout_entries);
        self.prepare_binding_resources(assets, &builder.bindings);
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
    // BindGroup common operations
    // ========================================================================

    pub(crate) fn prepare_binding_resources(
        &mut self,
        assets: &AssetServer,
        bindings: &[Binding<'_>],
    ) {
        for b in bindings {
            match &b.resource {
                BindingResource::Buffer {
                    buffer: buffer_ref,
                    offset: _,
                    size: _,
                    data,
                } => {
                    let id = buffer_ref.id();
                    if let Some(bytes) = data {
                        let handle = if let Some(&h) = self.buffer_index.get(&id) {
                            h
                        } else {
                            let mut buf = GpuBuffer::new(
                                &self.device,
                                bytes,
                                buffer_ref.usage,
                                buffer_ref.label(),
                            );
                            buf.last_uploaded_version = buffer_ref.version;
                            buf.last_used_frame = self.frame_index;
                            let h = self.gpu_buffers.insert(buf);
                            self.buffer_index.insert(id, h);
                            h
                        };

                        if let Some(gpu_buf) = self.gpu_buffers.get_mut(handle) {
                            if buffer_ref.version > gpu_buf.last_uploaded_version {
                                gpu_buf.write_to_gpu(&self.device, &self.queue, bytes);
                                gpu_buf.last_uploaded_version = buffer_ref.version;
                            }
                            gpu_buf.last_used_frame = self.frame_index;
                        }
                    } else if let Some(&h) = self.buffer_index.get(&id) {
                        if let Some(gpu_buf) = self.gpu_buffers.get_mut(h) {
                            gpu_buf.last_used_frame = self.frame_index;
                        }
                    } else {
                        panic!(
                            "ResourceManager: Trying to bind buffer {:?} (ID: {}) but it is not initialized!",
                            buffer_ref.label(),
                            id
                        );
                    }
                }
                BindingResource::Texture(Some(source)) => match source {
                    TextureSource::Asset(handle) => {
                        self.prepare_texture(assets, *handle);
                    }
                    TextureSource::Attachment(_, _) => {}
                },
                BindingResource::Texture(None) | BindingResource::_Phantom(_) => {}
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
        let mut binding_index = 0u32;

        for b in &builder.bindings {
            match &b.resource {
                BindingResource::Buffer {
                    buffer,
                    data: _,
                    offset,
                    size,
                } => {
                    let cpu_id = buffer.id();
                    let gpu_buf = self
                        .get_gpu_buffer_by_cpu_id(cpu_id)
                        .expect("Buffer should be prepared");
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &gpu_buf.buffer,
                            offset: *offset,
                            size: size.and_then(wgpu::BufferSize::new),
                        }),
                    });
                    binding_index += 1;
                }
                BindingResource::Texture(source_opt) => {
                    // 1. Texture view entry
                    let view = if let Some(source) = source_opt {
                        self.get_texture_view(source)
                    } else if let BindingDesc::Texture { view_dimension, .. } = &b.desc {
                        match view_dimension {
                            wgpu::TextureViewDimension::D2Array => {
                                &self.system_textures.depth_d2array
                            }
                            wgpu::TextureViewDimension::Cube => &self.system_textures.black_cube,
                            _ => &self.system_textures.black_2d,
                        }
                    } else {
                        &self.system_textures.black_2d
                    };
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                    binding_index += 1;

                    // 2. Auto-paired sampler entry
                    let sampler = self.resolve_texture_sampler(source_opt.as_ref(), &b.desc);
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding_index,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                    binding_index += 1;
                }
                BindingResource::_Phantom(_) => unreachable!("_Phantom should never be used"),
            }
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auto BindGroup"),
            layout,
            entries: &entries,
        });

        (bind_group, generate_gpu_resource_id())
    }

    /// Resolves the `wgpu::Sampler` for a texture binding.
    ///
    /// For asset textures, returns the sampler cached during `prepare_texture`.
    /// For attachments or missing textures, falls back to the dummy sampler
    /// (or the shadow comparison sampler for depth/comparison bindings).
    fn resolve_texture_sampler(
        &self,
        source: Option<&TextureSource>,
        desc: &BindingDesc,
    ) -> &wgpu::Sampler {
        if let Some(TextureSource::Asset(handle)) = source
            && let Some(binding) = self.texture_bindings.get(*handle)
            && let Some(sampler) = self
                .sampler_registry
                .get_sampler_by_index(binding.sampler_id)
        {
            return sampler;
        }

        // Fallback: comparison sampler for depth textures, default otherwise
        if let BindingDesc::Texture {
            sampler_binding_type: wgpu::SamplerBindingType::Comparison,
            ..
        } = desc
        {
            &self.system_textures.shadow_compare_sampler
        } else {
            self.sampler_registry.default_sampler().1
        }
    }

    // ========================================================================
    // Global bindings (Group 0)
    // ========================================================================

    /// Prepare global binding resources
    ///
    /// Uses an "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" pattern
    pub fn prepare_global(
        &mut self,
        assets: &AssetServer,
        scene: &Scene,
        render_state: &RenderState,
    ) -> u32 {
        let has_active_environment = matches!(scene.background.mode, myth_scene::background::BackgroundMode::Procedural(_))
            || scene.environment.has_env_map();

        // === Ensure: upload all buffers, obtain physical resource IDs ===
        let (_, camera_result) = self.ensure_buffer(render_state.uniforms());
        let (_, env_result) = self.ensure_buffer(&scene.uniforms_buffer);
        let (_, light_result) = self.ensure_buffer(&scene.light_storage_buffer);
        let (_, scene_uniform_result) = self.ensure_buffer(&scene.uniforms_buffer);

        // Resolve environment texture IDs from GpuEnvironment cache.
        // resolve_gpu_environment runs before prepare_global and always creates
        // cache entries, so a miss here should not happen in normal operation.
        let (processed_env_map_id, pmrem_map_id) = if has_active_environment {
            self.gpu_environment(scene.id())
                .map(|gpu_env| (gpu_env.base_cube_view.id(), gpu_env.pmrem_view.id()))
                .unwrap_or((
                    self.system_textures.black_cube.id(),
                    self.system_textures.black_cube.id(),
                ))
        } else {
            (
                self.system_textures.black_cube.id(),
                self.system_textures.black_cube.id(),
            )
        };

        let brdf_lut_id = self
            .brdf_lut_view_id
            .unwrap_or(self.system_textures.black_2d.id());

        // === Collect: gather all resource IDs ===
        let mut current_ids = super::ResourceIdSet::with_capacity(8);
        current_ids.push(camera_result.resource_id);
        current_ids.push(env_result.resource_id);
        current_ids.push(light_result.resource_id);
        current_ids.push(scene_uniform_result.resource_id);
        current_ids.push(processed_env_map_id);
        current_ids.push(pmrem_map_id);
        current_ids.push(brdf_lut_id);

        let state_id = Self::compute_global_state_key(render_state.id, scene.id());

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

        self.prepare_binding_resources(assets, &builder.bindings);
        let layout_entries = builder.generate_layout_entries();
        let (layout, layout_id) = self.get_or_create_layout(&layout_entries);
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
        use myth_resources::WgslStructName;
        use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage};

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
        let env_map_source = if matches!(scene.background.mode, myth_scene::background::BackgroundMode::Procedural(_))
            || scene.environment.has_env_map()
        {
            self.gpu_environment(scene.id()).map_or_else(
                || TextureHandle::dummy_env_map().into(),
                |gpu_env| {
                    TextureSource::Attachment(
                        gpu_env.base_cube_view.id(),
                        wgpu::TextureViewDimension::Cube,
                    )
                },
            )
        } else {
            TextureHandle::dummy_env_map().into()
        };

        builder.add_texture(
            "env_map",
            Some(env_map_source),
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::ShaderStages::FRAGMENT,
        );

        // Resolve pmrem_map from GpuEnvironment cache
        let pmrem_source = if matches!(scene.background.mode, myth_scene::background::BackgroundMode::Procedural(_))
            || scene.environment.has_env_map()
        {
            self.gpu_environment(scene.id()).map(|gpu_env| {
                TextureSource::Attachment(gpu_env.pmrem_view.id(), wgpu::TextureViewDimension::Cube)
            })
        } else {
            None
        };

        builder.add_texture(
            "pmrem_map",
            pmrem_source,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
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
    }

    pub fn get_global_state(&self, render_state_id: u32, scene_id: u32) -> Option<&GpuGlobalState> {
        let state_id = Self::compute_global_state_key(render_state_id, scene_id);
        self.global_states.get(&state_id)
    }
}
