//! BindGroup 相关操作
//!
//! 包括 Object BindGroup (Group 2)、骨骼管理和环境绑定

use glam::Mat4;
use wgpu::ShaderStages;

use crate::Mesh;
use crate::assets::{AssetServer, GeometryHandle};
use crate::resources::buffer::CpuBuffer;
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::resources::uniforms::{DynamicModelUniforms, MorphUniforms};
use crate::scene::SkeletonKey;
use crate::scene::skeleton::SkinBinding;
use crate::scene::environment::Environment;

use crate::renderer::core::binding::{BindingResource, Bindings};
use crate::renderer::core::builder::{ResourceBuilder, WgslStructName};
use crate::renderer::graph::RenderState;

use super::{
    ResourceManager, GpuBuffer, GpuEnvironment, 
    ObjectBindGroupKey, ObjectBindingData, CachedBindGroupId,
    generate_resource_id, ModelBufferAllocator,
};

impl ResourceManager {
    // ========================================================================
    // 骨骼管理
    // ========================================================================

    /// 更新骨骼数据到 GPU
    pub fn prepare_skeleton(&mut self, skeleton_id: SkeletonKey, matrices: &[Mat4]) {
        // 先检查是否存在，如果不存在则插入
        if !self.skeleton_buffers.contains_key(&skeleton_id) {
            self.skeleton_buffers.insert(skeleton_id, CpuBuffer::new(
                matrices.to_vec(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Skeleton")
            ));
        }

        // 现在安全地更新数据
        let buffer = self.skeleton_buffers.get_mut(&skeleton_id).unwrap();
        buffer.write()[..matrices.len()].copy_from_slice(matrices);
        
        // 获取需要的信息后再调用 write_buffer
        let handle = buffer.handle().clone();
        let bytes_vec: Vec<u8> = buffer.as_bytes().to_vec();
        self.write_buffer(&handle, &bytes_vec);
    }

    /// 获取骨骼 Buffer
    pub fn get_skeleton_buffer(&self, skeleton_id: SkeletonKey) -> Option<&CpuBuffer<Vec<Mat4>>> {
        self.skeleton_buffers.get(&skeleton_id)
    }

    // ========================================================================
    // 统一的 prepare_mesh 入口
    // ========================================================================

    /// 准备 Mesh 的基础资源（几何体、材质、Morph Uniform Buffer）
    pub fn prepare_mesh(&mut self, assets: &AssetServer, mesh: &Mesh) {
        // 处理 morph uniform buffer
        let buf_id = mesh.morph_uniforms.handle().id;
        if let Some(gpu_buffer) = self.gpu_buffers.get_mut(&buf_id) {
            self.queue.write_buffer(&gpu_buffer.buffer, 0, mesh.morph_uniforms.as_bytes());
            gpu_buffer.last_used_frame = self.frame_index;
        } else {
            let mut gpu_buf = GpuBuffer::new(&self.device, mesh.morph_uniforms.as_bytes(), mesh.morph_uniforms.handle().usage(), mesh.morph_uniforms.handle().label());
            gpu_buf.last_uploaded_version = mesh.morph_uniforms.handle().version;
            gpu_buf.last_used_frame = self.frame_index;
            self.gpu_buffers.insert(buf_id, gpu_buf);
        }

        self.prepare_geometry(assets, mesh.geometry);
        self.prepare_material(assets, mesh.material);
    }

    /// 准备 Object BindGroup (Group 2)
    pub fn prepare_object_bind_group(
        &mut self,
        assets: &AssetServer,
        geometry_handle: GeometryHandle,
        geometry: &Geometry,
        mesh: &Mesh,
        skin_binding: Option<&SkinBinding>,
    ) -> ObjectBindingData {
        let features = geometry.get_features();
        let is_static = !features.intersects(GeometryFeatures::USE_MORPHING | GeometryFeatures::USE_SKINNING);

        let has_skin_binding = skin_binding.is_some();
        let geo_supports_skinning = features.contains(GeometryFeatures::USE_SKINNING);
        let use_skinning = geo_supports_skinning && has_skin_binding;

        let skeleton_id = if use_skinning {
            skin_binding.map(|s| s.skeleton)
        } else {
            None
        };

        let model_buffer_id = self.model_allocator.buffer_id();
        let morph_buffer_id = Some(mesh.morph_uniforms.handle().id);

        let key = ObjectBindGroupKey {
            geo_id: if is_static { None } else { Some(geometry_handle) },
            model_buffer_id,
            skeleton_id,
            morph_buffer_id,
        };

        if let Some(binding_data) = self.object_bind_group_cache.get(&key) {
            return binding_data.clone();
        }

        // 提取 skeleton buffer 信息（如果需要）
        let skeleton_buffer_info = if use_skinning {
            if let Some(skel_id) = skeleton_id {
                self.skeleton_buffers.get(&skel_id)
            } else {
                None
            }
        } else {
            None
        };

        // 提前提取需要的数据，避免借用冲突
        let min_binding_size = ModelBufferAllocator::uniform_stride();
        let model_buffer_ref = self.model_allocator.cpu_buffer().handle().clone();
        let model_buffer_bytes = self.model_allocator.cpu_buffer().as_bytes().to_vec();

        // 提取 skeleton buffer 的数据（如果需要）
        let skeleton_data = skeleton_buffer_info.map(|cpu_buffer| {
            (cpu_buffer.handle().clone(), cpu_buffer.as_bytes().to_vec())
        });

        // 现在构建 ResourceBuilder - 不再借用 self
        let mut builder = ResourceBuilder::new();

        builder.add_dynamic_uniform_raw::<DynamicModelUniforms>("model", &model_buffer_ref, &model_buffer_bytes, min_binding_size, ShaderStages::VERTEX);

        mesh.define_bindings(&mut builder);

        let use_morphing = features.contains(GeometryFeatures::USE_MORPHING);
        if use_morphing {
            builder.add_uniform::<MorphUniforms>(
                "morph_targets",
                &mesh.morph_uniforms,
                ShaderStages::VERTEX
            );
        }
        geometry.define_bindings(&mut builder);

        if let Some((skel_handle, skel_bytes)) = &skeleton_data {
            builder.add_storage_buffer_raw(
                "skins", 
                skel_handle, 
                skel_bytes, 
                true, 
                ShaderStages::VERTEX,
                Some(WgslStructName::Name("mat4x4<f32>".into()))
            );
        }

        let binding_wgsl = builder.generate_wgsl(2);
        let layout_entries = builder.layout_entries.clone();
        let resources = std::mem::take(&mut builder.resources);
        
        let (layout, _layout_id) = self.get_or_create_layout(&layout_entries);
        
        // 先上传 buffer 数据到 GPU，再创建 bind group
        self.prepare_binding_resources(assets, &resources);
        let (bind_group, bind_group_id) = self.create_bind_group(&layout, &resources);

        let cached_id = CachedBindGroupId {
            bind_group_id,
            model_buffer_id,
        };

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl,
            cached_id,
        };

        // 同时更新两个缓存
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
                        self.gpu_textures.get(*handle).unwrap_or(&self.dummy_texture)
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
    // 环境绑定
    // ========================================================================

    pub fn prepare_global(&mut self, assets: &AssetServer, env: &Environment, render_state: &RenderState) {
        let world_id = Self::compose_env_render_state_id(render_state.id, env.id);

        if let Some(gpu_env) = self.worlds.get_mut(&world_id) {
            let uniform_match = gpu_env.last_uniform_version == env.uniforms().buffer.version;
            let binding_match = gpu_env.last_binding_version == env.binding_version();
            let layout_match = gpu_env.last_layout_version == env.layout_version();
            let render_state_match = render_state.uniforms().buffer.version == gpu_env.last_render_state_version;

            if uniform_match && binding_match && layout_match && render_state_match {
                gpu_env.last_used_frame = self.frame_index;
                return;
            }
        }

        let mut builder = ResourceBuilder::new();
        render_state.define_bindings(&mut builder);
        env.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);

        let needs_new_bind_group = if let Some(gpu_env) = self.worlds.get(&world_id) {
            gpu_env.layout_id != layout_id || gpu_env.last_binding_version != env.binding_version()
        } else { true };

        if !needs_new_bind_group {
            if let Some(gpu_env) = self.worlds.get_mut(&world_id) {
                gpu_env.last_uniform_version = env.uniforms().buffer.version;
                gpu_env.last_render_state_version = render_state.uniforms().buffer.version;
                gpu_env.last_used_frame = self.frame_index;
            }
            return;
        }

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(0);

        let gpu_world = GpuEnvironment {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            last_uniform_version: env.uniforms().buffer.version,
            last_binding_version: env.binding_version(),
            last_layout_version: env.layout_version(),
            last_render_state_version: render_state.uniforms().buffer.version,
            last_used_frame: self.frame_index,
        };
        self.worlds.insert(world_id, gpu_world);
    }

    fn compose_env_render_state_id(render_state_id: u32, env_id: u32) -> u64 {
        ((render_state_id as u64) << 32) | (env_id as u64)
    }

    pub fn get_world(&self, render_state_id: u32, env_id: u32) -> Option<&GpuEnvironment> {
        let world_id = Self::compose_env_render_state_id(render_state_id, env_id);
        self.worlds.get(&world_id)
    }
}
