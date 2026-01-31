//! Material 相关操作
//!
//! 采用 "Ensure -> Check -> Rebuild" 模式：
//! 1. 确保所有 GPU 资源存在且数据最新
//! 2. 比较物理资源 ID 是否变化（决定 BindGroup 重建）
//! 3. 更新材质版本号（用于 Pipeline 缓存）
//! 
//! # 版本追踪三维分离
//! 
//! 1. **资源拓扑 (BindGroup)**: 由 `ResourceIdSet` 追踪
//!    - 纹理/采样器/Buffer ID 变化 -> 重建 BindGroup
//! 
//! 2. **资源内容 (Buffer Data)**: 由 `BufferRef` 追踪
//!    - Atomic 版本号变化 -> 上传 Buffer
//! 
//! 3. **管线状态 (RenderPipeline)**: 由 `Material.version()` 追踪
//!    - 深度写入/透明度/双面渲染等变化 -> 切换 Pipeline

use crate::assets::{AssetServer, MaterialHandle};
use crate::renderer::core::resources::EnsureResult;
use crate::resources::material::{Material, RenderableMaterialTrait};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::texture::TextureSource;

use super::{ResourceManager, GpuMaterial, ResourceIdSet, hash_layout_entries};


impl ResourceManager {
    /// 准备 Material 的 GPU 资源
    /// 
    /// 三维分离的变更检测：
    /// - 资源拓扑变化 -> 重建 BindGroup（由 ResourceIdSet 检测）
    /// - 资源内容变化 -> 上传 Buffer（由 BufferRef 自动处理）
    /// - 管线状态变化 -> 切换 Pipeline（由 version 记录，供外部使用）
    pub(crate) fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let Some(material) = assets.materials.get(handle) else {
            return;
        };
   
        // [Fast Path] 帧内缓存检查
        if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            if gpu_mat.last_verified_frame == self.frame_index {
                return;
            }
        }

        // 1. Ensure 阶段：确保所有资源存在且数据最新，收集物理资源 ID
        let mut current_resource_ids = self.ensure_material_resources(assets, &material);

        // 2. Check 阶段：检查是否需要重建 BindGroup
        // 注意：这里只看 ID，不看 version！
        // 即使 material.version() 变了（比如改了混合模式），只要 ID 没变，就不重建 BindGroup
        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            let mut cached_ids = gpu_mat.resource_ids.clone();
            !current_resource_ids.matches(&mut cached_ids)
        } else {
            true
        };

        if needs_rebuild_bindgroup {
            // 3. Rebuild 阶段：重建 BindGroup（耗时操作）
            self.rebuild_material_bindgroup(assets, handle, &material, current_resource_ids);
        }

        // 4. 更新版本号和帧计数（极速操作）
        // 无论是否重建了 BindGroup，都需要确保 gpu_mat 里的 version 是最新的
        // 这样 PipelineCache 在渲染时可以用这个 version 做快速 check
        if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
            gpu_mat.version = material.data.version();
            gpu_mat.last_used_frame = self.frame_index;
            gpu_mat.last_verified_frame = self.frame_index;
        }

    }

    /// 确保 Material 资源并返回资源 ID 集合
    /// 
    /// 使用 `visit_textures` 遍历所有纹理资源
    fn ensure_material_resources(&mut self, assets: &AssetServer, material: &Material) -> ResourceIdSet {
        // 确保 Uniform Buffer
        // let uniform_result = self.ensure_buffer_ref(
        //     material.data.uniform_buffer(), 
        //     material.data.uniform_bytes()
        // );

        let mut uniform_result = EnsureResult::existing(0); 

        // 2. 调用 with_uniform_bytes，传入闭包
        material.data.with_uniform_bytes(&mut |bytes| {
            uniform_result = self.ensure_buffer_ref(
                &material.data.uniform_buffer(), 
                bytes
            );
        });

        // 收集资源 ID
        let mut resource_ids = ResourceIdSet::with_capacity(16);
        resource_ids.push(uniform_result.resource_id);

        // 使用 visit_textures 遍历所有纹理资源
        material.data.visit_textures(&mut |tex_source| {
            match tex_source {
                TextureSource::Asset(tex_handle) => {
                    self.prepare_texture(assets, *tex_handle);
                    if let Some(binding) = self.texture_bindings.get(*tex_handle) {
                        resource_ids.push(binding.view_id);
                        resource_ids.push(binding.sampler_id);
                    } else {
                        resource_ids.push(self.dummy_image.id);
                        resource_ids.push(self.dummy_sampler.id);
                    }
                },
                TextureSource::Attachment(id,_) => {
                    resource_ids.push(*id);
                    resource_ids.push(self.dummy_sampler.id);
                },
            }
        });

        resource_ids
    }

    /// 重建 Material 的 BindGroup（可能包括 Layout）
    fn rebuild_material_bindgroup(
        &mut self, 
        assets: &AssetServer, 
        handle: MaterialHandle, 
        material: &Material,
        resource_ids: ResourceIdSet,
    ) {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        
        // 计算 layout entries 的哈希值
        let layout_hash = hash_layout_entries(&builder.layout_entries);
        
        // 检查是否需要新的 Layout
        let (layout, layout_id) = if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            if gpu_mat.layout_hash == layout_hash {
                // Layout 未变化，复用
                (gpu_mat.layout.clone(), gpu_mat.layout_id)
            } else {
                // Layout 变化，重建
                self.get_or_create_layout(&builder.layout_entries)
            }
        } else {
            self.get_or_create_layout(&builder.layout_entries)
        };

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GpuMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            layout_hash,
            binding_wgsl,
            resource_ids,
            version: material.data.version(),
            last_used_frame: self.frame_index,
            last_verified_frame: self.frame_index,
        };

        self.gpu_materials.insert(handle, gpu_mat);
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }
}
