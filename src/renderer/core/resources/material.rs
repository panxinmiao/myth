//! Material 相关操作
//!
//! 采用 "Ensure -> Check -> Rebuild" 模式：
//! 1. 确保所有 GPU 资源存在且数据最新
//! 2. 比较物理资源 ID 是否变化
//! 3. 如需重建，比较 LayoutEntries 决定是否需要新 Layout

use crate::assets::{AssetServer, MaterialHandle};
use crate::resources::material::Material;
use crate::resources::material::MaterialData;

use crate::renderer::core::binding::Bindings;
use crate::renderer::core::builder::ResourceBuilder;

use super::{ResourceManager, GpuMaterial, ResourceIdSet, hash_layout_entries};

/// Material 准备结果
/// 
/// 预留的 API，用于未来更精细的 Pipeline 缓存控制
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MaterialPrepareResult {
    /// Uniform Buffer 的物理 ID
    pub uniform_buffer_id: u64,
    /// 所有 GpuImage 的物理 ID（对应纹理像素数据）
    pub image_ids: Vec<u64>,
    /// 所有 GpuSampler 的物理 ID（对应采样参数）
    pub sampler_ids: Vec<u64>,
    /// Layout ID
    pub layout_id: u64,
    /// BindGroup ID
    pub bind_group_id: u64,
    /// 是否有资源被重建
    pub any_recreated: bool,
}

impl ResourceManager {
    /// 准备 Material 的 GPU 资源
    /// 
    /// 使用新的资源 ID 追踪机制，自动检测变化并按需重建
    pub(crate) fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) -> Option<MaterialPrepareResult> {
        let material = assets.get_material(handle)?;

        // 1. Ensure 阶段：确保所有资源存在且数据最新，收集物理资源 ID
        let (uniform_result, image_ids, sampler_ids) = self.ensure_material_resources_with_ids(assets, material);
        
        // 2. 构建当前资源 ID 集合（包含 image_ids 和 sampler_ids）
        let mut current_resource_ids = ResourceIdSet::with_capacity(1 + image_ids.len() + sampler_ids.len());
        current_resource_ids.push(uniform_result.resource_id);
        for id in &image_ids {
            current_resource_ids.push(*id);
        }
        for id in &sampler_ids {
            current_resource_ids.push(*id);
        }

        // 3. Check 阶段：检查是否需要重建 BindGroup
        let needs_rebuild = if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            // 比较资源 ID 是否变化
            let mut cached_ids = gpu_mat.resource_ids.clone();
            !current_resource_ids.matches(&mut cached_ids) || uniform_result.was_recreated
        } else {
            true
        };

        if needs_rebuild {
            // 4. Rebuild 阶段
            self.rebuild_material_with_ids(assets, handle, material, current_resource_ids);
        } else {
            // 仅更新帧计数
            if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
                gpu_mat.last_used_frame = self.frame_index;
            }
        }

        let gpu_mat = self.gpu_materials.get(handle)?;
        Some(MaterialPrepareResult {
            uniform_buffer_id: uniform_result.resource_id,
            image_ids,
            sampler_ids,
            layout_id: gpu_mat.layout_id,
            bind_group_id: gpu_mat.bind_group_id,
            any_recreated: needs_rebuild,
        })
    }

    /// 确保 Material 资源并返回物理 ID
    /// 
    /// 返回: (uniform_result, image_ids, sampler_ids)
    fn ensure_material_resources_with_ids(&mut self, assets: &AssetServer, material: &Material) -> (super::EnsureResult, Vec<u64>, Vec<u64>) {
        let uniform_result = match &material.data {
            MaterialData::Basic(m) => self.ensure_buffer(&m.uniforms),
            MaterialData::Phong(m) => self.ensure_buffer(&m.uniforms),
            MaterialData::Standard(m) => self.ensure_buffer(&m.uniforms),
        };

        // 分别收集 Image ID 和 Sampler ID
        let mut image_ids = Vec::new();
        let mut sampler_ids = Vec::new();
        let bindings = material.data.bindings();
        
        for tex_handle in [
            bindings.map,
            bindings.normal_map,
            bindings.roughness_map,
            bindings.metalness_map,
            bindings.emissive_map,
            bindings.ao_map,
            bindings.specular_map,
        ].into_iter().flatten() {
            self.prepare_texture(assets, tex_handle);
            if let Some(binding) = self.texture_bindings.get(tex_handle) {
                image_ids.push(binding.image_id);
                sampler_ids.push(binding.sampler_id);
            }
        }

        (uniform_result, image_ids, sampler_ids)
    }

    /// 重建 Material 的 BindGroup（可能包括 Layout）
    fn rebuild_material_with_ids(
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

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GpuMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            layout_hash,
            binding_wgsl,
            resource_ids,
            last_used_frame: self.frame_index,
        };

        self.gpu_materials.insert(handle, gpu_mat);
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }
}
