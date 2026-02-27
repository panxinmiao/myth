//! Material operations
//!
//! Uses an "Ensure -> Check -> Rebuild" pattern:
//! 1. Ensure all GPU resources exist with up-to-date data
//! 2. Compare physical resource IDs for changes (decide `BindGroup` rebuild)
//! 3. Update material version number (for Pipeline cache)
//!
//! # Three-dimensional version tracking separation
//!
//! 1. **Resource topology (`BindGroup`)**: Tracked by `ResourceIdSet`
//!    - Texture/sampler/Buffer ID changes -> rebuild `BindGroup`
//!
//! 2. **Resource content (Buffer Data)**: Tracked by `BufferRef`
//!    - Atomic version number changes -> upload Buffer
//!
//! 3. **Pipeline state (`RenderPipeline`)**: Tracked by `Material.version()`
//!    - Depth write/transparency/double-sided rendering changes -> switch Pipeline

use crate::assets::{AssetServer, MaterialHandle};
use crate::renderer::core::resources::EnsureResult;
use crate::resources::material::{Material, RenderableMaterialTrait};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::texture::TextureSource;

use super::{ResourceIdSet, ResourceManager, hash_layout_entries};

/// GPU-side material resource
///
/// Uses resource ID tracking for automatic change detection
///
/// # Three-dimensional version tracking separation
///
/// 1. **Resource topology (`BindGroup`)**: Tracked by `resource_ids`
///    - Texture/sampler/Buffer ID changes -> rebuild `BindGroup`
///
/// 2. **Resource content (Buffer Data)**: Tracked by `BufferRef` (external)
///    - Atomic version number changes -> upload Buffer
///
/// 3. **Pipeline state (`RenderPipeline`)**: Tracked by `version`
///    - Depth write/transparency/double-sided rendering changes -> switch Pipeline
pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    /// Hash of layout entries (for fast comparison to determine if Layout rebuild is needed)
    pub layout_hash: u64,
    pub binding_wgsl: String,
    /// Set of physical IDs of all dependent resources (guards `BindGroup` validity)
    pub resource_ids: ResourceIdSet,
    /// Records the Material version when this `GpuMaterial` was generated (for Pipeline cache)
    pub version: u64,
    pub last_used_frame: u64,
    pub last_verified_frame: u64,
}

impl ResourceManager {
    /// Prepare Material GPU resources
    ///
    /// Three-dimensional change detection:
    /// - Resource topology change -> rebuild BindGroup (detected by `ResourceIdSet`)
    /// - Resource content change -> upload Buffer (handled automatically by `BufferRef`)
    /// - Pipeline state change -> switch Pipeline (recorded by version, used externally)
    pub(crate) fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let Some(material) = assets.materials.get(handle) else {
            return;
        };

        // [Fast Path] Per-frame cache check
        if let Some(gpu_mat) = self.gpu_materials.get(handle)
            && gpu_mat.last_verified_frame == self.frame_index
        {
            return;
        }

        // 1. Ensure phase: ensure all resources exist with up-to-date data, collect physical resource IDs
        let mut current_resource_ids = self.ensure_material_resources(assets, &material);

        // 2. Check phase: determine if BindGroup needs to be rebuilt
        // Note: only IDs are checked here, not version!
        // Even if material.version() changed (e.g. blend mode), BindGroup is not rebuilt as long as IDs are unchanged
        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            let mut cached_ids = gpu_mat.resource_ids.clone();
            !current_resource_ids.matches(&mut cached_ids)
        } else {
            true
        };

        if needs_rebuild_bindgroup {
            // 3. Rebuild phase: rebuild BindGroup (expensive operation)
            self.rebuild_material_bindgroup(assets, handle, &material, current_resource_ids);
        }

        // 4. Update version number and frame counter (very fast operation)
        // Regardless of whether BindGroup was rebuilt, gpu_mat's version must be up-to-date
        // so PipelineCache can use this version for fast checks during rendering
        if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
            gpu_mat.version = material.data.version();
            gpu_mat.last_used_frame = self.frame_index;
            gpu_mat.last_verified_frame = self.frame_index;
        }
    }

    /// Ensure Material resources and return the resource ID set
    ///
    /// Uses `visit_textures` to iterate over all texture resources
    fn ensure_material_resources(
        &mut self,
        assets: &AssetServer,
        material: &Material,
    ) -> ResourceIdSet {
        let mut uniform_result = EnsureResult::existing(0);

        // 2. Call with_uniform_bytes, passing a closure
        material.data.with_uniform_bytes(&mut |bytes| {
            uniform_result = self.ensure_buffer_ref(&material.data.uniform_buffer(), bytes);
        });

        // Collect resource IDs
        let mut resource_ids = ResourceIdSet::with_capacity(16);
        resource_ids.push(uniform_result.resource_id);

        // Use visit_textures to iterate over all texture resources
        material
            .data
            .visit_textures(&mut |tex_source| match tex_source {
                TextureSource::Asset(tex_handle) => {
                    self.prepare_texture(assets, *tex_handle);
                    if let Some(binding) = self.texture_bindings.get(*tex_handle) {
                        resource_ids.push(binding.view_id);
                        resource_ids.push(binding.sampler_id);
                    } else {
                        resource_ids.push(self.dummy_image.id);
                        resource_ids.push(self.dummy_sampler.id);
                    }
                }
                TextureSource::Attachment(id, _) => {
                    resource_ids.push(*id);
                    resource_ids.push(self.dummy_sampler.id);
                }
            });

        resource_ids
    }

    /// Rebuild Material's BindGroup (may include Layout)
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

        // Compute hash of layout entries
        let layout_hash = hash_layout_entries(&builder.layout_entries);

        // Check if a new Layout is needed
        let (layout, layout_id) = if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            if gpu_mat.layout_hash == layout_hash {
                // Layout unchanged, reuse
                (gpu_mat.layout.clone(), gpu_mat.layout_id)
            } else {
                // Layout changed, rebuild
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
