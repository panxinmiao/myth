//! Scene environment and global IBL utility resource storage.

use rustc_hash::FxHashMap;

use super::{GpuEnvironment, InternalTextureRegistry};

/// Scene environment and global IBL utility resources.
#[derive(Default)]
pub(crate) struct EnvironmentStore {
    scene_gpu_environments: FxHashMap<u32, GpuEnvironment>,
    brdf_lut_texture: Option<wgpu::Texture>,
    brdf_lut_view_id: Option<u64>,
    needs_brdf_compute: bool,
}

impl EnvironmentStore {
    pub(super) fn prune(&mut self, cutoff: u64, internal_textures: &mut InternalTextureRegistry) {
        let stale_scene_envs: Vec<u32> = self
            .scene_gpu_environments
            .iter()
            .filter_map(|(scene_id, gpu_env)| {
                (gpu_env.last_used_frame < cutoff).then_some(*scene_id)
            })
            .collect();

        for scene_id in stale_scene_envs {
            if let Some(gpu_env) = self.scene_gpu_environments.remove(&scene_id) {
                internal_textures.remove(gpu_env.base_cube_view.id());
                internal_textures.remove(gpu_env.pmrem_view.id());
            }
        }
    }

    #[inline]
    #[must_use]
    pub(super) fn scene_environment(&self, scene_id: u32) -> Option<&GpuEnvironment> {
        self.scene_gpu_environments.get(&scene_id)
    }

    #[inline]
    pub(super) fn scene_environment_mut(&mut self, scene_id: u32) -> Option<&mut GpuEnvironment> {
        self.scene_gpu_environments.get_mut(&scene_id)
    }

    #[inline]
    pub(super) fn remove_scene_environment(&mut self, scene_id: u32) -> Option<GpuEnvironment> {
        self.scene_gpu_environments.remove(&scene_id)
    }

    #[inline]
    pub(super) fn insert_scene_environment(&mut self, scene_id: u32, gpu_env: GpuEnvironment) {
        self.scene_gpu_environments.insert(scene_id, gpu_env);
    }

    #[inline]
    #[must_use]
    pub(super) fn brdf_lut_view_id(&self) -> Option<u64> {
        self.brdf_lut_view_id
    }

    #[inline]
    pub(super) fn set_brdf_lut(&mut self, texture: wgpu::Texture, view_id: u64) {
        self.brdf_lut_texture = Some(texture);
        self.brdf_lut_view_id = Some(view_id);
        self.needs_brdf_compute = true;
    }

    #[inline]
    #[must_use]
    pub(super) fn brdf_lut_texture(&self) -> Option<&wgpu::Texture> {
        self.brdf_lut_texture.as_ref()
    }

    #[inline]
    #[must_use]
    pub(super) fn needs_brdf_compute(&self) -> bool {
        self.needs_brdf_compute
    }

    #[inline]
    pub(super) fn mark_brdf_lut_computed(&mut self) {
        self.needs_brdf_compute = false;
    }
}
