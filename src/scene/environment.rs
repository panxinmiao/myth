use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::BufferRef;
use crate::resources::version_tracker::MutGuard;
use crate::resources::uniforms::{GpuLightStorage, EnvironmentUniforms};
use crate::assets::TextureHandle;
use crate::scene::light::Light;

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Default, Clone, Debug)]
pub struct EnvironmentBindings {
    pub env_map: Option<TextureHandle>,
    pub diffuse_map: Option<TextureHandle>,
    pub specular_map: Option<TextureHandle>,
    pub brdf_lut: Option<TextureHandle>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct EnvironmentSettings {
    pub shadows_enabled: bool,
    pub fog_enabled: bool,
    pub tone_mapping: u32,
    pub ibl_enabled: bool,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            shadows_enabled: false,
            fog_enabled: false,
            tone_mapping: 0,
            ibl_enabled: false,
        }
    }
}

pub struct Environment {
    pub id: u32,
    
    uniforms: EnvironmentUniforms,
    bindings: EnvironmentBindings,
    settings: EnvironmentSettings,
    
    pub lights: Vec<Light>,
    pub(crate) light_storage_buffer: BufferRef,
    
    pub uniform_version: u64,
    pub binding_version: u64,
    pub layout_version: u64,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            uniforms: EnvironmentUniforms::default(),
            bindings: EnvironmentBindings::default(),
            settings: EnvironmentSettings::default(),
            lights: Vec::new(),
            light_storage_buffer: BufferRef::new_with_capacity(
                std::mem::size_of::<GpuLightStorage>() * 32, 
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, 
                Some("Light Storage Buffer")
            ),
            uniform_version: 0,
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms(&self) -> &EnvironmentUniforms { &self.uniforms }
    pub fn bindings(&self) -> &EnvironmentBindings { &self.bindings }
    pub fn settings(&self) -> &EnvironmentSettings { &self.settings }
    
    pub fn uniforms_mut(&mut self) -> MutGuard<'_, EnvironmentUniforms> {
        MutGuard::new(&mut self.uniforms, &mut self.uniform_version)
    }
    
    pub fn bindings_mut(&mut self) -> MutGuard<'_, EnvironmentBindings> {
        MutGuard::new(&mut self.bindings, &mut self.binding_version)
    }
    
    pub fn settings_mut(&mut self) -> MutGuard<'_, EnvironmentSettings> {
        MutGuard::new(&mut self.settings, &mut self.layout_version)
    }
    
    pub fn set_env_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.env_map.is_some() != texture.is_some();
        self.bindings_mut().env_map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
        }
    }
    
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.settings_mut().shadows_enabled = enabled;
    }
    
    pub fn set_fog_enabled(&mut self, enabled: bool) {
        self.settings_mut().fog_enabled = enabled;
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
        if gpu_lights.is_empty() {
            return;
        }

        self.light_storage_buffer.update(&gpu_lights);
        self.uniforms_mut().num_lights = gpu_lights.len() as u32;
    }
}