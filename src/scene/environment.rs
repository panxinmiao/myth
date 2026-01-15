use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{GpuLightStorage, EnvironmentUniforms};
use crate::assets::TextureHandle;
use crate::scene::light::Light;

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Default, Clone, Debug)]
pub struct EnvironmentBindings {
    pub env_map: Option<TextureHandle>,
    pub brdf_lut: Option<TextureHandle>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
#[derive(Default)]
pub struct EnvironmentSettings {
    pub shadows_enabled: bool,
    pub fog_enabled: bool,
    pub tone_mapping: u32,
    pub ibl_enabled: bool,
}


pub struct Environment {
    pub id: u32,
    
    pub uniforms: CpuBuffer<EnvironmentUniforms>,
    pub bindings: EnvironmentBindings,
    pub settings: EnvironmentSettings,
    
    pub lights: Vec<Light>,
    pub light_storage: CpuBuffer<Vec<GpuLightStorage>>,
    
    pub binding_version: u64,
    pub layout_version: u64,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            uniforms: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("Environment Uniforms")
            ),
            bindings: EnvironmentBindings::default(),
            settings: EnvironmentSettings::default(),
            lights: Vec::new(),
            light_storage: CpuBuffer::new(
                Vec::new(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Light Storage Buffer")
            ),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, EnvironmentUniforms> {
        self.uniforms.write()
    }
    
    pub fn set_env_map(&mut self, texture: Option<TextureHandle>) {
        let layout_changed = self.bindings.env_map.is_some() != texture.is_some();
        self.bindings.env_map = texture;
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.binding_version = self.binding_version.wrapping_add(1);
        } else {
            self.binding_version = self.binding_version.wrapping_add(1);
        }
    }
    
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        if self.settings.shadows_enabled != enabled {
            self.settings.shadows_enabled = enabled;
            self.layout_version = self.layout_version.wrapping_add(1);
        }
    }
    
    pub fn set_fog_enabled(&mut self, enabled: bool) {
        if self.settings.fog_enabled != enabled {
            self.settings.fog_enabled = enabled;
            self.layout_version = self.layout_version.wrapping_add(1);
        }
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
        if gpu_lights.is_empty() {
            if !self.light_storage.read().is_empty() {
                *self.light_storage.write() = Vec::new();
                self.uniforms.write().num_lights = 0;
            }
            return;
        }

        let old_count = self.light_storage.read().len();
        *self.light_storage.write() = gpu_lights;
        
        if old_count != self.light_storage.read().len() {
            self.uniforms.write().num_lights = self.light_storage.read().len() as u32;
        }
    }
}