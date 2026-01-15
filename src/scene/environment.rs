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
    
    uniforms: EnvironmentUniforms,
    bindings: EnvironmentBindings,
    settings: EnvironmentSettings,
    
    pub lights: Vec<Light>,
    pub(crate) gpu_light_data: Vec<GpuLightStorage>, 
    pub(crate) light_storage_buffer: BufferRef,
    
    pub uniform_version: u64,
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
            uniforms: EnvironmentUniforms::default(),
            bindings: EnvironmentBindings::default(),
            settings: EnvironmentSettings::default(),
            lights: Vec::new(),
            gpu_light_data: Vec::new(), // 初始化为空
            light_storage_buffer: BufferRef::with_capacity(
                std::mem::size_of::<GpuLightStorage>() * 32, // 预估大小
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
            if !self.gpu_light_data.is_empty() {
                self.gpu_light_data.clear();
                self.uniforms_mut().num_lights = 0;
            }
            return;
        }

        self.gpu_light_data = gpu_lights;
        if (self.uniforms.num_lights as usize) != self.gpu_light_data.len() {
            self.uniforms_mut().num_lights = self.gpu_light_data.len() as u32;
        }

        // Check capacity and resize if needed (2x growth)
        let current_bytes = std::mem::size_of_val(self.gpu_light_data.as_slice());
        if current_bytes > self.light_storage_buffer.size() {
            let new_capacity = current_bytes * 2;
            self.light_storage_buffer = BufferRef::with_capacity(
                new_capacity,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Light Storage Buffer (Resized)")
            );
            self.binding_version = self.binding_version.wrapping_add(1);
        }else{
            self.light_storage_buffer.version = self.light_storage_buffer.version.wrapping_add(1);
        }
    }
}