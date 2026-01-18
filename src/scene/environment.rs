use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{GpuLightStorage, EnvironmentUniforms};
use crate::assets::TextureHandle;
use crate::scene::light::Light;
use crate::resources::texture::Texture;

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Default, Clone, Debug, PartialEq)]
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

/// BindingsGuard for Environment
pub struct EnvBindingsGuard<'a> {
    bindings: &'a mut EnvironmentBindings,
    binding_version: &'a mut u64,
    _layout_version: &'a mut u64,
    old_bindings: EnvironmentBindings,
}

impl<'a> std::ops::Deref for EnvBindingsGuard<'a> {
    type Target = EnvironmentBindings;
    fn deref(&self) -> &Self::Target {
        self.bindings
    }
}

impl<'a> std::ops::DerefMut for EnvBindingsGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bindings
    }
}

impl<'a> Drop for EnvBindingsGuard<'a> {
    fn drop(&mut self) {
        let old = &self.old_bindings;
        let new = &*self.bindings;
        
        if old.env_map != new.env_map || old.brdf_lut != new.brdf_lut {
            *self.binding_version = self.binding_version.wrapping_add(1);
        }

        // *self.layout_version;
        
    }
}

/// SettingsGuard for Environment
pub struct EnvSettingsGuard<'a> {
    settings: &'a mut EnvironmentSettings,
    layout_version: &'a mut u64,
    old_settings: EnvironmentSettings,
}

impl<'a> std::ops::Deref for EnvSettingsGuard<'a> {
    type Target = EnvironmentSettings;
    fn deref(&self) -> &Self::Target {
        self.settings
    }
}

impl<'a> std::ops::DerefMut for EnvSettingsGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.settings
    }
}

impl<'a> Drop for EnvSettingsGuard<'a> {
    fn drop(&mut self) {
        if &self.old_settings != &*self.settings {
            *self.layout_version = self.layout_version.wrapping_add(1);
        }
    }
}

pub struct Environment {
    pub id: u32,
    
    uniforms: CpuBuffer<EnvironmentUniforms>,
    bindings: EnvironmentBindings,
    settings: EnvironmentSettings,
    
    pub lights: Vec<Light>,
    pub light_storage: CpuBuffer<Vec<GpuLightStorage>>,
    
    binding_version: u64,
    layout_version: u64,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        // 初始化时预分配一定数量的灯光存储空间(16 个)
        let initial_light_data = vec![GpuLightStorage::default(); 16];

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
                initial_light_data,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Light Storage Buffer")
            ),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    // Uniforms accessor
    pub fn uniforms(&self) -> &CpuBuffer<EnvironmentUniforms> {
        &self.uniforms
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, EnvironmentUniforms> {
        self.uniforms.write()
    }

    // Bindings accessors
    pub fn bindings(&self) -> &EnvironmentBindings {
        &self.bindings
    }
    
    pub fn bindings_mut(&mut self) -> EnvBindingsGuard<'_> {
        EnvBindingsGuard {
            old_bindings: self.bindings.clone(),
            bindings: &mut self.bindings,
            binding_version: &mut self.binding_version,
            _layout_version: &mut self.layout_version,
        }
    }

    // Settings accessors
    pub fn settings(&self) -> &EnvironmentSettings {
        &self.settings
    }
    
    pub fn settings_mut(&mut self) -> EnvSettingsGuard<'_> {
        EnvSettingsGuard {
            old_settings: self.settings.clone(),
            settings: &mut self.settings,
            layout_version: &mut self.layout_version,
        }
    }

    // Version accessors
    pub fn binding_version(&self) -> u64 {
        self.binding_version
    }
    
    pub fn layout_version(&self) -> u64 {
        self.layout_version
    }
    
    // Convenience methods
    pub fn set_env_map(&mut self, texture_bundle: Option<(TextureHandle, &Texture)>) {
        let mut bindings = self.bindings_mut();
        bindings.env_map = texture_bundle.map(|(handle, _)| handle);

        let max_mip_count = texture_bundle
            .map(|(_, texture)| texture.mip_level_count())
            .unwrap_or(1);

        drop(bindings);
        self.uniforms_mut().env_map_max_mip_level = (max_mip_count - 1) as f32;
    }
    
    pub fn set_brdf_lut(&mut self, handle: Option<TextureHandle>) {
        self.bindings_mut().brdf_lut = handle;
    }
    
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.settings_mut().shadows_enabled = enabled;
    }
    
    pub fn set_fog_enabled(&mut self, enabled: bool) {
        self.settings_mut().fog_enabled = enabled;
    }
    
    pub fn set_ibl_enabled(&mut self, enabled: bool) {
        self.settings_mut().ibl_enabled = enabled;
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
        if gpu_lights.is_empty() {
            self.uniforms.write().num_lights = 0;
            return;
        }

        let old_count = self.light_storage.read().len();

        *self.light_storage.write() = gpu_lights;
        
        if old_count != self.light_storage.read().len() {
            self.uniforms.write().num_lights = self.light_storage.read().len() as u32;
        }
    }
}