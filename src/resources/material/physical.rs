use glam::{Vec3, Vec4};

use crate::resources::{buffer::CpuBuffer, material::{BindingsGuard, MaterialBindings, MaterialSettings, SettingsGuard}, texture::TextureSource, uniforms::{MeshPhysicalUniforms}};

// MeshPhysicalMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub uniforms: CpuBuffer<MeshPhysicalUniforms>,
    bindings: MaterialBindings,
    settings: MaterialSettings,
    
    binding_version: u64,
    layout_version: u64,
}

impl MeshPhysicalMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhysicalUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhysicalUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms(&self) -> &MeshPhysicalUniforms {
        self.uniforms.read()
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshPhysicalUniforms> {
        self.uniforms.write()
    }
    
    pub fn bindings(&self) -> &MaterialBindings {
        &self.bindings
    }
    
    pub fn bindings_mut(&mut self) -> BindingsGuard<'_> {
        BindingsGuard {
            initial_layout: self.bindings.clone(),
            bindings: &mut self.bindings,
            binding_version: &mut self.binding_version,
            layout_version: &mut self.layout_version,
        }
    }
    
    pub fn settings(&self) -> &MaterialSettings {
        &self.settings
    }
    
    pub fn settings_mut(&mut self) -> SettingsGuard<'_> {
        SettingsGuard {
            initial_settings: self.settings.clone(),
            settings: &mut self.settings,
            layout_version: &mut self.layout_version,
        }
    }
    
    pub fn binding_version(&self) -> u64 { self.binding_version }
    pub fn layout_version(&self) -> u64 { self.layout_version }
    
    pub fn set_color(&mut self, color: Vec4) {
        self.uniforms.write().color = color;
    }
    
    pub fn set_roughness(&mut self, roughness: f32) {
        self.uniforms.write().roughness = roughness;
    }
    
    pub fn set_metalness(&mut self, metalness: f32) {
        self.uniforms.write().metalness = metalness;
    }
    
    pub fn set_emissive(&mut self, emissive: Vec3) {
        self.uniforms.write().emissive = emissive;
    }

    pub fn set_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().map = texture.into();
    }
    
    pub fn set_normal_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().normal_map = texture.into();
    }
    
    pub fn set_roughness_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().roughness_map = texture.into();
    }
    
    pub fn set_metalness_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().metalness_map = texture.into();
    }

    pub fn set_emissive_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().emissive_map = texture.into();
    }
    
    pub fn set_ao_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().ao_map = texture.into();
    }
}

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
