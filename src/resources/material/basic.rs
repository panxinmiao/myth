use glam::Vec4;

use crate::resources::{buffer::CpuBuffer, material::{BindingsGuard, MaterialBindings, MaterialSettings, SettingsGuard}, texture::TextureSource, uniforms::MeshBasicUniforms};

// MeshBasicMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub uniforms: CpuBuffer<MeshBasicUniforms>,
    bindings: MaterialBindings,
    settings: MaterialSettings,
    
    binding_version: u64,
    layout_version: u64,
}

impl MeshBasicMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshBasicUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data, 
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshBasicUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms(&self) -> &MeshBasicUniforms {
        self.uniforms.read()
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshBasicUniforms> {
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
    
    pub fn set_opacity(&mut self, opacity: f32) {
        self.uniforms.write().opacity = opacity;
    }
    
    pub fn set_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().map = texture.into();
    }
}

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}