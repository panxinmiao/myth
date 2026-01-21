use glam::Vec4;

use crate::resources::{buffer::CpuBuffer, material::{BindingsGuard, MaterialBindings, MaterialSettings, SettingsGuard}, texture::TextureSource, uniforms::MeshPhongUniforms};


// MeshPhongMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshPhongMaterial {
    pub uniforms: CpuBuffer<MeshPhongUniforms>,
    bindings: MaterialBindings,
    settings: MaterialSettings,
    
    binding_version: u64,
    layout_version: u64,
}

impl MeshPhongMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhongUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhongUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            binding_version: 0,
            layout_version: 0,
        }
    }
    
    pub fn uniforms(&self) -> &MeshPhongUniforms {
        self.uniforms.read()
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshPhongUniforms> {
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
    
    pub fn set_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().map = texture.into();
    }
    
    pub fn set_normal_map(&mut self, texture: impl Into<Option<TextureSource>>) {
        self.bindings_mut().normal_map = texture.into();
    }
}

impl Default for MeshPhongMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}

