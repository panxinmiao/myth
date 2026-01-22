use std::any::Any;

use glam::Vec4;

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::{buffer::CpuBuffer, material::{BindingsGuard, MaterialBindings, MaterialFeatures, MaterialSettings, MaterialTrait, SettingsGuard}, texture::TextureSource, uniforms::MeshPhongUniforms};


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

impl MaterialTrait for MeshPhongMaterial {
    fn shader_name(&self) -> &'static str {
        "mesh_phong"
    }

    fn features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        if self.bindings.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
        if self.bindings.specular_map.is_some() { features |= MaterialFeatures::USE_SPECULAR_MAP; }
        if self.bindings.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
        features
    }

    fn settings(&self) -> &MaterialSettings {
        &self.settings
    }

    fn bindings(&self) -> &MaterialBindings {
        &self.bindings
    }

    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<MeshPhongUniforms>(
            "material",
            &self.uniforms,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );

        if let Some(map) = &self.bindings.map {
            builder.add_texture("map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            builder.add_sampler("map", Some(*map), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.normal_map {
            builder.add_texture("normal_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            builder.add_sampler("normal_map", Some(*map), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.specular_map {
            builder.add_texture("specular_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            builder.add_sampler("specular_map", Some(*map), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.emissive_map {
            builder.add_texture("emissive_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            builder.add_sampler("emissive_map", Some(*map), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }
    }

    fn uniform_buffer(&self) -> &BufferRef {
        self.uniforms.handle()
    }

    fn uniform_bytes(&self) -> &[u8] {
        self.uniforms.as_bytes()
    }

    fn uniform_version(&self) -> u64 {
        self.uniforms.handle().version
    }

    fn layout_version(&self) -> u64 {
        self.layout_version
    }

    fn binding_version(&self) -> u64 {
        self.binding_version
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

