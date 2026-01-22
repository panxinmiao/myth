use std::any::Any;

use glam::Vec4;

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::{buffer::CpuBuffer, material::{MaterialBindings, MaterialFeatures, MaterialSettings, MaterialTrait, SettingsGuard}, texture::TextureSource, uniforms::MeshBasicUniforms};

// MeshBasicMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub uniforms: CpuBuffer<MeshBasicUniforms>,
    bindings: MaterialBindings,
    settings: MaterialSettings,
    
    /// 材质配置版本号（Settings 变化时递增）
    version: u64,
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
            version: 0,
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
    
    /// 直接修改 bindings（纹理变化由 ResourceIdSet 自动检测）
    pub fn bindings_mut(&mut self) -> &mut MaterialBindings {
        &mut self.bindings
    }
    
    pub fn settings(&self) -> &MaterialSettings {
        &self.settings
    }
    
    pub fn settings_mut(&mut self) -> SettingsGuard<'_> {
        SettingsGuard {
            initial_settings: self.settings.clone(),
            settings: &mut self.settings,
            version: &mut self.version,
        }
    }
    
    pub fn version(&self) -> u64 { self.version }
    
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

impl MaterialTrait for MeshBasicMaterial {
    fn shader_name(&self) -> &'static str {
        "mesh_basic"
    }

    fn features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        if self.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        features
    }

    fn settings(&self) -> &MaterialSettings {
        &self.settings
    }

    fn bindings(&self) -> &MaterialBindings {
        &self.bindings
    }

    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource)) {
        if let Some(ref tex) = self.bindings.map { visitor(tex); }
    }

    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<MeshBasicUniforms>(
            "material",
            &self.uniforms,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );

        if let Some(map) = &self.bindings.map {
            builder.add_texture("map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            builder.add_sampler("map", Some(*map), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }
    }

    fn uniform_buffer(&self) -> &BufferRef {
        self.uniforms.handle()
    }

    fn uniform_bytes(&self) -> &[u8] {
        self.uniforms.as_bytes()
    }

    fn version(&self) -> u64 {
        self.version
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}