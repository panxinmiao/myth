use std::any::Any;

use glam::{Vec3, Vec4};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::texture::SamplerSource;
use crate::resources::{buffer::CpuBuffer, material::{MaterialBindings, MaterialFeatures, MaterialSettings, MaterialTrait, SettingsGuard}, texture::TextureSource, uniforms::MeshStandardUniforms};

// MeshStandardMaterial
// ----------------------------------------------------------------------------
#[derive(Debug)]
pub struct MeshStandardMaterial {
    pub uniforms: CpuBuffer<MeshStandardUniforms>,
    bindings: MaterialBindings,
    settings: MaterialSettings,
    
    /// 材质配置版本号（Settings 变化时递增）
    version: u64,
}

impl MeshStandardMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshStandardUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshStandardUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,
        }
    }
    
    pub fn uniforms(&self) -> &MeshStandardUniforms {
        self.uniforms.read()
    }
    
    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshStandardUniforms> {
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

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}

impl MaterialTrait for MeshStandardMaterial {
    fn shader_name(&self) -> &'static str {
        "mesh_standard"
    }

    fn features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::USE_IBL;
        if self.bindings.map.is_some() { features |= MaterialFeatures::USE_MAP; }
        if self.bindings.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
        if self.bindings.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
        if self.bindings.metalness_map.is_some() { features |= MaterialFeatures::USE_METALNESS_MAP; }
        if self.bindings.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
        if self.bindings.ao_map.is_some() { features |= MaterialFeatures::USE_AO_MAP; }
        if self.bindings.specular_map.is_some() { features |= MaterialFeatures::USE_SPECULAR_MAP; }
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
        if let Some(ref tex) = self.bindings.normal_map { visitor(tex); }
        if let Some(ref tex) = self.bindings.roughness_map { visitor(tex); }
        if let Some(ref tex) = self.bindings.metalness_map { visitor(tex); }
        if let Some(ref tex) = self.bindings.emissive_map { visitor(tex); }
        if let Some(ref tex) = self.bindings.ao_map { visitor(tex); }
        if let Some(ref tex) = self.bindings.specular_map { visitor(tex); }
    }

    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<MeshStandardUniforms>(
            "material",
            &self.uniforms,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );

        if let Some(map) = &self.bindings.map {
            builder.add_texture("map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            
            let sampler_source= &self.bindings.specular_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.normal_map {
            builder.add_texture("normal_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.normal_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("normal_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.roughness_map {
            builder.add_texture("roughness_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.roughness_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("roughness_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.metalness_map {
            builder.add_texture("metalness_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.metalness_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("metalness_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.ao_map {
            builder.add_texture("ao_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.ao_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("ao_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.emissive_map {
            builder.add_texture("emissive_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.emissive_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("emissive_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
        }

        if let Some(map) = &self.bindings.specular_map {
            builder.add_texture("specular_map", Some(*map), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
            let sampler_source= &self.bindings.specular_map_sampler.or(
                match map {
                    TextureSource::Asset(handle) => Some(SamplerSource::FromTexture(*handle)),
                    TextureSource::Attachment(_) => None,
                }
            );
            builder.add_sampler("specular_map", *sampler_source, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
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
