use glam::{Vec4, Vec3};
use std::borrow::Cow;

use crate::resources::{material::*, texture::TextureSource};

pub struct MeshBasicMaterialBuilder {
    // Specific Properties
    color: Vec4,
    map: Option<TextureSource>,

    // Common Properties (默认值)
    name: Option<Cow<'static, str>>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    cull_mode: Option<wgpu::Face>,
    side: Side,
}

impl Default for MeshBasicMaterialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshBasicMaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Vec4::ONE,
            map: None,
            // Common Defaults
            name: None,
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: Side::Double,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.map = map.into() ; self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(Cow::Owned(name.to_string())); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self { self.cull_mode = mode; self }
    pub fn side(mut self, side: Side) -> Self { self.side = side; self }

    pub fn build(self) -> MeshBasicMaterial {
        let mut basic = MeshBasicMaterial::new(self.color);
        basic.bindings_mut().map = self.map;
        basic.uniforms_mut().opacity = self.opacity;
        
        {
            let mut settings = basic.settings_mut();
            settings.transparent = self.transparent;
            settings.depth_write = self.depth_write;
            settings.depth_test = self.depth_test;
            settings.side = self.side;
        }

        basic
    }
}

/// MeshStandardMaterial 的构建器
pub struct MeshStandardMaterialBuilder {
    // Specific
    color: Vec4,
    roughness: f32,
    metalness: f32,
    emissive: Vec3,
    map: Option<TextureSource>,
    normal_map: Option<TextureSource>,
    roughness_map: Option<TextureSource>,
    metalness_map: Option<TextureSource>,
    emissive_map: Option<TextureSource>,
    ao_map: Option<TextureSource>,

    // Common
    name: Option<Cow<'static, str>>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    side: Side,
}

impl Default for MeshStandardMaterialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshStandardMaterialBuilder {
    pub fn new() -> Self {
        Self {
            color: Vec4::ONE,
            roughness: 0.5,
            metalness: 0.5,
            emissive: Vec3::ZERO,
            map: None, normal_map: None, roughness_map: None, metalness_map: None, emissive_map: None, ao_map: None,
            // Common Defaults
            name: None,
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            side: Side::Double,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn roughness(mut self, value: f32) -> Self { self.roughness = value; self }
    pub fn metalness(mut self, value: f32) -> Self { self.metalness = value; self }
    pub fn emissive(mut self, value: Vec3) -> Self { self.emissive = value; self }
    
    pub fn map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.map = map.into(); self }
    pub fn normal_map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.normal_map = map.into(); self }
    pub fn roughness_map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.roughness_map = map.into(); self }
    pub fn metalness_map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.metalness_map = map.into(); self }
    pub fn emissive_map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.emissive_map = map.into(); self }
    pub fn ao_map(mut self, map: impl Into<Option<TextureSource>>) -> Self { self.ao_map = map.into(); self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(Cow::Owned(name.to_string())); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn side(mut self, side: Side) -> Self { self.side = side; self }
    
    pub fn build(self) -> MeshStandardMaterial {
        let mut standard = MeshStandardMaterial::new(self.color);
        
        {
            let mut uniforms = standard.uniforms_mut();
            uniforms.color = self.color;
            uniforms.emissive = Vec3::ZERO;
            uniforms.roughness = self.roughness;
            uniforms.metalness = self.metalness;
            uniforms.occlusion_strength = 1.0;
        }

        {
            let bindings = standard.bindings_mut();
            bindings.map = self.map;
            bindings.normal_map = self.normal_map;
            bindings.roughness_map = self.roughness_map;
            bindings.metalness_map = self.metalness_map;
            bindings.emissive_map = self.emissive_map;
            bindings.ao_map = self.ao_map;
        }

        {
            let mut settings = standard.settings_mut();
            settings.transparent = self.transparent;
            settings.depth_write = self.depth_write;
            settings.depth_test = self.depth_test;
            settings.side = self.side;
        }

        standard
    }
}
