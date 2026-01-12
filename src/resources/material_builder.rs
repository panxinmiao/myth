use uuid::Uuid;
use std::sync::atomic::{AtomicU64};
use glam::{Vec4, Vec3};

use crate::assets::TextureHandle;
use crate::resources::material::*;

pub struct MeshBasicMaterialBuilder {
    // Specific Properties
    color: Vec4,
    map: Option<TextureHandle>,

    // Common Properties (默认值)
    name: Option<String>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    cull_mode: Option<wgpu::Face>,
    side: u32,
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
            side: 0,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn map(mut self, map: TextureHandle) -> Self { self.map = Some(map); self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(name.into()); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self { self.cull_mode = mode; self }
    pub fn side(mut self, side: u32) -> Self { self.side = side; self }

    /// 构建最终的 Material
    pub fn build(self) -> Material {
        let mut basic = MeshBasicMaterial::new(self.color);
        basic.map = self.map;

        Material {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
            name: self.name,
            data: MaterialData::Basic(basic), // 自动装箱
            transparent: self.transparent,
            depth_write: self.depth_write,
            depth_test: self.depth_test,
            cull_mode: self.cull_mode,
            side: self.side,
        }
    }
}

/// MeshStandardMaterial 的构建器
pub struct MeshStandardMaterialBuilder {
    // Specific
    color: Vec4,
    roughness: f32,
    metalness: f32,
    emissive: Vec3,
    map: Option<TextureHandle>,
    normal_map: Option<TextureHandle>,
    roughness_map: Option<TextureHandle>,
    metalness_map: Option<TextureHandle>,
    emissive_map: Option<TextureHandle>,
    ao_map: Option<TextureHandle>,

    // Common
    name: Option<String>,
    transparent: bool,
    opacity: f32,
    depth_write: bool,
    depth_test: bool,
    cull_mode: Option<wgpu::Face>,
    side: u32,
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
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    // --- Specific Setters ---
    pub fn color(mut self, color: Vec4) -> Self { self.color = color; self }
    pub fn roughness(mut self, value: f32) -> Self { self.roughness = value; self }
    pub fn metalness(mut self, value: f32) -> Self { self.metalness = value; self }
    pub fn emissive(mut self, value: Vec3) -> Self { self.emissive = value; self }
    
    pub fn map(mut self, map: TextureHandle) -> Self { self.map = Some(map); self }
    pub fn normal_map(mut self, map: TextureHandle) -> Self { self.normal_map = Some(map); self }
    pub fn roughness_map(mut self, map: TextureHandle) -> Self { self.roughness_map = Some(map); self }
    pub fn metalness_map(mut self, map: TextureHandle) -> Self { self.metalness_map = Some(map); self }
    pub fn emissive_map(mut self, map: TextureHandle) -> Self { self.emissive_map = Some(map); self }
    pub fn ao_map(mut self, map: TextureHandle) -> Self { self.ao_map = Some(map); self }

    // --- Common Setters ---
    pub fn name(mut self, name: &str) -> Self { self.name = Some(name.into()); self }
    pub fn transparent(mut self, transparent: bool) -> Self { self.transparent = transparent; self }
    pub fn opacity(mut self, opacity: f32) -> Self { self.opacity = opacity; self }
    pub fn depth_write(mut self, enabled: bool) -> Self { self.depth_write = enabled; self }
    pub fn depth_test(mut self, enabled: bool) -> Self { self.depth_test = enabled; self }
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self { self.cull_mode = mode; self }
    pub fn side(mut self, side: u32) -> Self { self.side = side; self }
    
    pub fn build(self) -> Material {
        let mut standard = MeshStandardMaterial::new(self.color);
        standard.uniforms.color = self.color;
        standard.occlusion_strength = 1.0;
        standard.uniforms.emissive = Vec3::ZERO;
        standard.uniforms.roughness = self.roughness;
        standard.uniforms.metalness = self.metalness;

        standard.map = self.map;
        standard.normal_map = self.normal_map;
        standard.roughness_map = self.roughness_map;
        standard.metalness_map = self.metalness_map;
        standard.emissive_map = self.emissive_map;
        standard.ao_map = self.ao_map;

        Material {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
            name: self.name,
            data: MaterialData::Standard(standard),
            transparent: self.transparent,
            depth_write: self.depth_write,
            depth_test: self.depth_test,
            cull_mode: self.cull_mode,
            side: self.side,
        }
    }
}
