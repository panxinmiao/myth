use std::collections::HashMap;
use crate::renderer::core::gpu::Tracked;

use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy)]
pub struct SamplerKey {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<wgpu::CompareFunction>,
    pub anisotropy_clamp: u16,
    pub border_color: Option<wgpu::SamplerBorderColor>,
}

impl SamplerKey {
    pub const LINEAR_CLAMP: Self = Self {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    };
}

impl PartialEq for SamplerKey {
    fn eq(&self, other: &Self) -> bool {
        self.address_mode_u == other.address_mode_u &&
        self.address_mode_v == other.address_mode_v &&
        self.address_mode_w == other.address_mode_w &&
        self.mag_filter == other.mag_filter &&
        self.min_filter == other.min_filter &&
        self.mipmap_filter == other.mipmap_filter &&
        self.lod_min_clamp.to_bits() == other.lod_min_clamp.to_bits() &&
        self.lod_max_clamp.to_bits() == other.lod_max_clamp.to_bits() &&
        self.compare == other.compare &&
        self.anisotropy_clamp == other.anisotropy_clamp &&
        self.border_color == other.border_color
    }
}
impl Eq for SamplerKey {}

impl Hash for SamplerKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.address_mode_u.hash(state);
        self.address_mode_v.hash(state);
        self.address_mode_w.hash(state);
        self.mag_filter.hash(state);
        self.min_filter.hash(state);
        self.mipmap_filter.hash(state);
        self.lod_min_clamp.to_bits().hash(state);
        self.lod_max_clamp.to_bits().hash(state);
        self.compare.hash(state);
        self.anisotropy_clamp.hash(state);
        self.border_color.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CommonSampler {
    LinearClamp = 0,
    NearestClamp = 1,
    LinearRepeat = 2,
    NearestRepeat = 3,
}

impl CommonSampler {
    pub const COUNT: usize = 4;
}

pub struct SamplerRegistry {
    common_samplers: [Tracked<wgpu::Sampler>; CommonSampler::COUNT],
    custom_samplers: HashMap<SamplerKey, Tracked<wgpu::Sampler>>,
}

impl SamplerRegistry {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let common_samplers = [
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Common: LinearClamp"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })),
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Common: NearestClamp"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })),
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Common: LinearRepeat"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            })),
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Common: NearestRepeat"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            })),
        ];

        Self {
            common_samplers,
            custom_samplers: HashMap::new(),
        }
    }

    #[must_use]
    pub fn get_common(&self, typ: CommonSampler) -> &Tracked<wgpu::Sampler> {
        &self.common_samplers[typ as usize]
    }

    pub fn get_custom(
        &mut self,
        device: &wgpu::Device,
        key: SamplerKey,
    ) -> &Tracked<wgpu::Sampler> {
        self.custom_samplers.entry(key).or_insert_with(|| {
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Custom Sampler"),
                address_mode_u: key.address_mode_u,
                address_mode_v: key.address_mode_v,
                address_mode_w: key.address_mode_w,
                mag_filter: key.mag_filter,
                min_filter: key.min_filter,
                mipmap_filter: key.mipmap_filter,
                lod_min_clamp: key.lod_min_clamp,
                lod_max_clamp: key.lod_max_clamp,
                compare: key.compare,
                anisotropy_clamp: key.anisotropy_clamp,
                border_color: key.border_color,
            }))
        })
    }

    #[must_use]
    pub fn get_custom_ref(&self, key: &SamplerKey) -> &Tracked<wgpu::Sampler> {
        self.custom_samplers
            .get(key)
            .expect("Custom sampler not found — call get_custom() first to ensure creation")
    }
}
