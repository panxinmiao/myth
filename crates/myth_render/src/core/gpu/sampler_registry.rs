use crate::core::gpu::Tracked;
use myth_resources::texture::TextureSampler;
use std::collections::HashMap;

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
    custom_samplers: HashMap<TextureSampler, Tracked<wgpu::Sampler>>,
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
        key: TextureSampler,
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
    pub fn get_custom_ref(&self, key: &TextureSampler) -> &Tracked<wgpu::Sampler> {
        self.custom_samplers
            .get(key)
            .expect("Custom sampler not found — call get_custom() first to ensure creation")
    }
}
