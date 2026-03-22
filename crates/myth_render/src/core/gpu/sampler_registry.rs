use myth_resources::texture::TextureSampler;
use rustc_hash::FxHashMap;

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
    samplers: Vec<wgpu::Sampler>,

    lookup: FxHashMap<TextureSampler, usize>,

    global_anisotropy: u16,
}

impl SamplerRegistry {
    #[must_use]
    pub fn new(device: &wgpu::Device, global_anisotropy: u16) -> Self {
        let mut samplers = Vec::with_capacity(16);

        // Common samplers (first 4 slots, never removed)
        samplers.push(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Common: LinearClamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        samplers.push(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Common: NearestClamp"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        samplers.push(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Common: LinearRepeat"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        }));

        samplers.push(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Common: NearestRepeat"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        }));

        Self {
            samplers,
            lookup: FxHashMap::default(),
            global_anisotropy,
        }
    }

    pub fn set_global_anisotropy(&mut self, new_anisotropy: u16) {
        if self.global_anisotropy != new_anisotropy {
            self.global_anisotropy = new_anisotropy;
        }
    }

    #[must_use]
    #[inline]
    pub fn get_common(&self, typ: CommonSampler) -> &wgpu::Sampler {
        &self.samplers[typ as usize]
    }

    #[must_use]
    #[inline]
    pub fn default_sampler(&self) -> (usize, &wgpu::Sampler) {
        (
            CommonSampler::LinearClamp as usize,
            &self.samplers[CommonSampler::LinearClamp as usize],
        )
    }

    // This method is the only way to create new samplers, ensuring all samplers are tracked and deduplicated.
    pub fn get_custom(
        &mut self,
        device: &wgpu::Device,
        key: &TextureSampler,
    ) -> (usize, &wgpu::Sampler) {
        let mut actual_key = *key;

        let actual_af = match key.anisotropy_clamp {
            Some(explicit_val) => explicit_val,
            None => {
                if key.min_filter == wgpu::FilterMode::Linear
                    && key.mipmap_filter == wgpu::MipmapFilterMode::Linear
                {
                    self.global_anisotropy
                } else {
                    1
                }
            }
        };
        actual_key.anisotropy_clamp = Some(actual_af);

        if let Some(&index) = self.lookup.get(&actual_key) {
            return (index, &self.samplers[index]);
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Custom Sampler"),
            address_mode_u: actual_key.address_mode_u,
            address_mode_v: actual_key.address_mode_v,
            address_mode_w: actual_key.address_mode_w,
            mag_filter: actual_key.mag_filter,
            min_filter: actual_key.min_filter,
            mipmap_filter: actual_key.mipmap_filter,
            lod_min_clamp: actual_key.lod_min_clamp,
            lod_max_clamp: actual_key.lod_max_clamp,
            compare: actual_key.compare,
            anisotropy_clamp: actual_af,
            border_color: actual_key.border_color,
        });

        let new_index = self.samplers.len(); // get index before pushing
        self.samplers.push(sampler);
        self.lookup.insert(actual_key, new_index); // record the index corresponding to the configuration

        (new_index, &self.samplers[new_index])
    }

    #[inline]
    pub fn get_sampler_by_index(&self, index: usize) -> Option<&wgpu::Sampler> {
        self.samplers.get(index)
    }
}
