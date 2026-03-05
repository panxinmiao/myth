use crate::renderer::core::resources::Tracked;
use rustc_hash::FxHashMap;
use wgpu::{Device, TextureView};

use super::types::RdgTextureDesc;

// --- Sub-View Key --------------------------------------------------------------

/// Discriminator for lazily-cached sub-views of a physical texture.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SubViewKey {
    pub base_mip: u32,
    pub mip_count: Option<u32>,
    pub base_layer: u32,
    pub layer_count: Option<u32>,
    pub aspect: wgpu::TextureAspect,
}

impl Default for SubViewKey {
    fn default() -> Self {
        Self {
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
            aspect: wgpu::TextureAspect::All,
        }
    }
}

// --- Physical Texture ----------------------------------------------------------

pub(crate) struct PhysicalTexture {
    /// Monotonically-increasing allocation identifier.
    pub(crate) uid: u64,
    pub(crate) desc: RdgTextureDesc,
    /// Raw wgpu texture handle -- kept alive for sub-view creation.
    pub(crate) texture: wgpu::Texture,
    /// Full-texture default view (all mips, all layers, aspect = All).
    pub(crate) default_view: Tracked<wgpu::TextureView>,
    /// Lazily-populated sub-view cache.
    pub(crate) sub_views: FxHashMap<SubViewKey, Tracked<wgpu::TextureView>>,
}

// --- RDG Transient Pool --------------------------------------------------------

pub struct RdgTransientPool {
    pub(crate) resources: Vec<PhysicalTexture>,
    active_allocations: Vec<usize>,
    uid_counter: u64,
}

impl RdgTransientPool {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            active_allocations: Vec::new(),
            uid_counter: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.active_allocations.fill(0);
    }

    pub fn acquire(
        &mut self,
        device: &Device,
        desc: &RdgTextureDesc,
        first_use: usize,
        last_use: usize,
    ) -> usize {
        for (i, res) in self.resources.iter().enumerate() {
            if self.active_allocations[i] <= first_use && res.desc == *desc {
                self.active_allocations[i] = last_use + 1;
                return i;
            }
        }

        let wgpu_desc = wgpu::TextureDescriptor {
            label: Some("RDG Transient Texture"),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        };

        let texture = device.create_texture(&wgpu_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.uid_counter += 1;

        let index = self.resources.len();
        self.resources.push(PhysicalTexture {
            uid: self.uid_counter,
            desc: desc.clone(),
            texture,
            default_view: Tracked::new(view),
            sub_views: FxHashMap::default(),
        });

        self.active_allocations.push(last_use + 1);
        index
    }

    #[inline]
    pub fn get_view(&self, index: usize) -> &TextureView {
        &self.resources[index].default_view
    }

    #[inline]
    pub fn get_tracked_view(&self, index: usize) -> &Tracked<wgpu::TextureView> {
        &self.resources[index].default_view
    }

    #[inline]
    pub fn get_texture(&self, index: usize) -> &wgpu::Texture {
        &self.resources[index].texture
    }

    #[inline]
    pub fn get_uid(&self, index: usize) -> u64 {
        self.resources[index].uid
    }

    pub fn get_or_create_sub_view(
        &mut self,
        physical_index: usize,
        key: SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let res = &mut self.resources[physical_index];
        res.sub_views.entry(key.clone()).or_insert_with(|| {
            let view = res.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("RDG Sub-View"),
                format: None,
                dimension: None,
                usage: None,
                aspect: key.aspect,
                base_mip_level: key.base_mip,
                mip_level_count: key.mip_count,
                base_array_layer: key.base_layer,
                array_layer_count: key.layer_count,
                ..Default::default()
            });
            Tracked::new(view)
        })
    }
}
