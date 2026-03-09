use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureNodeId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdgTextureDesc {
    pub size: wgpu::Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

impl RdgTextureDesc {
    pub fn new(
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        mip_level_count: u32,
        sample_count: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers,
            },
            mip_level_count,
            sample_count,
            dimension,
            format,
            usage,
        }
    }

    pub fn new_2d(
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
        }
    }
}

pub struct ResourceRecord {
    pub name: &'static str,
    pub desc: RdgTextureDesc,
    pub is_external: bool,
    // Transient resources are short-lived textures that exist only within a single pass execution.
    // They are allocated from a shared pool and must be explicitly declared by passes.
    // Mostly, this is used for intermediate render targets
    pub is_inner_transient: bool,

    pub producers: SmallVec<[usize; 4]>,
    pub consumers: SmallVec<[usize; 8]>,

    pub first_use: usize,
    pub last_use: usize,
    pub physical_index: Option<usize>,
}
