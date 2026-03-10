use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureNodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    pub size: wgpu::Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

impl TextureDesc {
    #[must_use]
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

    #[must_use]
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
    pub desc: TextureDesc,
    pub is_external: bool,

    // 在严格 SSA 下，一个资源最多只能有一个生产者
    pub producer: Option<usize>,
    pub consumers: SmallVec<[usize; 8]>,

    pub first_use: usize,
    pub last_use: usize,
    pub physical_index: Option<usize>,

    /// If this resource is a versioned alias produced by [`mutate_texture`],
    /// points to the previous logical version.  Aliased resources share the
    /// same physical GPU memory as their root ancestor, enabling in-place
    /// relay rendering (e.g. Opaque → Skybox → Transparent) without
    /// ambiguous read-write edges in the dependency graph.
    pub alias_of: Option<TextureNodeId>,
}
