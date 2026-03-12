use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureNodeId(pub u32);

/// Describes the initial load behaviour for an RDG color attachment.
///
/// Unlike the raw `Option<wgpu::Color>` convention, this enum makes the
/// caller's **intent** explicit and enables compile-time and runtime
/// validation by the render graph framework.
///
/// # Variants
///
/// | Variant    | GPU Effect                | When to Use |
/// |------------|---------------------------|-------------|
/// | `Clear(c)` | `LoadOp::Clear(c)`       | Passes that need a known background colour. |
/// | `Load`     | `LoadOp::Load`            | Relay / alias passes that inherit prior content. |
/// | `DontCare` | `LoadOp::Clear(BLACK)`    | Full-screen replace shaders that overwrite every pixel. |
///
/// `DontCare` maps to a zero-cost `Clear` rather than a true `DontCare`
/// because TBDR mobile GPUs must otherwise read back tile memory, incurring
/// significant bandwidth overhead.
#[derive(Debug, Clone, Copy)]
pub enum RenderTargetOps {
    /// Clear the attachment to a specific colour (maps to `LoadOp::Clear`).
    Clear(wgpu::Color),
    /// Preserve existing content (maps to `LoadOp::Load`).
    ///
    /// Only legal when the resource was already written in the current frame
    /// (e.g. an SSA alias produced by `mutate_and_export`).  The RDG will
    /// panic if this is used on a freshly created transient resource that
    /// has never been written.
    Load,
    /// Content is irrelevant — the shader will overwrite every pixel.
    ///
    /// Maps to `LoadOp::DontCare` to avoid TBDR read-back overhead
    /// while conveying the semantic that initial content does not matter.
    DontCare,
}

impl RenderTargetOps {
    /// Convert to the corresponding `wgpu::LoadOp`.
    #[inline]
    #[must_use]
    pub fn to_wgpu_load_op(self) -> wgpu::LoadOp<wgpu::Color> {
        match self {
            Self::Clear(c) => wgpu::LoadOp::Clear(c),
            Self::Load => wgpu::LoadOp::Load,
            Self::DontCare => wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
        }
    }
}

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

    /// If this resource is a versioned alias produced by
    /// [`PassBuilder::mutate_and_export`], points to the root (non-alias)
    /// resource.  Aliased resources share the same physical GPU memory as
    /// their root ancestor, enabling in-place relay rendering (e.g.
    /// Opaque → Skybox → Transparent) without ambiguous read-write edges.
    pub alias_of: Option<TextureNodeId>,
}
