use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::core::gpu::Tracked;

pub trait GraphResourceType: 'static + Send + Sync + Copy + Clone {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Texture;

impl GraphResourceType for Texture {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Buffer;

impl GraphResourceType for Buffer {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErasedResourceNodeId {
    index: u32,
    version: u32,
}

impl ErasedResourceNodeId {
    #[inline]
    #[must_use]
    pub const fn new(index: u32, version: u32) -> Self {
        Self { index, version }
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[inline]
    #[must_use]
    pub const fn version(self) -> u32 {
        self.version
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceNodeId<T: GraphResourceType> {
    raw: ErasedResourceNodeId,
    _marker: PhantomData<fn() -> T>,
}

impl<T: GraphResourceType> ResourceNodeId<T> {
    #[inline]
    #[must_use]
    pub const fn new(index: u32, version: u32) -> Self {
        Self {
            raw: ErasedResourceNodeId::new(index, version),
            _marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self::new(index, 0)
    }

    #[inline]
    #[must_use]
    pub(crate) const fn from_erased(raw: ErasedResourceNodeId) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub const fn erase(self) -> ErasedResourceNodeId {
        self.raw
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.raw.index()
    }

    #[inline]
    #[must_use]
    pub const fn version(self) -> u32 {
        self.raw.version()
    }
}

impl<T: GraphResourceType> From<ResourceNodeId<T>> for ErasedResourceNodeId {
    #[inline]
    fn from(value: ResourceNodeId<T>) -> Self {
        value.erase()
    }
}

impl<T: GraphResourceType> PartialEq<ResourceNodeId<T>> for ErasedResourceNodeId {
    #[inline]
    fn eq(&self, other: &ResourceNodeId<T>) -> bool {
        *self == (*other).erase()
    }
}

impl<T: GraphResourceType> PartialEq<ErasedResourceNodeId> for ResourceNodeId<T> {
    #[inline]
    fn eq(&self, other: &ErasedResourceNodeId) -> bool {
        (*self).erase() == *other
    }
}

pub type TextureNodeId = ResourceNodeId<Texture>;
pub type BufferNodeId = ResourceNodeId<Buffer>;

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
    /// (e.g. an SSA alias produced by `mutate_texture`).  The RDG will
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub logical_size: u64,
    pub usage: wgpu::BufferUsages,
}

impl BufferDesc {
    pub const MIN_ALLOCATION_SIZE: u64 = 256;

    #[inline]
    #[must_use]
    pub const fn new(logical_size: u64, usage: wgpu::BufferUsages) -> Self {
        Self {
            logical_size,
            usage,
        }
    }

    #[inline]
    #[must_use]
    pub fn physical_allocation_size(&self) -> u64 {
        self.logical_size
            .max(1)
            .next_power_of_two()
            .max(Self::MIN_ALLOCATION_SIZE)
    }

    #[inline]
    #[must_use]
    pub fn logical_binding_size(&self) -> Option<wgpu::BufferSize> {
        wgpu::BufferSize::new(self.logical_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceClass {
    Texture,
    Buffer,
}

#[derive(Debug, Clone, Copy)]
pub enum ResourceKind {
    Texture {
        desc: TextureDesc,
        external_view_ptr: Option<*const Tracked<wgpu::TextureView>>,
    },
    Buffer {
        desc: BufferDesc,
        external_buffer_ptr: Option<*const Tracked<wgpu::Buffer>>,
    },
}

impl ResourceKind {
    #[inline]
    #[must_use]
    pub const fn texture(desc: TextureDesc) -> Self {
        Self::Texture {
            desc,
            external_view_ptr: None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn external_texture(
        desc: TextureDesc,
        external_view_ptr: *const Tracked<wgpu::TextureView>,
    ) -> Self {
        Self::Texture {
            desc,
            external_view_ptr: Some(external_view_ptr),
        }
    }

    #[inline]
    #[must_use]
    pub const fn buffer(desc: BufferDesc) -> Self {
        Self::Buffer {
            desc,
            external_buffer_ptr: None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn external_buffer(
        desc: BufferDesc,
        external_buffer_ptr: *const Tracked<wgpu::Buffer>,
    ) -> Self {
        Self::Buffer {
            desc,
            external_buffer_ptr: Some(external_buffer_ptr),
        }
    }

    #[inline]
    #[must_use]
    pub const fn class(&self) -> ResourceClass {
        match self {
            Self::Texture { .. } => ResourceClass::Texture,
            Self::Buffer { .. } => ResourceClass::Buffer,
        }
    }

    #[inline]
    #[must_use]
    pub const fn without_external_binding(self) -> Self {
        match self {
            Self::Texture { desc, .. } => Self::texture(desc),
            Self::Buffer { desc, .. } => Self::buffer(desc),
        }
    }

    #[inline]
    #[must_use]
    pub const fn texture_desc(&self) -> Option<TextureDesc> {
        match self {
            Self::Texture { desc, .. } => Some(*desc),
            Self::Buffer { .. } => None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn buffer_desc(&self) -> Option<BufferDesc> {
        match self {
            Self::Texture { .. } => None,
            Self::Buffer { desc, .. } => Some(*desc),
        }
    }

    #[inline]
    #[must_use]
    pub const fn external_texture_ptr(&self) -> Option<*const Tracked<wgpu::TextureView>> {
        match self {
            Self::Texture {
                external_view_ptr, ..
            } => *external_view_ptr,
            Self::Buffer { .. } => None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn external_buffer_ptr(&self) -> Option<*const Tracked<wgpu::Buffer>> {
        match self {
            Self::Texture { .. } => None,
            Self::Buffer {
                external_buffer_ptr,
                ..
            } => *external_buffer_ptr,
        }
    }
}

pub struct ResourceRecord {
    pub name: &'static str,
    pub is_external: bool,

    // 在严格 SSA 下，一个资源最多只能有一个生产者
    pub producer: Option<usize>,
    pub consumers: SmallVec<[usize; 8]>,

    pub first_use: usize,
    pub last_use: usize,
    pub physical_index: Option<usize>,

    pub kind: ResourceKind,

    /// If this resource is a versioned alias produced by
    /// [`PassBuilder::mutate_texture`], points to the root (non-alias)
    /// resource.  Aliased resources share the same physical GPU memory as
    /// their root ancestor, enabling in-place relay rendering (e.g.
    /// Opaque → Skybox → Transparent) without ambiguous read-write edges.
    pub alias_of: Option<ErasedResourceNodeId>,
}

impl ResourceRecord {
    #[inline]
    #[must_use]
    pub const fn class(&self) -> ResourceClass {
        self.kind.class()
    }

    #[inline]
    #[must_use]
    pub fn texture_desc(&self) -> TextureDesc {
        match self.kind {
            ResourceKind::Texture { desc, .. } => desc,
            ResourceKind::Buffer { .. } => {
                panic!("Resource '{}' is not a texture", self.name)
            }
        }
    }

    #[inline]
    #[must_use]
    pub fn buffer_desc(&self) -> BufferDesc {
        match self.kind {
            ResourceKind::Buffer { desc, .. } => desc,
            ResourceKind::Texture { .. } => {
                panic!("Resource '{}' is not a buffer", self.name)
            }
        }
    }

    #[inline]
    #[must_use]
    pub const fn external_texture_ptr(&self) -> Option<*const Tracked<wgpu::TextureView>> {
        self.kind.external_texture_ptr()
    }

    #[inline]
    #[must_use]
    pub const fn external_buffer_ptr(&self) -> Option<*const Tracked<wgpu::Buffer>> {
        self.kind.external_buffer_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_desc_rounds_to_power_of_two() {
        let desc = BufferDesc::new(4097, wgpu::BufferUsages::STORAGE);
        assert_eq!(desc.physical_allocation_size(), 8192);

        let tiny = BufferDesc::new(1, wgpu::BufferUsages::COPY_DST);
        assert_eq!(
            tiny.physical_allocation_size(),
            BufferDesc::MIN_ALLOCATION_SIZE
        );
    }

    #[test]
    fn test_buffer_desc_binding_size_tracks_logical_size() {
        let desc = BufferDesc::new(1024, wgpu::BufferUsages::UNIFORM);
        assert_eq!(
            desc.logical_binding_size().map(wgpu::BufferSize::get),
            Some(1024)
        );

        let empty = BufferDesc::new(0, wgpu::BufferUsages::STORAGE);
        assert!(empty.logical_binding_size().is_none());
    }
}
