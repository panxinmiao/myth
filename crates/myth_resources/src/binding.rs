//! GPU binding resource descriptions.
//!
//! These types describe how CPU-side resources map to GPU bind group entries.
//! The actual bind group creation is performed by the renderer.

use crate::buffer::BufferRef;
use crate::texture::{SamplerSource, TextureSource};

/// Describes a single resource binding for [`wgpu::BindGroup`] creation.
#[derive(Debug, Clone)]
pub enum BindingResource<'a> {
    /// A buffer binding with optional inline data.
    Buffer {
        /// The CPU-side buffer reference.
        buffer: BufferRef,
        /// Byte offset into the buffer.
        offset: u64,
        /// Optional explicit size (defaults to the entire buffer).
        size: Option<u64>,
        /// Optional inline data for immediate upload.
        data: Option<&'a [u8]>,
    },
    /// A texture binding.
    Texture(Option<TextureSource>),
    /// A sampler binding.
    Sampler(Option<SamplerSource>),
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<&'a ()>),
}
