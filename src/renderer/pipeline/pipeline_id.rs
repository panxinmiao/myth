//! Strongly-typed pipeline handles.
//!
//! Thin `Copy` wrappers around a `u32` index into the central [`PipelineCache`]
//! storage arrays. Using distinct newtypes prevents accidentally mixing up
//! render and compute pipeline handles.
//!
//! [`PipelineCache`]: super::cache::PipelineCache

/// Handle to a cached `wgpu::RenderPipeline`.
///
/// Returned by [`PipelineCache::get_or_create_render`] and friends.
/// Resolve to an actual pipeline reference via [`PipelineCache::get_render_pipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPipelineId(pub(crate) u32);

impl RenderPipelineId {
    /// Raw index into the pipeline storage array.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Handle to a cached `wgpu::ComputePipeline`.
///
/// Returned by [`PipelineCache::get_or_create_compute`].
/// Resolve to an actual pipeline reference via [`PipelineCache::get_compute_pipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineId(pub(crate) u32);

impl ComputePipelineId {
    /// Raw index into the pipeline storage array.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
