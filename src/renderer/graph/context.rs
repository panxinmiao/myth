//! 渲染上下文
//!
//! `RenderContext` 在渲染图的各个 Pass 之间传递共享数据，
//! 避免参数列表过长，统一数据访问方式。

use crate::scene::camera::RenderCamera;
use crate::scene::{Scene};
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::pipeline::PipelineCache;
use crate::assets::AssetServer;
use crate::renderer::graph::{RenderState, ExtractedScene};

/// 渲染上下文
/// 
/// 在 RenderGraph 执行期间，所有 RenderNode 共享此上下文。
/// 包含 GPU 上下文、资源管理器、场景数据等。
/// 
/// # 性能考虑
/// - 所有字段都是引用，避免数据复制
/// - `surface_view` 每帧更新，指向当前交换链纹理
/// - 通过借用规则确保线程安全
pub struct RenderContext<'a> {
    /// WGPU 核心上下文（device, queue, surface）
    pub wgpu_ctx: &'a WgpuContext,
    /// GPU 资源管理器
    pub resource_manager: &'a mut ResourceManager,
    /// Pipeline 缓存
    pub pipeline_cache: &'a mut PipelineCache,
    /// 资产服务器
    pub assets: &'a AssetServer,
    /// 当前场景
    pub scene: &'a mut Scene,
    /// 相机
    pub camera: &'a RenderCamera,
    /// 当前帧的 Surface View
    pub surface_view: &'a wgpu::TextureView,
    /// 渲染状态
    pub render_state: &'a RenderState,
    /// 提取的场景数据
    pub extracted_scene: &'a ExtractedScene,
    /// 当前时间
    pub time: f32,
}
