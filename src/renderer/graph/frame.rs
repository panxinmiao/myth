//! 渲染帧管理
//!
//! `RenderFrame` 负责：
//! - 持有内置渲染 Pass（BRDF LUT、IBL、Forward）
//! - 持有提取的场景数据（`ExtractedScene`）和渲染状态（`RenderState`）
//! - 提供 Extract 和 Prepare 阶段的执行
//!
//! # 三阶段渲染架构
//!
//! 渲染流程分为三个明确的阶段：
//!
//! 1. **Prepare (准备)**：`extract_and_prepare()` - 提取场景数据，准备 GPU 资源
//! 2. **Compose (组装)**：通过 `FrameComposer` 链式添加渲染节点
//! 3. **Execute (执行)**：`FrameComposer::render()` - 获取 Surface 并提交 GPU 命令
//!
//! # 示例
//!
//! ```ignore
//! // 优雅的链式调用
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .render();
//! ```

use glam::Mat4;
use rustc_hash::FxHashMap;

use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::renderer::core::{BindGroupContext, RenderView, ResourceManager};
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use super::extracted::ExtractedScene;
use super::render_state::RenderState;

// ============================================================================
// RenderCommand & RenderLists
// ============================================================================

/// 单个渲染命令
///
/// 包含绘制一个物体所需的全部信息，由 `CullPass` 生成，供 OpaquePass/TransparentPass 消费。
///
/// # 性能考虑
/// - Pipeline 通过 clone `获得（wgpu::RenderPipeline` 内部是 Arc）
/// - `dynamic_offset` 支持动态 Uniform 缓冲
/// - `sort_key` 用于高效排序（Front-to-Back / Back-to-Front）
pub struct RenderCommand {
    /// 物体级别的 BindGroup（模型矩阵、骨骼等）
    pub object_bind_group: BindGroupContext,
    /// 几何体句柄
    pub geometry_handle: GeometryHandle,
    /// 材质句柄
    pub material_handle: MaterialHandle,
    /// Pipeline ID（用于状态追踪，避免冗余切换）
    pub pipeline_id: u16,
    /// `渲染管线（wgpu::RenderPipeline` 内部已经是 Arc）
    pub pipeline: wgpu::RenderPipeline,
    /// 模型世界矩阵
    pub model_matrix: Mat4,
    /// 排序键
    pub sort_key: RenderKey,
    /// 动态 Uniform 偏移
    pub dynamic_offset: u32,
}

pub struct ShadowRenderCommand {
    pub object_bind_group: BindGroupContext,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
    pub pipeline: wgpu::RenderPipeline,
    pub dynamic_offset: u32,
}

pub struct ShadowLightInstance {
    pub light_id: u64,
    pub layer_index: u32,
    pub light_buffer_index: usize,
    pub light_view_projection: Mat4,
}

/// Prepared skybox draw state for inline rendering (LDR path).
///
/// Populated by [`SkyboxPass::prepare()`] and consumed by
/// [`SimpleForwardPass::run()`] to draw the skybox between
/// opaque and transparent objects within a single render pass.
pub struct PreparedSkyboxDraw {
    /// The skybox render pipeline (variant-specific).
    pub pipeline: wgpu::RenderPipeline,
    /// The skybox bind group (uniforms + optional texture/sampler).
    pub bind_group: wgpu::BindGroup,
}

impl PreparedSkyboxDraw {
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, global_bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// 渲染列表
///
/// 存储经过剔除和排序的渲染命令，由 `SceneCullPass` 填充，
/// 供 `OpaquePass`、`TransparentPass` 和 `SimpleForwardPass` 消费。
///
/// # 设计原则
/// - **数据与逻辑分离**：仅存储数据，不包含渲染逻辑
/// - **帧间复用**：预分配内存，每帧 `clear()` 后重用
/// - **可扩展**：未来可添加 `alpha_test`、`shadow_casters` 等列表
pub struct RenderLists {
    /// 不透明物体命令列表（Front-to-Back 排序）
    pub opaque: Vec<RenderCommand>,
    /// 透明物体命令列表（Back-to-Front 排序）
    pub transparent: Vec<RenderCommand>,
    /// Shadow command queues, keyed by `(light_id, layer_index)` for per-view culling.
    ///
    /// Each cascade of a directional light (or each spot light) gets its own queue.
    pub shadow_queues: FxHashMap<(u64, u32), Vec<ShadowRenderCommand>>,
    pub shadow_lights: Vec<ShadowLightInstance>,

    /// All active render views for the current frame.
    ///
    /// Populated by `SceneCullPass`, consumed by `ShadowPass` and other passes.
    /// Contains main camera view + all shadow views.
    pub active_views: Vec<RenderView>,

    /// 全局 `BindGroup` ID（用于状态追踪）
    pub gpu_global_bind_group_id: u64,
    /// 全局 BindGroup（相机、光照、环境等）
    pub gpu_global_bind_group: Option<wgpu::BindGroup>,

    /// 是否需要 Transmission 拷贝
    pub use_transmission: bool,

    /// Prepared skybox draw state for inline rendering in the LDR path.
    ///
    /// Set by `SkyboxPass::prepare()`, consumed by `SimpleForwardPass::run()`.
    pub prepared_skybox: Option<PreparedSkyboxDraw>,
}

impl RenderLists {
    /// 创建空的渲染列表，预分配默认容量
    #[must_use]
    pub fn new() -> Self {
        Self {
            opaque: Vec::with_capacity(512),
            transparent: Vec::with_capacity(128),
            shadow_queues: FxHashMap::default(),
            shadow_lights: Vec::with_capacity(16),
            active_views: Vec::with_capacity(16),
            gpu_global_bind_group_id: 0,
            gpu_global_bind_group: None,
            use_transmission: false,
            prepared_skybox: None,
        }
    }

    /// 清空列表（保留容量以复用内存）
    #[inline]
    pub fn clear(&mut self) {
        self.opaque.clear();
        self.transparent.clear();
        self.shadow_queues.clear();
        self.shadow_lights.clear();
        self.active_views.clear();
        self.gpu_global_bind_group = None;
        self.use_transmission = false;
        self.prepared_skybox = None;
    }

    /// 插入不透明命令
    #[inline]
    pub fn insert_opaque(&mut self, cmd: RenderCommand) {
        self.opaque.push(cmd);
    }

    /// 插入透明命令
    #[inline]
    pub fn insert_transparent(&mut self, cmd: RenderCommand) {
        self.transparent.push(cmd);
    }

    /// 对命令列表进行排序
    ///
    /// - 不透明：按 Pipeline > Material > Depth (Front-to-Back)
    /// - 透明：按 Depth (Back-to-Front) > Pipeline > Material
    pub fn sort(&mut self) {
        self.opaque
            .sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        self.transparent
            .sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
    }

    /// 检查列表是否为空
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }
}

impl Default for RenderLists {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RenderKey - 排序键
// ============================================================================

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
///
/// 使用 64 位整数编码排序信息，支持高效的基数排序。
///
/// # 排序策略
/// - **不透明物体**：Pipeline > Material > Depth (Front-to-Back)
///   - 最小化 Pipeline 切换开销
///   - Front-to-Back 利用 Early-Z 提升性能
/// - **透明物体**：Depth (Back-to-Front) > Pipeline > Material
///   - 确保正确的 Alpha 混合顺序
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    /// 构造排序键
    ///
    /// # 参数
    /// - `pipeline_id`: Pipeline 索引（14 bits）
    /// - `material_index`: 材质索引（20 bits）
    /// - `depth`: 到相机的平方距离
    /// - `transparent`: 是否为透明物体
    #[must_use]
    pub fn new(pipeline_id: u16, material_index: u32, depth: f32, transparent: bool) -> Self {
        // 1. 统一处理深度压缩 (30 bits)
        // 注意：这里假设 depth >= 0.0。如果 depth 可能为负，clamp 到 0 是安全的。
        let d_u32 = if depth.is_sign_negative() {
            0
        } else {
            depth.to_bits() >> 2
        };
        let raw_d_bits = u64::from(d_u32) & 0x3FFF_FFFF;

        // 2. 准备其他数据
        let p_bits = u64::from(pipeline_id & 0x3FFF); // 14 bits
        let m_bits = u64::from(material_index & 0xFFFFF); // 20 bits

        if transparent {
            // 【透明物体】：
            // 排序优先级：Depth (远->近) > Pipeline > Material

            // 1. 反转深度，让远处的物体(大深度)变成小数值，从而排在前面
            let d_bits = raw_d_bits ^ 0x3FFF_FFFF;

            // 2. 重新排列位布局
            // Depth (30 bits) << 34 | Pipeline (14 bits) << 20 | Material (20 bits)
            // 总共 64 bits
            Self((d_bits << 34) | (p_bits << 20) | m_bits)
        } else {
            // 【不透明物体】：
            // 排序优先级：Pipeline > Material > Depth (近->远)

            // Depth 保持正序 (小深度排前面，Front-to-Back)
            let d_bits = raw_d_bits;

            // Pipeline (14 bits) << 50 | Material (20 bits) << 30 | Depth (30 bits)
            Self((p_bits << 50) | (m_bits << 30) | d_bits)
        }
    }
}

// ============================================================================
// RenderFrame
// ============================================================================

/// 渲染帧管理器
///
/// 采用 Render Graph 架构：
/// 1. Extract 阶段：从 Scene 提取渲染数据
/// 2. Prepare 阶段：准备 GPU 资源
/// 3. Execute 阶段：通过 `FrameComposer` 执行渲染 Pass
///
/// # 性能考虑
/// - `ExtractedScene` 持久化以复用内存
/// - `FrameComposer` 每帧创建，但开销极低（仅 Vec 指针操作）
///
/// # 注意
/// `RenderLists` 存储在 `RendererState` 中而非此处，
/// 以避免借用检查器的限制。
pub struct RenderFrame {
    pub(crate) render_state: RenderState,
    pub(crate) extracted_scene: ExtractedScene,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFrame {
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_state: RenderState::new(),
            extracted_scene: ExtractedScene::with_capacity(1024),
        }
    }

    /// 获取渲染状态引用
    #[inline]
    pub fn render_state(&self) -> &RenderState {
        &self.render_state
    }

    /// 获取提取的场景数据引用
    #[inline]
    pub fn extracted_scene(&self) -> &ExtractedScene {
        &self.extracted_scene
    }

    /// 阶段 1: 提取场景数据并准备全局资源
    ///
    /// 执行 Extract 和 Prepare 阶段，为后续的 Compose 和 Execute 做准备。
    ///
    /// # 阶段说明
    ///
    /// 1. **Extract**：从 Scene 提取渲染数据到 `ExtractedScene`
    /// 2. **Prepare**：准备全局 GPU 资源（相机 Uniform、光照数据等）
    ///
    /// # 注意
    ///
    /// 此方法不获取 Surface，Surface 获取延迟到 `FrameComposer::render()` 中，
    /// 以减少 `SwapChain` Buffer 的持有时间。
    pub fn extract_and_prepare(
        &mut self,
        resource_manager: &mut ResourceManager,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        time: f32,
    ) {
        resource_manager.next_frame();

        // 1. Extract: reuse memory, avoid per-frame allocation
        self.extracted_scene
            .extract_into(scene, camera, assets, resource_manager);

        // 2. Resolve GPU environment and BRDF LUT before prepare_global.
        //    This creates textures in the cache and determines env_map_max_mip_level.
        let env_max_mip = resource_manager.resolve_gpu_environment(assets, &scene.environment);
        resource_manager.ensure_brdf_lut();

        // Patch env_map_max_mip_level into the scene uniform buffer
        {
            let current = scene.uniforms_buffer.read().env_map_max_mip_level;
            if (current - env_max_mip).abs() > f32::EPSILON {
                scene.uniforms_buffer.write().env_map_max_mip_level = env_max_mip;
            }
        }

        // 3. Prepare: global GPU resources
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    /// 定期清理资源
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // 定期清理资源(Todo: LRU 策略)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}
