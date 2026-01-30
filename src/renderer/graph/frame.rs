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

use crate::assets::AssetServer;
use crate::renderer::core::ResourceManager;
use crate::scene::camera::RenderCamera;
use crate::scene::Scene;

use super::extracted::ExtractedScene;
use super::passes::{BRDFLutComputePass, ForwardRenderPass, IBLComputePass};
use super::render_state::RenderState;
use super::stage::RenderStage;
use super::builder::FrameBuilder;

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    pub fn new(pipeline_id: u16, material_index: u32, depth: f32, transparent: bool) -> Self {
        // 1. 统一处理深度压缩 (30 bits)
        // 注意：这里假设 depth >= 0.0。如果 depth 可能为负，clamp 到 0 是安全的。
        let d_u32 = if depth.is_sign_negative() { 0 } else { depth.to_bits() >> 2 };
        let raw_d_bits = (d_u32 as u64) & 0x3FFF_FFFF;

        // 2. 准备其他数据
        let p_bits = (pipeline_id & 0x3FFF) as u64; // 14 bits
        let m_bits = (material_index & 0xFFFFF) as u64; // 20 bits

        if transparent {
            // 【透明物体】：
            // 排序优先级：Depth (远->近) > Pipeline > Material
            
            // 1. 反转深度，让远处的物体(大深度)变成小数值，从而排在前面
            let d_bits = raw_d_bits ^ 0x3FFF_FFFF;
            
            // 2. 重新排列位布局
            // Depth (30 bits) << 34 | Pipeline (14 bits) << 20 | Material (20 bits)
            // 总共 64 bits
            Self(
                (d_bits << 34) | 
                (p_bits << 20) | 
                m_bits
            )
        } else {
            // 【不透明物体】：
            // 排序优先级：Pipeline > Material > Depth (近->远)
            
            // Depth 保持正序 (小深度排前面，Front-to-Back)
            let d_bits = raw_d_bits; 
            
            // Pipeline (14 bits) << 50 | Material (20 bits) << 30 | Depth (30 bits)
            Self(
                (p_bits << 50) | 
                (m_bits << 30) | 
                d_bits
            )
        }
    }
}

/// 渲染帧管理器
///
/// 采用 Render Graph 架构：
/// 1. Extract 阶段：从 Scene 提取渲染数据
/// 2. Prepare 阶段：准备 GPU 资源
/// 3. Execute 阶段：通过 `FrameComposer` 执行渲染 Pass
///
/// # 性能考虑
/// - `ExtractedScene` 和内置 Pass 持久化以复用内存
/// - 命令列表在 Pass 内部管理，避免跨帧分配
/// - `FrameComposer` 每帧创建，但开销极低（仅 Vec 指针操作）
pub struct RenderFrame {
    render_state: RenderState,
    extracted_scene: ExtractedScene,
    forward_pass: ForwardRenderPass,
    brdf_pass: BRDFLutComputePass,
    ibl_pass: IBLComputePass,
}

impl RenderFrame {
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            render_state: RenderState::new(),
            extracted_scene: ExtractedScene::with_capacity(1024),
            forward_pass: ForwardRenderPass::new(wgpu::Color::BLACK),
            brdf_pass: BRDFLutComputePass::new(&device),
            ibl_pass: IBLComputePass::new(&device),
        }
    }

    /// 获取内置的 BRDF LUT 计算 Pass
    #[inline]
    pub fn brdf_pass(&self) -> &BRDFLutComputePass {
        &self.brdf_pass
    }

    /// 获取内置的 IBL 计算 Pass
    #[inline]
    pub fn ibl_pass(&self) -> &IBLComputePass {
        &self.ibl_pass
    }

    /// 获取内置的 Forward 渲染 Pass
    #[inline]
    pub fn forward_pass(&self) -> &ForwardRenderPass {
        &self.forward_pass
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
    /// 以减少 SwapChain Buffer 的持有时间。
    #[allow(clippy::too_many_arguments)]
    pub fn extract_and_prepare(
        &mut self,
        // _wgpu_ctx: &mut crate::renderer::core::WgpuContext,
        resource_manager: &mut ResourceManager,
        // _pipeline_cache: &mut crate::renderer::pipeline::PipelineCache,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        time: f32,
    ) {
        resource_manager.next_frame();

        // 1. Extract：复用内存，避免每帧分配
        self.extracted_scene
            .extract_into(scene, camera, assets, resource_manager);

        // 2. Prepare：准备全局 GPU 资源
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    /// 将内置 Pass 注入到 `FrameBuilder` 中
    ///
    /// 这是一个辅助方法，用于在 `FrameComposer` 创建时注入内置 Pass。
    /// 返回修改后的 `FrameBuilder` 以支持链式调用。
    #[inline]
    pub fn inject_builtin_passes<'a>(&'a self, mut builder: FrameBuilder<'a>) -> FrameBuilder<'a> {
        builder
            .add_node(RenderStage::PreProcess, &self.brdf_pass)
            .add_node(RenderStage::PreProcess, &self.ibl_pass)
            .add_node(RenderStage::Opaque, &self.forward_pass);
        builder
    }

    /// 定期清理资源
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // 定期清理资源(Todo: LRU 策略)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}
