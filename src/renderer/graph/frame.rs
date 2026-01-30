//! 渲染帧管理
//!
//! `RenderFrame` 负责：extract, prepare, 构建渲染上下文
//! 采用 Render Graph 架构，将渲染逻辑拆分到独立的 Pass 中。
//!
//! # 新架构：FrameBuilder
//!
//! 引入 `FrameBuilder` 模式，允许用户在指定阶段插入自定义渲染节点：
//!
//! ```ignore
//! // 旧 API（已废弃）
//! renderer.render(scene, camera, assets, time, &[&ui_pass]);
//!
//! // 新 API
//! renderer.begin_frame(scene, camera, assets, time)
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .render();
//! ```

use crate::scene::Scene;
use crate::scene::camera::RenderCamera;
use crate::assets::AssetServer;

use crate::renderer::core::{WgpuContext, ResourceManager};
use super::node::RenderNode;
use super::render_state::RenderState;
use super::context::RenderContext;
use crate::renderer::graph::extracted::ExtractedScene;
use crate::renderer::graph::passes::{ForwardRenderPass, BRDFLutComputePass, IBLComputePass};
use crate::renderer::graph::stage::RenderStage;
use crate::renderer::graph::builder::FrameBuilder;
use crate::renderer::pipeline::PipelineCache;

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
/// 3. Execute 阶段：通过 `FrameBuilder` 执行渲染 Pass
/// 
/// # 性能考虑
/// - `ExtractedScene` 和内置 Pass 持久化以复用内存
/// - 命令列表在 Pass 内部管理，避免跨帧分配
/// - `FrameBuilder` 每帧创建，但开销极低（仅 Vec 指针操作）
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

    /// 准备帧渲染
    /// 
    /// 执行 Extract 和 Prepare 阶段，返回可用于构建渲染管线的 `PreparedFrame`。
    /// 
    /// # 阶段说明
    /// 
    /// 1. **Extract**：从 Scene 提取渲染数据到 `ExtractedScene`
    /// 2. **Prepare**：准备全局 GPU 资源（相机 Uniform、光照数据等）
    /// 3. **Acquire Surface**：获取交换链纹理
    /// 
    /// # 返回
    /// 
    /// 返回 `Some(PreparedFrame)` 如果成功获取 Surface 纹理，否则返回 `None`。
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<'a>(
        &'a mut self,
        wgpu_ctx: &'a mut WgpuContext,
        resource_manager: &'a mut ResourceManager,
        pipeline_cache: &'a mut PipelineCache,
        scene: &'a mut Scene,
        camera: &'a RenderCamera,
        assets: &'a AssetServer,
        time: f32,
    ) -> Option<PreparedFrame<'a>> {
        resource_manager.next_frame();

        // ========================================================================
        // 1. Extract 阶段：复用内存，避免每帧分配
        // ========================================================================
        self.extracted_scene.extract_into(scene, camera, assets, resource_manager);

        // ========================================================================
        // 2. Prepare 阶段：准备 GPU 资源
        // ========================================================================
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);

        // ========================================================================
        // 3. Acquire Surface
        // ========================================================================
        let output = match wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return None,
            Err(e) => {
                eprintln!("Render error: {:?}", e);
                return None;
            }
        };

        Some(PreparedFrame {
            render_frame: self,
            wgpu_ctx,
            resource_manager,
            pipeline_cache,
            scene,
            camera,
            assets,
            time,
            surface_texture: output,
        })
    }

    /// 定期清理资源
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // 定期清理资源(Todo: LRU 策略)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}

/// 准备完成的帧
/// 
/// 持有一帧渲染所需的所有资源引用。
/// 通过 `render_default()` 或 `render_with_nodes()` 方法执行渲染。
/// 
/// # 生命周期
/// 
/// `PreparedFrame` 的生命周期较短，仅在单帧渲染期间存在。
/// 它持有 Surface 纹理的所有权，在 `render()` 后自动 present。
/// 
/// # 设计说明
/// 
/// 采用"分离构建"模式：
/// - `PreparedFrame` 持有渲染资源和内置 Pass
/// - 用户可以通过 `render_with_nodes()` 添加自定义节点
/// - 支持按阶段精确插入节点
pub struct PreparedFrame<'a> {
    render_frame: &'a RenderFrame,
    wgpu_ctx: &'a mut WgpuContext,
    resource_manager: &'a mut ResourceManager,
    pipeline_cache: &'a mut PipelineCache,
    scene: &'a mut Scene,
    camera: &'a RenderCamera,
    assets: &'a AssetServer,
    time: f32,
    surface_texture: wgpu::SurfaceTexture,
}

impl<'a> PreparedFrame<'a> {
    /// 获取内置的 BRDF LUT 计算 Pass
    #[inline]
    pub fn brdf_pass(&self) -> &BRDFLutComputePass {
        self.render_frame.brdf_pass()
    }
    
    /// 获取内置的 IBL 计算 Pass
    #[inline]
    pub fn ibl_pass(&self) -> &IBLComputePass {
        self.render_frame.ibl_pass()
    }
    
    /// 获取内置的 Forward 渲染 Pass
    #[inline]
    pub fn forward_pass(&self) -> &ForwardRenderPass {
        self.render_frame.forward_pass()
    }
    
    /// 使用默认内置 Pass 渲染
    /// 
    /// 包含：BRDF LUT → IBL → Forward
    /// 
    /// # 示例
    /// 
    /// ```ignore
    /// if let Some(frame) = renderer.begin_frame(scene, camera, assets, time) {
    ///     frame.render_default();
    /// }
    /// ```
    #[inline]
    pub fn render_default(self) {
        self.render_with_nodes(&[]);
    }
    
    /// 使用自定义节点渲染
    /// 
    /// 在内置 Pass 基础上添加自定义渲染节点。
    /// 节点按阶段排序执行。
    /// 
    /// # 参数
    /// 
    /// - `extra_nodes`: 额外的渲染节点，元组格式为 `(RenderStage, &dyn RenderNode)`
    /// 
    /// # 示例
    /// 
    /// ```ignore
    /// frame.render_with_nodes(&[
    ///     (RenderStage::UI, &ui_pass),
    ///     (RenderStage::PostProcess, &bloom_pass),
    /// ]);
    /// ```
    pub fn render_with_nodes(self, extra_nodes: &[(RenderStage, &dyn RenderNode)]) {
        let mut builder = FrameBuilder::new();
        
        // 添加内置 Pass
        builder
            .add_node(RenderStage::PreProcess, self.render_frame.brdf_pass())
            .add_node(RenderStage::PreProcess, self.render_frame.ibl_pass())
            .add_node(RenderStage::Opaque, self.render_frame.forward_pass());
        
        // 添加用户自定义节点
        for (stage, node) in extra_nodes {
            builder.add_node(*stage, *node);
        }
        
        self.render(builder);
    }
    
    /// 执行渲染并呈现
    /// 
    /// 接受配置好的 `FrameBuilder`，执行渲染管线并呈现到屏幕。
    /// 
    /// # 参数
    /// 
    /// - `builder`: 配置好的帧构建器
    fn render(self, builder: FrameBuilder<'_>) {
        let view = self.surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut ctx = RenderContext {
            wgpu_ctx: self.wgpu_ctx,
            resource_manager: self.resource_manager,
            pipeline_cache: self.pipeline_cache,
            assets: self.assets,
            scene: self.scene,
            camera: self.camera,
            surface_view: &view,
            render_state: &self.render_frame.render_state,
            extracted_scene: &self.render_frame.extracted_scene,
            time: self.time,
        };
        
        // 执行渲染管线
        builder.execute(&mut ctx);
        
        // Present
        self.surface_texture.present();
    }
}
