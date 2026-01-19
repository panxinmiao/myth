//! 渲染帧管理
//!
//! RenderFrame 负责：extract, prepare, 执行渲染图
//! 采用 Render Graph 架构，将渲染逻辑拆分到独立的 Pass 中

use slotmap::SlotMap;
use log::warn;

use crate::scene::skeleton::Skeleton;
use crate::scene::{Scene, SkeletonKey};
use crate::scene::camera::Camera;
use crate::assets::AssetServer;

use crate::renderer::core::{WgpuContext, ResourceManager};
use crate::renderer::graph::{RenderState, RenderContext, RenderGraph};
use crate::renderer::graph::extracted::ExtractedScene;
use crate::renderer::graph::passes::ForwardRenderPass;
use crate::renderer::pipeline::PipelineCache;

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    pub fn new(pipeline_id: u16, material_index: u32, depth: f32) -> Self {
        let p_bits = ((pipeline_id & 0x3FFF) as u64) << 50;
        let m_bits = ((material_index & 0xFFFFF) as u64) << 30;
        let d_u32 = if depth.is_sign_negative() { 0 } else { depth.to_bits() >> 2 };
        let d_bits = (d_u32 as u64) & 0x3FFF_FFFF;
        Self(p_bits | m_bits | d_bits)
    }
}

/// 准备好提交给 GPU 的指令
pub struct RenderCommand {
    pub object_data: crate::renderer::core::ObjectBindingData,
    pub geometry_handle: crate::assets::GeometryHandle,
    pub material_handle: crate::assets::MaterialHandle,
    pub render_state_id: u32,
    pub pipeline_id: u16,
    pub pipeline: wgpu::RenderPipeline,
    pub model_matrix: glam::Mat4,
    pub sort_key: RenderKey,
    pub dynamic_offset: u32,
}

/// 渲染帧管理器
/// 
/// 采用 Render Graph 架构：
/// 1. Extract 阶段：从 Scene 提取渲染数据
/// 2. Prepare 阶段：准备 GPU 资源
/// 3. Execute 阶段：通过 RenderGraph 执行渲染 Pass
/// 
/// # 性能考虑
/// - `ExtractedScene` 和 `ForwardRenderPass` 持久化以复用内存
/// - 命令列表在 Pass 内部管理，避免跨帧分配
/// - 后续可缓存整个 RenderGraph 配置
pub struct RenderFrame {
    render_state: RenderState,
    extracted_scene: ExtractedScene,
    forward_pass: ForwardRenderPass,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFrame {
    pub fn new() -> Self {
        Self {
            render_state: RenderState::new(),
            extracted_scene: ExtractedScene::with_capacity(1024, 16),
            forward_pass: ForwardRenderPass::new(wgpu::Color::BLACK),
        }
    }

    /// 主渲染入口
    /// 
    /// 执行完整的渲染流程：
    /// 1. 获取 Surface 纹理
    /// 2. Extract 阶段
    /// 3. Prepare 阶段
    /// 4. 构建 RenderContext
    /// 5. 执行 RenderGraph
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        wgpu_ctx: &mut WgpuContext,
        resource_manager: &mut ResourceManager,
        pipeline_cache: &mut PipelineCache,
        scene: &mut Scene,
        camera: &Camera,
        assets: &AssetServer,
        time: f32,
    ) {
        resource_manager.next_frame();

        // ========================================================================
        // 1. Acquire Surface
        // ========================================================================
        let output = match wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                eprintln!("Render error: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());


        scene.sync_morph_weights();

        // ========================================================================
        // 2. Extract 阶段：复用内存，避免每帧分配
        // ========================================================================
        self.extracted_scene.extract_into(scene, camera, assets);

        // 更新清屏颜色
        if let Some(bg_color) = self.extracted_scene.background {
            self.forward_pass.clear_color = wgpu::Color {
                r: bg_color.x as f64,
                g: bg_color.y as f64,
                b: bg_color.z as f64,
                a: bg_color.w as f64,
            };
        }

        // ========================================================================
        // 3. Prepare 阶段：准备 GPU 资源
        // ========================================================================
        self.prepare_global_resources(resource_manager, assets, scene, camera, time);
        self.upload_skeletons_extracted(resource_manager, &scene.skins, &self.extracted_scene);

        // ========================================================================
        // 4. 构建瞬态 RenderGraph 并执行
        // ========================================================================
        {
            let mut ctx = RenderContext {
                wgpu_ctx,
                resource_manager,
                pipeline_cache,
                assets,
                scene,
                camera,
                surface_view: &view,
                render_state: &self.render_state,
                extracted_scene: &self.extracted_scene,
                time,
            };

            // 构建瞬态 Render Graph（每帧重建，开销极低）
            let mut graph = RenderGraph::new();
            
            // Pass 编排：未来可在此添加更多 Pass
            // 例如: graph.add_node(&self.shadow_pass);
            //       graph.add_node(&self.ibl_pass);
            graph.add_node(&self.forward_pass);
            
            // 执行渲染图
            graph.execute(&mut ctx);
        }

        // ========================================================================
        // 5. Present 并清理
        // ========================================================================
        output.present();

        if resource_manager.frame_index().is_multiple_of(60) {
            resource_manager.prune(6000);
        }
    }

    fn prepare_global_resources(
        &mut self,
        resource_manager: &mut ResourceManager,
        assets: &AssetServer,
        scene: &Scene,
        camera: &Camera,
        time: f32,
    ) {
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    fn upload_skeletons_extracted(
        &self,
        resource_manager: &mut ResourceManager,
        skins: &SlotMap<SkeletonKey, Skeleton>,
        extracted: &ExtractedScene,
    ) {
        let mut processed_skeletons = rustc_hash::FxHashSet::default();

        for skel in &extracted.skeletons {
            if processed_skeletons.contains(&skel.skeleton_key) {
                continue;
            }

            let skeleton = match skins.get(skel.skeleton_key) {
                Some(s) => s,
                None => {
                    warn!("Skeleton {:?} missing during upload", skel.skeleton_key);
                    continue;
                }
            };
            resource_manager.prepare_skeleton(skel.skeleton_key, skeleton);
            processed_skeletons.insert(skel.skeleton_key);
        }
    }
}
