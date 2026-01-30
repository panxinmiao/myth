//! 渲染帧管理
//!
//! RenderFrame 负责：extract, prepare, 执行渲染图
//! 采用 Render Graph 架构，将渲染逻辑拆分到独立的 Pass 中


use crate::scene::{Scene};
use crate::scene::camera::{RenderCamera};
use crate::assets::AssetServer;

use crate::renderer::core::{WgpuContext, ResourceManager};
use crate::renderer::graph::{RenderState, RenderContext, RenderGraph, RenderNode};
use crate::renderer::graph::extracted::ExtractedScene;
use crate::renderer::graph::passes::{ForwardRenderPass, BRDFLutComputePass, IBLComputePass};
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

    /// 主渲染入口
    /// 
    /// 执行完整的渲染流程：
    /// 1. 获取 Surface 纹理
    /// 2. Extract 阶段
    /// 3. Prepare 阶段
    /// 4. 构建 RenderContext
    /// 5. 执行 RenderGraph
    /// 
    /// # 参数
    /// - `extra_nodes`: 额外的渲染节点（如 UI Pass），将在内置 Pass 之后执行
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        wgpu_ctx: &mut WgpuContext,
        resource_manager: &mut ResourceManager,
        pipeline_cache: &mut PipelineCache,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        time: f32,
        extra_nodes: &[&dyn RenderNode],
    ) {
        resource_manager.next_frame();

        // ========================================================================
        // 1. Extract 阶段：复用内存，避免每帧分配
        // ========================================================================
        self.extracted_scene.extract_into(scene, camera, assets, resource_manager);

        // ========================================================================
        // 2. Prepare 阶段：准备 GPU 资源
        // ========================================================================
        self.prepare_global_resources(resource_manager, assets, scene, camera, time);


        // ========================================================================
        // 3. 构建瞬态 RenderGraph 并执行
        // ========================================================================

        // let _ = &self.forward_pass.prepare(&mut PrepareContext {
        //     resource_manager,
        //     pipeline_cache,
        //     assets,
        //     extracted_scene: &self.extracted_scene,
        //     render_state: &self.render_state,
        //     wgpu_ctx,
        // });
        // ========================================================================
        //  Acquire Surface
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
            
            // 1. 内置 Pass
            // 例如: graph.add_node(&self.shadow_pass);
            //       graph.add_node(&self.ibl_pass);
            graph.add_node(&self.brdf_pass);
            graph.add_node(&self.ibl_pass);
            graph.add_node(&self.forward_pass);
            
            // 2. 外部注入的 Pass（如 UI、后处理等）
            for node in extra_nodes {
                graph.add_node(*node);
            }
            
            // 执行渲染图（内部创建 encoder 并提交）
            graph.execute(&mut ctx);
        }
        
        // 执行渲染图（内部创建 encoder 并提交）
        // graph.execute(render_ctx);

        // ========================================================================
        // 5. Present
        // ========================================================================
        output.present();


        // 定期清理资源(Todo: LRU 策略)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }

    fn prepare_global_resources(
        &mut self,
        resource_manager: &mut ResourceManager,
        assets: &AssetServer,
        scene: &mut Scene,
        camera: &RenderCamera,
        time: f32,
    ) {
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    // fn upload_skeletons_extracted(
    //     &self,
    //     resource_manager: &mut ResourceManager,
    //     skins: &SlotMap<SkeletonKey, Skeleton>,
    //     extracted: &ExtractedScene,
    // ) {
    //     let mut processed_skeletons = rustc_hash::FxHashSet::default();

    //     for skel in &extracted.skeletons {
    //         if processed_skeletons.contains(&skel.skeleton_key) {
    //             continue;
    //         }

    //         let skeleton = match skins.get(skel.skeleton_key) {
    //             Some(s) => s,
    //             None => {
    //                 warn!("Skeleton {:?} missing during upload", skel.skeleton_key);
    //                 continue;
    //             }
    //         };
    //         resource_manager.prepare_skeleton(skel.skeleton_key, skeleton);
    //         processed_skeletons.insert(skel.skeleton_key);
    //     }
    // }

    // fn update_meshes_extracted(
    //     &self,
    //     resource_manager: &mut ResourceManager,
    //     extracted: &ExtractedScene,
    // ) {
    //     for item in &extracted.render_items {
    //         resource_manager.update_mesh_instance(item.node_key, &item.transform);
    //     }
    // }
}
