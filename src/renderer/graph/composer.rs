//! Frame Composer
//!
//! `FrameComposer` provides a fluent API to build and execute the render pipeline.
//! It acts as the "glue" connecting the Prepare phase and the Execute phase.
//!
//! # Three-Phase Rendering Architecture
//!
//! 1. **Prepare**: Extract data and prepare GPU resources
//! 2. **Compose**: Add `RenderNode`s via the fluent API
//! 3. **Execute**: Acquire the Surface, build the `RenderGraph`, and submit GPU commands
//!
//! # Example
//!
//! ```ignore
//! // Fluent chained invocation
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .add_node(RenderStage::PostProcess, &bloom_pass)
//!     .render();
//! ```

use super::frame::RenderLists;
use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::builder::FrameBuilder;
use crate::renderer::graph::context::{ExecuteContext, FrameResources, PrepareContext};
use crate::renderer::graph::frame::FrameBlackboard;
use crate::renderer::graph::node::RenderNode;
use crate::renderer::graph::stage::RenderStage;
use crate::renderer::graph::transient_pool::TransientTexturePool;
use crate::renderer::pipeline::PipelineCache;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,

    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,

    pub frame_resources: &'a FrameResources,
    pub transient_pool: &'a mut TransientTexturePool,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// Render lists (populated by `SceneCullPass`)
    pub render_lists: &'a mut RenderLists,

    /// Frame blackboard (cross-pass transient data communication)
    pub blackboard: &'a mut FrameBlackboard,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: &'a RenderCamera,
    pub assets: &'a AssetServer,
    pub time: f32,
}

/// Frame Composer
///
/// Holds all context references needed to render a single frame and provides
/// a fluent API for adding render nodes.
///
/// # Design Notes
///
/// - **Clear responsibilities**: `FrameComposer` only handles context and flow control
/// - **Lifetime safety**: Lifetime `'a` locks the mutable borrow on `Renderer`
/// - **Deferred Surface acquisition**: The Surface is acquired only in `.render()` to minimize hold time
///
/// # Performance Considerations
///
/// - Internal `FrameBuilder` pre-allocates capacity for 16 nodes
/// - Sorting uses `FrameBuilder`'s efficient sorting mechanism
/// - All fields are references — no heap allocation overhead
pub struct FrameComposer<'a> {
    // GPU context
    ctx: ComposerContext<'a>,

    // Builder (collects render nodes)
    builder: FrameBuilder<'a>,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    ///
    /// Built-in passes (BRDF LUT, IBL, Forward) are injected automatically.
    pub(crate) fn new(builder: FrameBuilder<'a>, ctx: ComposerContext<'a>) -> Self {
        Self { ctx, builder }
    }

    /// Adds a custom render node at the specified stage.
    ///
    /// Supports method chaining.
    ///
    /// # Arguments
    ///
    /// - `stage`: Render stage (determines execution order)
    /// - `node`: Render node reference
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_node(RenderStage::UI, &ui_pass)
    ///     .add_node(RenderStage::PostProcess, &bloom_pass)
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_node(mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> Self {
        self.builder.add_node(stage, node);
        self
    }

    /// Adds multiple nodes to the same stage in batch.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_nodes(RenderStage::PostProcess, &[&mut bloom, &mut fxaa, &mut tone_mapping])
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_nodes<I>(mut self, stage: RenderStage, nodes: I) -> Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
        self.builder.add_nodes(stage, nodes);
        self
    }

    /// Executes the render pipeline and presents to the screen.
    ///
    /// This is the final step of the rendering workflow:
    /// 1. Acquire the Surface texture
    /// 2. Build the `RenderContext`
    /// 3. Convert the Builder into a sorted `RenderGraph`
    /// 4. Execute the render graph
    /// 5. Present
    ///
    /// # Note
    ///
    /// This method consumes `self`; the `FrameComposer` cannot be reused after calling it.
    pub fn render(self) {
        // 1. Acquire the Surface Texture (deferred to the last moment to minimize hold time)
        let output = match self.ctx.wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Render error: {e:?}");
                return;
            }
        };

        let view_format = self.ctx.wgpu_ctx.surface_view_format;

        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..Default::default()
        });

        // 2. Convert the Builder into a sorted RenderGraph
        let mut graph = self.builder.build();

        // 3. Prepare phase — mutable context
        {
            let mut prepare_ctx = PrepareContext {
                wgpu_ctx: &*self.ctx.wgpu_ctx,
                resource_manager: self.ctx.resource_manager,
                pipeline_cache: self.ctx.pipeline_cache,
                assets: self.ctx.assets,
                scene: self.ctx.scene,
                camera: self.ctx.camera,
                render_state: self.ctx.render_state,
                extracted_scene: self.ctx.extracted_scene,
                render_lists: self.ctx.render_lists,
                blackboard: self.ctx.blackboard,
                frame_resources: self.ctx.frame_resources,
                transient_pool: self.ctx.transient_pool,
                time: self.ctx.time,
                global_bind_group_cache: self.ctx.global_bind_group_cache,
                color_view_flip_flop: 0,
            };
            graph.prepare(&mut prepare_ctx);
        }

        // 4. Execute phase — read-only context
        let execute_ctx = ExecuteContext::new(
            &*self.ctx.wgpu_ctx,
            &*self.ctx.resource_manager,
            &surface_view,
            &*self.ctx.render_lists,
            &*self.ctx.blackboard,
            self.ctx.frame_resources,
            &*self.ctx.transient_pool,
        );
        graph.execute(&execute_ctx);

        // 5. Present
        output.present();

        // 6. Return transient textures to the pool for next frame reuse
        self.ctx.transient_pool.reset();
    }
}
