//! Render Graph Executor
//!
//! `RenderGraph` manages the execution order of render nodes.
//! Uses a transient-graph design: a new graph instance is created each frame, storing only node references.

use smallvec::SmallVec;

use super::context::{ExecuteContext, PrepareContext};
use super::node::RenderNode;

/// Render graph (transient reference container).
///
/// Manages and executes a list of render nodes. Uses a transient design —
/// a new instance is created each frame.
///
/// # Design Notes
/// - Lifetime parameter `'a` stores node references without ownership transfer
/// - Creating a new Graph each frame has minimal cost (only Vec pointer pushes)
/// - Nodes themselves are persistently stored in `RenderFrame`, reusing memory
///
/// # Performance Considerations
/// - The transient graph avoids complex cache invalidation logic
/// - The per-frame rebuild cost is roughly a few pointer pushes — negligible
/// - May be extended to a DAG structure in the future to support parallel encoding
pub struct RenderGraph<'a> {
    nodes: SmallVec<[&'a mut dyn RenderNode; 8]>,
}

impl Default for RenderGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RenderGraph<'a> {
    /// Creates an empty render graph.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: SmallVec::new(),
        }
    }

    /// Pre-allocates node capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: SmallVec::with_capacity(capacity),
        }
    }

    /// Adds a render node reference.
    ///
    /// Nodes execute in the order they are added.
    #[inline]
    pub fn add_node(&mut self, node: &'a mut dyn RenderNode) {
        self.nodes.push(node);
    }

    pub fn prepare(&mut self, ctx: &mut PrepareContext) {
        for node in &mut self.nodes {
            node.prepare(ctx);
        }
    }

    /// Executes the render graph.
    ///
    /// Creates a CommandEncoder, executes all nodes in order, and submits the commands.
    ///
    /// # Performance Notes
    /// - All nodes share a single CommandEncoder, reducing submission count
    /// - Debug groups are used for GPU profiling
    pub fn execute(&self, ctx: &ExecuteContext) {
        let mut encoder =
            ctx.wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Graph Encoder"),
                });

        for node in &self.nodes {
            #[cfg(debug_assertions)]
            encoder.push_debug_group(node.name());
            node.run(ctx, &mut encoder);
            #[cfg(debug_assertions)]
            encoder.pop_debug_group();
        }

        ctx.wgpu_ctx.queue.submit(Some(encoder.finish()));
    }

    /// Returns the number of nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
