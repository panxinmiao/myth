//! Frame Builder
//!
//! `FrameBuilder` provides a fluent API for building each frame's render pipeline.
//! Allows users to insert, replace, or remove render nodes at specified stages.

use super::graph::RenderGraph;
use super::node::RenderNode;
use super::stage::RenderStage;

/// Render node entry.
///
/// Stores a node reference together with its owning stage, used for sorting and execution.
struct NodeEntry<'a> {
    /// Render stage
    stage: RenderStage,
    /// Insertion order within the stage (for stable sorting)
    order: u16,
    /// Node reference
    node: &'a mut dyn RenderNode,
}

/// Frame Builder
///
/// Provides a builder pattern to organize each frame's render pipeline.
///
/// # Design Principles
///
/// - **Staged rendering**: Rendering order defined via `RenderStage`
/// - **Flexible insertion**: Custom nodes can be inserted at any stage
/// - **Zero-overhead abstraction**: Stage ordering determined at compile time, no runtime lookup cost
/// - **Non-owning**: Only stores node references; the caller manages node lifetimes
///
/// # Usage
///
/// ```ignore
/// let mut builder = FrameBuilder::new();
///
/// // Add built-in passes
/// builder.add_node(RenderStage::PreProcess, &brdf_pass);
/// builder.add_node(RenderStage::Opaque, &forward_pass);
///
/// // Add custom passes
/// builder.add_node(RenderStage::UI, &ui_pass);
///
/// // Execute rendering
/// builder.execute(&mut render_context);
/// ```
///
/// # Performance Considerations
///
/// - Internal smallvec pre-allocates 16 entries, covering most scenarios
/// - Sorting uses the standard library's `sort_unstable_by_key` — efficient with no extra memory overhead
/// - Nodes are stored as references — no heap allocation overhead
pub struct FrameBuilder<'a> {
    /// Node list (unsorted)
    nodes: smallvec::SmallVec<[NodeEntry<'a>; 16]>,
    /// Next insertion order number
    next_order: u16,
}

impl Default for FrameBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> FrameBuilder<'a> {
    /// Creates a new frame builder.
    ///
    /// Pre-allocates space for 16 nodes, covering a typical render pipeline.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: smallvec::SmallVec::with_capacity(16),
            next_order: 0,
        }
    }

    /// Creates a frame builder with the specified capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: smallvec::SmallVec::with_capacity(capacity),
            next_order: 0,
        }
    }

    /// Adds a render node at the specified stage.
    ///
    /// Nodes within the same stage execute in insertion order.
    ///
    /// # Arguments
    ///
    /// - `stage`: Render stage
    /// - `node`: Render node reference
    ///
    /// # Returns
    ///
    /// Returns `&mut Self` for method chaining.
    #[inline]
    pub fn add_node(&mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> &mut Self {
        self.nodes.push(NodeEntry {
            stage,
            order: self.next_order,
            node,
        });
        self.next_order = self.next_order.wrapping_add(1);
        self
    }

    /// Adds multiple nodes to the same stage in batch.
    ///
    /// Suitable for adding several post-processing effects or multiple UI layers.
    #[inline]
    pub fn add_nodes<I>(&mut self, stage: RenderStage, nodes: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
        for node in nodes {
            // `node` is already `&'a mut dyn RenderNode`, move it directly
            self.add_node(stage, node);
        }
        self
    }

    /// Returns the current number of nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Checks whether the specified stage has any nodes.
    #[inline]
    #[must_use]
    pub fn has_stage(&self, stage: RenderStage) -> bool {
        self.nodes.iter().any(|e| e.stage == stage)
    }

    /// Removes all nodes (retains allocated capacity).
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_order = 0;
    }

    /// Builds a `RenderGraph` without executing it (useful for debugging or deferred execution).
    ///
    /// # Note
    ///
    /// The returned `RenderGraph` shares the same lifetime as the `FrameBuilder`.
    #[must_use]
    pub fn build(mut self) -> RenderGraph<'a> {
        self.nodes
            .sort_unstable_by_key(|e| (e.stage.order(), e.order));

        let mut graph = RenderGraph::with_capacity(self.nodes.len());

        for entry in self.nodes {
            // Now we can move entry.node (i.e. &'a mut dyn RenderNode)
            // into the graph
            graph.add_node(entry.node);
        }

        graph
    }
}
