use crate::renderer::graph::core::context::{ExecuteContext, PrepareContext};

use super::types::TextureNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// Pure GPU command recorder for a single render or compute pass.
///
/// `PassNode` is intentionally minimal — it carries only lightweight IDs
/// and transient bind-group slots.  All persistent GPU resources (layouts,
/// pipelines, buffers) live in the owning `Feature`.
///
/// # Lifecycle (Eager Setup)
///
/// Resource topology and naming are declared **outside** the `PassNode`,
/// inside the closure passed to [`RenderGraph::add_pass`].  The node
/// itself only participates in two runtime phases:
///
/// 1. **`prepare`** — called after graph compilation and transient memory
///    allocation.  Assemble `BindGroup`s that reference RDG-managed
///    transient textures.
/// 2. **`execute`** — record GPU commands into the shared encoder.
pub trait PassNode: Send + Sync + 'static {
    /// Assemble transient `BindGroup`s after physical resource allocation.
    ///
    /// Only `BindGroup`s that reference RDG-managed transient textures
    /// should be created here.  The context deliberately excludes heavy
    /// infrastructure (shader compiler, asset server, etc.).
    #[allow(unused_variables)]
    fn prepare(&mut self, ctx: &mut PrepareContext) {}

    /// Record GPU commands into the shared encoder.
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder);
}

/// Per-pass metadata stored in the [`RenderGraph`].
///
/// Owns the ephemeral [`PassNode`] (via `Box`) for the duration of
/// the current frame.  The graph drops all records in `begin_frame()`.
pub struct PassRecord {
    /// Human-readable name for debug labels, topology dumps, and GPU
    /// debug groups.  Set once by [`RenderGraph::add_pass`].
    pub name: &'static str,

    /// Logical group this pass belongs to (e.g. `"Bloom_System"`).
    ///
    /// Populated by [`RenderGraph::with_group`] when the `rdg_inspector`
    /// feature is enabled; otherwise always `None`.  Used exclusively
    /// by [`RenderGraph::dump_mermaid`] to emit Mermaid `subgraph` blocks.
    #[cfg(feature = "rdg_inspector")]
    pub groups: smallvec::SmallVec<[&'static str; 4]>,

    /// Owned ephemeral pass node.
    pub node: Option<Box<dyn PassNode>>,

    pub reads: SmallVec<[TextureNodeId; 8]>,
    pub writes: SmallVec<[TextureNodeId; 4]>,
    pub creates: SmallVec<[TextureNodeId; 4]>,

    // Compile-time state
    pub physical_dependencies: SmallVec<[usize; 8]>,
    pub has_side_effect: bool,
    pub reference_count: u32,
}

impl PassRecord {
    /// Creates a placeholder record without a node.
    ///
    /// Used internally by [`RenderGraph::add_pass`] during the two-phase
    /// insertion: the record is pushed first so the [`PassBuilder`] can
    /// reference it, then the node is stored after the closure returns.
    #[must_use]
    pub fn new_empty(name: &'static str) -> Self {
        Self {
            name,
            #[cfg(feature = "rdg_inspector")]
            groups: SmallVec::new(),
            node: None,
            reads: SmallVec::new(),
            writes: SmallVec::new(),
            creates: SmallVec::new(),
            physical_dependencies: SmallVec::new(),
            has_side_effect: false,
            reference_count: 0,
        }
    }

    /// Returns a mutable reference to the owned pass node.
    ///
    /// # Panics
    /// Panics if the node has not been inserted yet.
    #[inline]
    pub fn get_pass_mut(&mut self) -> &mut dyn PassNode {
        self.node
            .as_mut()
            .expect("PassRecord node not set")
            .as_mut()
    }
}
