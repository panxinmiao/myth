use crate::renderer::graph::rdg::context::RdgPrepareContext;

use super::builder::PassBuilder;
use super::context::RdgExecuteContext;
use super::types::TextureNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// An ephemeral per-frame render or compute pass in the declarative render graph.
///
/// PassNodes are **data-only packets** created by their corresponding
/// `Feature` each frame via `Feature::add_to_graph()`.  They carry only
/// lightweight IDs and transient bind-group slots — all persistent GPU
/// resources (layouts, pipelines, buffers) live in the Feature.
///
/// # Lifecycle
///
/// 1. **`setup`** — declare resource read/write topology for the graph.
/// 2. **`prepare`** — called *after* graph compilation and transient memory
///    allocation.  Only assemble `BindGroup`s that reference RDG-managed
///    transient textures.  Context is intentionally minimal.
/// 3. **`execute`** — record GPU commands into the shared encoder.
pub trait PassNode: 'static {
    fn name(&self) -> &'static str;

    /// Declare resource dependencies for the graph topology.
    fn setup(&mut self, builder: &mut PassBuilder);

    /// Assemble transient `BindGroup`s after physical resource allocation.
    ///
    /// Only `BindGroup`s that reference RDG-managed transient textures
    /// should be created here.  The context deliberately excludes heavy
    /// infrastructure (shader compiler, asset server, etc.).
    #[allow(unused_variables)]
    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {}

    /// Record GPU commands into the shared encoder.
    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut CommandEncoder);
}

/// Per-pass metadata stored in the [`RenderGraph`].
///
/// Owns the ephemeral [`PassNode`] (via `Box`) for the duration of
/// the current frame.  The graph drops all records in `begin_frame()`.
pub struct PassRecord {
    pub name: &'static str,
    /// Owned ephemeral pass node — `None` only briefly during `add_pass()`
    /// between the placeholder push and the node insertion.
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
    /// Used by `RenderGraph::add_pass` during the two-phase insertion:
    /// the record is pushed first, then setup is called, then the node is stored.
    pub fn new_empty(name: &'static str) -> Self {
        Self {
            name,
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
    /// Panics if the node has not been inserted yet (should never happen
    /// outside of the `add_pass` two-phase window).
    #[inline]
    pub fn get_pass_mut(&mut self) -> &mut dyn PassNode {
        self.node.as_mut().expect("PassRecord node not set").as_mut()
    }
}
