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
///
/// # Lifetime Model
///
/// The `'static` bound has been intentionally **removed**.  Concrete
/// `PassNode` implementations may carry frame-scoped borrowed references
/// (e.g. `&'a [Vertex]`, `&'a wgpu::BindGroup`) whose lifetimes are
/// tied to the [`FrameArena`](super::arena::FrameArena) that allocates
/// them.  The type system enforces that all such references remain valid
/// for the duration of the frame.
pub trait PassNode: Send + Sync {
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

// ─── NodeSlot ──────────────────────────────────────────────────────────────

/// Type-erased handle to a [`PassNode`] allocated on the [`FrameArena`].
///
/// Stores a fat pointer (data + vtable) to a `dyn PassNode` trait object.
/// The pointed-to memory resides either in the frame arena (owned) or in
/// an externally managed location (borrowed).
///
/// # Safety Contract
///
/// The arena allocation must outlive this handle.  In practice this is
/// guaranteed by the [`FrameComposer`] which ensures the sequence:
/// `cleanup_nodes()` → `arena.reset()`.
pub(crate) struct NodeSlot {
    /// Fat pointer to the `dyn PassNode` trait object.
    pub(crate) ptr: *mut dyn PassNode,
    /// Whether this node was allocated on the frame arena.
    ///
    /// - `true` — arena-owned; [`cleanup_nodes`](RenderGraph::cleanup_nodes)
    ///   calls [`drop_in_place`](std::ptr::drop_in_place) during frame teardown.
    /// - `false` — externally borrowed (e.g. via [`add_pass_borrowed`]);
    ///   the graph must **not** drop the node.
    owned: bool,
}

// SAFETY: `NodeSlot` wraps a pointer to a `dyn PassNode` which itself
// requires `Send + Sync`.  The engine's single-threaded frame model
// guarantees no concurrent access to the pointed-to data.
unsafe impl Send for NodeSlot {}
unsafe impl Sync for NodeSlot {}

impl NodeSlot {
    /// Creates a slot for an arena-allocated node.
    #[inline]
    pub(crate) fn new_owned(ptr: *mut dyn PassNode) -> Self {
        Self { ptr, owned: true }
    }

    /// Creates a slot for an externally-owned (borrowed) node.
    #[inline]
    pub(crate) fn new_borrowed(ptr: *mut dyn PassNode) -> Self {
        Self { ptr, owned: false }
    }

    /// Whether this slot owns the pointed-to node (arena-allocated).
    #[inline]
    pub(crate) fn is_owned(&self) -> bool {
        self.owned
    }
}

// ─── PassRecord ────────────────────────────────────────────────────────────

/// Per-pass metadata stored in the [`RenderGraph`].
///
/// Holds per-frame topology information (reads, writes, dependencies) and
/// a type-erased handle to the [`PassNode`] allocated on the
/// [`FrameArena`](super::arena::FrameArena).
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

    /// Type-erased handle to the arena-allocated pass node.
    pub(crate) node: Option<NodeSlot>,

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

    /// Returns a mutable reference to the pass node.
    ///
    /// # Panics
    /// Panics if the node has not been inserted yet.
    ///
    /// # Safety
    ///
    /// The returned reference is derived from a raw pointer stored in
    /// [`NodeSlot`].  Callers must ensure no aliasing mutable references
    /// exist.  In practice, the sequential prepare→execute pipeline
    /// guarantees this.
    #[inline]
    pub fn get_pass_mut(&mut self) -> &mut dyn PassNode {
        let slot = self.node.as_ref().expect("PassRecord node not set");
        // SAFETY: The pointer was set by `add_pass` or `add_pass_borrowed`
        // and remains valid until `cleanup_nodes()` + `arena.reset()`.
        // `&mut self` guarantees exclusive access to this record.
        unsafe { &mut *slot.ptr }
    }
}
