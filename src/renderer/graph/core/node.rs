use crate::renderer::graph::core::context::{ExecuteContext, PrepareContext};

use super::types::TextureNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// Pure GPU command recorder for a single render or compute pass.
///
/// `PassNode` is intentionally minimal — it carries only lightweight IDs,
/// borrowed references to persistent GPU resources (`&'a wgpu::BindGroup`),
/// and transient bind-group cache keys.  All persistent GPU resources
/// (layouts, pipelines, buffers) live in the owning `Feature`.
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
/// The lifetime parameter `'a` represents the duration of a single frame.
/// Concrete implementations may carry frame-scoped borrowed references
/// (e.g. `&'a wgpu::BindGroup`, `&'a [ShadowLightInstance]`) whose
/// lifetimes are tied to either:
///
/// - The [`FrameArena`](super::arena::FrameArena) that allocates the node.
/// - Persistent `Feature` state that outlives the frame.
///
/// Because all nodes are allocated on the arena and the arena is reset
/// at frame boundaries, **no `Drop` glue is required** — the arena
/// reclaims all memory in O(1) without running destructors.
pub trait PassNode<'a>: Send + Sync {
    /// Assemble transient `BindGroup`s after physical resource allocation.
    ///
    /// Only `BindGroup`s that reference RDG-managed transient textures
    /// should be created here.  The context deliberately excludes heavy
    /// infrastructure (shader compiler, asset server, etc.).
    #[allow(unused_variables)]
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {}

    /// Record GPU commands into the shared encoder.
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder);
}

// ─── NodeSlot ──────────────────────────────────────────────────────────────

/// Type-erased handle to a [`PassNode`] allocated on the [`FrameArena`].
///
/// Stores a fat pointer (data + vtable) to a `dyn PassNode` trait object.
/// No ownership or drop semantics — the arena reclaims all memory in O(1)
/// via [`FrameArena::reset()`](super::arena::FrameArena::reset) at frame
/// boundaries.
///
/// # Safety Contract
///
/// The arena allocation must outlive this handle.  In practice this is
/// guaranteed by the frame lifecycle: `arena.reset()` is called only
/// after the entire execute phase completes.
pub(crate) struct NodeSlot {
    /// Fat pointer to the `dyn PassNode` trait object.
    pub(crate) ptr: *mut dyn for<'a> PassNode<'a>,

    // ⚠️ 临时防泄漏机制（在所有节点变成 POD 前，我们需要它）
    pub(crate) needs_drop: bool,
}

// SAFETY: `NodeSlot` wraps a pointer to a `dyn PassNode` which itself
// requires `Send + Sync`.  The engine's single-threaded frame model
// guarantees no concurrent access to the pointed-to data.
unsafe impl Send for NodeSlot {}
unsafe impl Sync for NodeSlot {}

impl NodeSlot {
    /// Creates a slot for an arena-allocated or externally-owned node.
    #[inline]
    pub(crate) fn new<'a, N: PassNode<'a> + 'a>(ptr: *mut N) -> Self {
        let erased_ptr = unsafe { std::mem::transmute(ptr as *mut dyn PassNode<'a>) };
        Self { 
            ptr: erased_ptr,
            needs_drop: std::mem::needs_drop::<N>(), // 记录是否需要 Drop
        }
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
    pub fn get_pass_mut<'a>(&mut self) -> &mut (dyn PassNode<'a> + 'a) {
        let slot = self.node.as_ref().expect("PassRecord node not set");
        // SAFETY: The pointer was set by `add_pass` or `add_pass_borrowed`
        // and remains valid until `arena.reset()`.
        // `&mut self` guarantees exclusive access to this record.
        unsafe { std::mem::transmute(&mut *slot.ptr) }
    }
}
