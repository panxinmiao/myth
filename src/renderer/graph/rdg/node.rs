use crate::renderer::graph::rdg::context::RdgPrepareContext;

use super::builder::PassBuilder;
use super::context::RdgExecuteContext;
use super::types::TextureNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// A single render or compute pass in the declarative render graph.
///
/// Pass nodes are **transient**: created per frame by their owning
/// [`Feature`](super::feature) during graph assembly and destroyed after
/// execution. They carry only lightweight data (texture node IDs, pipeline
/// IDs, cached bind-group keys) — no persistent GPU resources.
///
/// # Lifecycle
///
/// 1. **`setup`** — declare resource read/write topology for the graph.
/// 2. **`prepare`** — called *after* graph compilation and transient memory
///    allocation. Assemble `BindGroup`s that reference RDG-managed
///    transient textures. Context is intentionally minimal.
/// 3. **`execute`** — record GPU commands into the shared encoder.
pub trait PassNode {
    /// Human-readable debug name for this pass.
    fn name(&self) -> &'static str;

    /// Declare resource dependencies for the graph topology.
    ///
    /// Called immediately when the node is added to the graph. Must declare
    /// all texture reads and writes so the compiler can build the DAG.
    fn setup(&mut self, builder: &mut PassBuilder);

    /// Assemble transient `BindGroup`s after physical resource allocation.
    ///
    /// Only `BindGroup`s that reference RDG-managed transient textures
    /// should be created here. The context deliberately excludes heavy
    /// infrastructure (shader compiler, asset server, etc.).
    #[allow(unused_variables)]
    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {}

    /// Record GPU commands into the shared encoder.
    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut CommandEncoder);
}

pub struct PassRecord {
    pub name: &'static str,
    // 直接借用外部的 mut 引用，不使用 Box 产生堆分配
    pub node: *mut (dyn PassNode + 'static),

    // 局部的依赖声明全部在栈上完成 (或随 RenderGraph 驻留)
    pub reads: SmallVec<[TextureNodeId; 8]>,
    pub writes: SmallVec<[TextureNodeId; 4]>,
    pub creates: SmallVec<[TextureNodeId; 4]>,

    // 编译期状态
    pub physical_dependencies: SmallVec<[usize; 8]>,
    pub has_side_effect: bool,
    pub reference_count: u32,
}

impl PassRecord {
    pub fn new(name: &'static str, node: *mut (dyn PassNode + 'static)) -> Self {
        Self {
            name,
            node,
            reads: SmallVec::new(),
            writes: SmallVec::new(),
            creates: SmallVec::new(),
            physical_dependencies: SmallVec::new(),
            has_side_effect: false,
            reference_count: 0,
        }
    }

    #[inline]
    pub fn get_pass_mut(&self) -> &mut dyn PassNode {
        unsafe { &mut *self.node }
    }
}
