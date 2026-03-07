use crate::renderer::graph::rdg::context::{PassPrepareContext, RdgPrepareContext};

use super::builder::PassBuilder;
use super::context::RdgExecuteContext;
use super::types::TextureNodeId;
use smallvec::SmallVec;
use wgpu::CommandEncoder;

/// A single render or compute pass in the declarative render graph.
///
/// # Lifecycle
///
/// 1. **`prepare_resources`** — called by the Composer *before* the graph is
///    built.  Full GPU infrastructure is available.  Create layouts, compile
///    pipelines, upload non-transient buffers here.
/// 2. **`setup`** — declare resource read/write topology for the graph.
/// 3. **`prepare`** — called *after* graph compilation and transient memory
///    allocation.  Only assemble `BindGroup`s that reference RDG-managed
///    transient textures.  Context is intentionally minimal.
/// 4. **`execute`** — record GPU commands into the shared encoder.
pub trait PassNode {
    fn name(&self) -> &'static str;

    /// Pre-RDG resource preparation.
    ///
    /// Create `BindGroupLayout`s, compile pipelines, upload persistent GPU
    /// data.  Called once per frame for each active pass, **before** the
    /// render graph is constructed.
    #[allow(unused_variables)]
    fn prepare_resources(&mut self, ctx: &mut PassPrepareContext) {}

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
