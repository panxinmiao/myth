use super::builder::PassBuilder;
use super::types::TextureNodeId;
use smallvec::SmallVec;

pub trait PassNode {
    fn name(&self) -> &'static str;
    fn setup(&mut self, builder: &mut PassBuilder);
}

pub struct PassRecord {
    pub name: &'static str,
    // 直接借用外部的 mut 引用，坚决不使用 Box 产生堆分配！
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
}
