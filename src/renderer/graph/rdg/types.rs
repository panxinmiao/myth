use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureNodeId(pub u32);

pub struct ResourceRecord {
    pub name: &'static str,
    pub is_external: bool,
    // 使用 SmallVec 避免堆分配
    pub producers: SmallVec<[usize; 4]>,
    pub consumers: SmallVec<[usize; 8]>,
}
