use crate::scene::NodeHandle;

/// 定义动画数据的目标属性
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPath {
    Translation, // 对应 transform.position
    Rotation,    // 对应 transform.rotation
    Scale,       // 对应 transform.scale
    Weights,     // 对应 Morph Target Weights
}

/// 绑定关系：将 Clip 中的第 `track_index` 条轨道，映射到 scene 中的 `node_handle` 的 target 属性
#[derive(Debug, Clone)]
pub struct PropertyBinding {
    pub track_index: usize,
    pub node_handle: NodeHandle,
    pub target: TargetPath,
}
