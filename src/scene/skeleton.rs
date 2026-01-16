use glam::{Affine3A, Mat4};
use uuid::Uuid;
use thunderdome::Arena;

use crate::scene::{Node, NodeIndex, SkeletonKey};


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BindMode {
    /// 骨骼跟随节点移动 (最常用，如角色蒙皮)
    /// 此时 inverse_bind_matrix = node.world_matrix.inverse()
    Attached,
    /// 骨骼与节点分离 (特殊用途)
    /// 此时使用绑定时记录的静态 inverse_matrix
    Detached,
}

#[derive(Debug, Clone)]
pub struct SkinBinding {
    pub skeleton: SkeletonKey,
    pub bind_mode: BindMode,
    /// 绑定时的逆矩阵快照 (用于 Detached 模式)
    pub bind_matrix_inv: Affine3A, 
}

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub id: Uuid,
    pub name: String,

    // === 核心数据 ===
    // 骨骼列表：有序排列，对应 shader 中的 joint index
    // bones[i] 对应的就是 shader 中的 joints[i]
    pub bones: Vec<NodeIndex>, 

    // 逆绑定矩阵 (Inverse Bind Matrices)
    // 这是静态数据，从 glTF 加载后通常不会变
    // 作用：把顶点从 Mesh 空间变换到 骨骼局部空间
    pub inverse_bind_matrices: Vec<Affine3A>,

    // === 运行时数据 ===
    // 最终计算出的矩阵数组，每一帧都会更新
    // 数据流向：这里的 data -> copy to GPU Uniform Buffer
    pub joint_matrices: Vec<Mat4>,
}

impl Skeleton {
    pub fn new(name: &str, bones: Vec<NodeIndex>, inverse_bind_matrices: Vec<Affine3A>) -> Self {
        let count = bones.len();

        let joint_matrices = vec![Mat4::IDENTITY; count];
        
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            bones,
            inverse_bind_matrices,
            joint_matrices,
        }
    }

    /// 更新骨骼矩阵
    /// 
    /// # 参数
    /// * `nodes`: 全局节点存储，我们需要从中读取每个 bone 的 world_matrix
    /// * `root_matrix_inv`: SkinnedMesh 所在节点的世界矩阵的逆矩阵
    ///                      (用于将骨骼变换转换回 Mesh 的局部空间)
    pub fn compute_joint_matrices(
        &mut self, 
        nodes: &Arena<Node>,
        root_matrix_inv: Affine3A
    ) {
        for (i, &bone_idx) in self.bones.iter().enumerate() {
            // 1. 获取当前骨骼在这一帧的世界变换 (由场景图系统计算好的)
            // 注意：这里假设 nodes[bone_idx] 绝对存在，生产环境需要安全检查
            let bone_world_matrix = nodes[bone_idx].transform.world_matrix;

            // 2. 获取对应的逆绑定矩阵
            let ibm = self.inverse_bind_matrices[i];

            // 3. 计算最终的 Joint Matrix
            // 顺序很重要：先应用 IBM (变回骨骼局部)，再应用当前骨骼世界变换，最后(可选)抵消 Mesh 自身变换
            self.joint_matrices[i] = (root_matrix_inv * bone_world_matrix * ibm).into();
        }
    }
}