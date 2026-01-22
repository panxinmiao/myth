use glam::{Affine3A, Mat4, Vec3};
use uuid::Uuid;
use slotmap::SlotMap;

use crate::resources::BoundingBox;
use crate::{resources::buffer::CpuBuffer, scene::{Node, NodeHandle, SkeletonKey}};


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
    pub bones: Vec<NodeHandle>, 

    // 逆绑定矩阵 (Inverse Bind Matrices)
    // 这是静态数据，从 glTF 加载后通常不会变
    // 作用：把顶点从 Mesh 空间变换到 骨骼局部空间
    pub(crate) inverse_bind_matrices: Vec<Affine3A>,

    // === 包围盒数据 (用于视锥剔除) ===
    /// 局部空间包围盒 (相对于根骨骼)
    pub(crate) local_bounds: Option<BoundingBox>,
    /// 根骨骼索引 (通常是 bones[0])
    pub(crate) root_bone_index: usize,

    // === 运行时数据 ===
    // 最终计算出的矩阵数组，每一帧都会更新
    // 数据流向：这里的 data -> copy to GPU Uniform Buffer
    pub(crate) joint_matrices: CpuBuffer<Vec<Mat4>>,
}

impl Skeleton {
    pub fn new(name: &str, bones: Vec<NodeHandle>, inverse_bind_matrices: Vec<Affine3A>) -> Self {
        let count = bones.len();

        let joint_matrices = CpuBuffer::new(
            vec![Mat4::IDENTITY; count],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some(&format!("SkeletonJointMatrices_{}", name)),
        );
        
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            bones,
            inverse_bind_matrices,
            local_bounds: None,
            root_bone_index: 0,
            joint_matrices,
        }
    }

    /// 获取局部空间包围盒（惰性计算）
    #[inline]
    pub fn local_bounds(&self) -> Option<&BoundingBox> {
        self.local_bounds.as_ref()
    }

    /// 获取根骨骼节点句柄
    #[inline]
    pub fn root_bone(&self) -> Option<NodeHandle> {
        self.bones.get(self.root_bone_index).copied()
    }

    /// 计算骨骼的局部空间包围盒
    /// 基于骨骼的绑定姿态位置，计算相对于根骨骼的包围盒
    pub fn compute_local_bounds(&mut self, nodes: &SlotMap<NodeHandle, Node>) {
        if self.bones.is_empty() {
            return;
        }

        // 获取根骨骼的世界矩阵逆矩阵
        let root_bone_handle = self.bones[self.root_bone_index];
        let Some(root_node) = nodes.get(root_bone_handle) else {
            return;
        };
        let root_world_inv = root_node.transform.world_matrix.inverse();

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut valid_count = 0;

        for &bone_handle in &self.bones {
            if let Some(bone_node) = nodes.get(bone_handle) {
                let world_pos: Vec3 = bone_node.transform.world_matrix.translation.into();
                let local_pos = root_world_inv.transform_point3(world_pos);
                
                min = min.min(local_pos);
                max = max.max(local_pos);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            let size = max - min;
            let padding = size * 0.15;
            
            self.local_bounds = Some(BoundingBox {
                min: min - padding,
                max: max + padding,
            });
        }
    }

    /// 获取世界空间包围盒
    #[inline]
    pub fn world_bounds(&self, nodes: &SlotMap<NodeHandle, Node>) -> Option<BoundingBox> {
        let local_bounds = self.local_bounds.as_ref()?;
        let root_bone_handle = self.bones.get(self.root_bone_index)?;
        let root_node = nodes.get(*root_bone_handle)?;
        Some(local_bounds.transform(&root_node.transform.world_matrix))
    }

    /// 更新骨骼矩阵
    /// 
    /// # 参数
    /// * `nodes`: 全局节点存储，我们需要从中读取每个 bone 的 world_matrix
    /// * `root_matrix_inv`: SkinnedMesh 所在节点的世界矩阵的逆矩阵
    ///                      (用于将骨骼变换转换回 Mesh 的局部空间)
    pub fn compute_joint_matrices(
        &mut self, 
        nodes: &SlotMap<NodeHandle, Node>,
        root_matrix_inv: Affine3A
    ) {
        for (i, &bone_handle) in self.bones.iter().enumerate() {
            // 1. 获取当前骨骼在这一帧的世界变换 (由场景图系统计算好的)
            let Some(bone_node) = nodes.get(bone_handle) else {
                continue;
            };
            let bone_world_matrix = bone_node.transform.world_matrix;

            // 2. 获取对应的逆绑定矩阵
            let ibm = self.inverse_bind_matrices[i];

            // 3. 计算最终的 Joint Matrix
            // 顺序很重要：先应用 IBM (变回骨骼局部)，再应用当前骨骼世界变换，最后(可选)抵消 Mesh 自身变换
            self.joint_matrices.write()[i] = (root_matrix_inv * bone_world_matrix * ibm).into();
        }
    }
}