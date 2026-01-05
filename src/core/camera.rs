use glam::{Mat4, Vec3, Affine3A};
use thunderdome::Index;
use super::scene::Scene; // 需要引用 Scene 来查找 Node

#[derive(Debug, Clone)]
pub struct Camera {
// === 关联 ===
    // 相机依附于哪个节点？
    // 通过这个 ID，我们可以去 Scene.nodes 里查到它的 World Matrix (用于计算 View Matrix)
    pub node_id: Option<Index>,

    // === 游离模式下的 Transform ===
    // 只有在 node_id 为 None 时，以下字段才生效
    // 我们直接存储 World Matrix (Affine3A)，方便计算
    pub transform: Affine3A,

    // === 投影属性 (Projection Only) ===
    pub projection_type: ProjectionType,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,

    // 正交参数
    pub ortho_size: f32, 
}

#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

impl Camera {
    pub fn new_perspective(fov_degrees: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            node_id: None, // 默认为游离状态
            transform: Affine3A::IDENTITY,

            projection_type: ProjectionType::Perspective,
            fov: fov_degrees.to_radians(),
            aspect,
            near,
            far,
            ortho_size: 10.0,
        }
    }

    /// 获取投影矩阵 (P)
    pub fn get_projection_matrix(&self) -> Mat4 {
        match self.projection_type {
            ProjectionType::Perspective => {
                // 注意：WGPU 的 NDC 深度范围是 [0, 1]，而 OpenGL 是 [-1, 1]
                // glam 的 perspective_rh 默认是为了 WGPU/Vulkan 设计的 (0 to 1)
                Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
            }
            ProjectionType::Orthographic => {
                let w = self.ortho_size * self.aspect;
                let h = self.ortho_size;
                Mat4::orthographic_rh(-w, w, -h, h, self.near, self.far)
            }
        }
    }

    /// 获取视图矩阵 (V)
    /// View Matrix 本质上是相机 World Matrix 的逆矩阵
    pub fn get_view_matrix(&self, scene: Option<&Scene>) -> Mat4 {
        let world_matrix = if let Some(id) = self.node_id {

            // A. Attached Mode: 从 Scene Graph 获取矩阵
            if let Some(node) = scene.and_then(|s| s.get_node(id)) {
                *node.world_matrix()
            } else {
                // 如果 ID 无效 (例如节点被删了)，回退到内部 Transform
                self.transform
            }
        } else {
            // B. Detached Mode: 使用自身管理的矩阵
            self.transform
        };

        // View Matrix 是 Camera World Matrix 的逆矩阵
        Mat4::from(world_matrix).inverse()
    }

    /// 获取 View-Projection 矩阵 (VP)
    pub fn get_view_projection_matrix(&self, scene: Option<&Scene>) -> Mat4 {
        self.get_projection_matrix() * self.get_view_matrix(scene)
    }

    pub fn get_view_projection_matrix_inverse(&self, scene: Option<&Scene>) -> Mat4 {
        self.get_view_projection_matrix(scene).inverse()
    }


    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let position = Vec3::from(self.transform.translation);
        // glam 的 look_at_rh 生成的是 View Matrix (World -> Camera)
        // 但我们需要存的是 World Matrix (Camera -> World)
        // 所以这里要计算 look_at 的逆，或者直接构建 basis
        
        let forward = (target - position).normalize();
        let right = forward.cross(up).normalize();
        let new_up = right.cross(forward); // 正交化

        // 构建旋转矩阵 (列向量)
        let rotation = Mat4::from_cols(
            right.extend(0.0),
            new_up.extend(0.0),
            -forward.extend(0.0), // Camera looks down -Z
            Vec3::ZERO.extend(1.0),
        );
        
        // 重新组合 Affine3A
        let mat = Mat4::from_translation(position) * rotation;
        self.transform = Affine3A::from_mat4(mat);
    }

}