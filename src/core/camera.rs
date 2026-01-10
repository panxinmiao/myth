use glam::{Mat4, Vec3, Vec4, Affine3A};
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
    pub transform: Affine3A,

    // === 投影属性 (Projection Only) ===
    pub projection_type: ProjectionType,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,

    // 正交参数
    pub ortho_size: f32, 

    world_matrix: Affine3A,
    projection_matrix: Mat4,
    view_matrix: Mat4,
    view_projection_matrix: Mat4,
    frustum: Frustum,
}

#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

impl Camera {
    pub fn new_perspective(fov_degrees: f32, aspect: f32, near: f32, far: f32) -> Self {
        let mut cam = Self {
            node_id: None, // 默认为游离状态
            transform: Affine3A::IDENTITY,
            projection_type: ProjectionType::Perspective,
            fov: fov_degrees.to_radians(),
            aspect,
            near,
            far,
            ortho_size: 10.0,
            
            world_matrix: Affine3A::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
            view_projection_matrix: Mat4::IDENTITY,
            frustum: Frustum::default(),
        };

        cam.update_projection_matrix();
        cam
    }

    pub fn update_projection_matrix(&mut self){
        self.projection_matrix =  match self.projection_type {
            ProjectionType::Perspective => {
                // glam 的 perspective_rh 默认是为了 WGPU/Vulkan 设计的 (0 to 1)
                Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
            }
            ProjectionType::Orthographic => {
                let w = self.ortho_size * self.aspect;
                let h = self.ortho_size;
                Mat4::orthographic_rh(-w, w, -h, h, self.near, self.far)
            }
        };
    }

    pub fn update_matrix_world(&mut self, scene: &Scene) {
        // 1. 获取相机在世界空间中的 Transform 矩阵
        let world_transform = if let Some(id) = self.node_id {
            // Attach 模式：相信 Scene Graph
            if let Some(node) = scene.nodes.get(id) {
                // 注意：这里我们拿的是 Node 的 World Matrix
                // 假设 Scene 已经调用过 update_matrix_world()
                *node.world_matrix()
            } else {
                // 节点丢了？回退
                self.transform
            }
        } else {
            // Detached 模式：相信自己的 transform
            self.transform
        };

        self.world_matrix = world_transform;

        // 2. 计算 View Matrix (World -> Camera)
        // View = WorldInverse
        self.view_matrix = Mat4::from(world_transform).inverse();

        // 3. 计算 VP Matrix
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;

        // 4. 更新视锥体 (用于剔除)
        self.frustum = Frustum::from_matrix(self.view_projection_matrix);
    }

    pub fn world_matrix(&self) -> &Affine3A { &self.world_matrix }
    pub fn projection_matrix(&self) -> &Mat4 { &self.projection_matrix }
    pub fn view_matrix(&self) -> &Mat4 { &self.view_matrix }
    pub fn view_projection_matrix(&self) -> &Mat4 { &self.view_projection_matrix }
    pub fn frustum(&self) -> &Frustum { &self.frustum }


    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let position = Vec3::from(self.transform.translation);        
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



#[derive(Debug, Clone, Copy, Default)]
pub struct Frustum {
    planes: [Vec4; 6], // Left, Right, Bottom, Top, Near, Far
}

impl Frustum {
    pub fn from_matrix(m: Mat4) -> Self {
        let rows = [
            m.row(0), m.row(1), m.row(2), m.row(3)
        ];
        
        let mut planes = [Vec4::ZERO; 6];
        // 提取公式: https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
        // Gribb-Hartmann 方法
        
        // Left:   row4 + row1
        planes[0] = rows[3] + rows[0];
        // Right:  row4 - row1
        planes[1] = rows[3] - rows[0];
        // Bottom: row4 + row2
        planes[2] = rows[3] + rows[1];
        // Top:    row4 - row2
        planes[3] = rows[3] - rows[1];
        // Near:   row4 + row3 (for OpenGL/WGPU range [0,1] might differ, usually row3 for [0,1])
        // WGPU NDC Z is [0, 1]. 
        // Plane extraction depends on projection matrix implementation. 
        // Assuming Standard:
        planes[4] = rows[2]; // Near
        planes[5] = rows[3] - rows[2]; // Far

        // Normalize
        for plane in &mut planes {
            let length = Vec3::new(plane.x, plane.y, plane.z).length();
            *plane /= length;
        }

        Self { planes }
    }

    // 简单的球体相交检测
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false;
            }
        }
        true
    }

}