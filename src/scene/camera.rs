use glam::{Mat4, Vec3, Vec4, Affine3A};
use std::borrow::Cow;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Camera {
    pub uuid: Uuid,
    pub name: Cow<'static, str>,

    // === 投影属性 (Projection Only) ===
    pub projection_type: ProjectionType,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub ortho_size: f32, 

    // 缓存的矩阵 renderer只读
    pub(crate) world_matrix: Affine3A,
    pub(crate) view_matrix: Mat4,
    pub(crate) projection_matrix: Mat4,
    pub(crate) view_projection_matrix: Mat4,
    pub(crate) frustum: Frustum,
}

#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

impl Camera {
    pub fn new_perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let mut cam = Self {
            uuid: Uuid::new_v4(),
            name: Cow::Owned("Camera".to_string()),
            projection_type: ProjectionType::Perspective,
            fov: fov.to_radians(),
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

        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
        self.frustum = Frustum::from_matrix(self.view_projection_matrix);
    }

    pub fn update_view_projection(&mut self, world_transform: &Affine3A) {
        self.world_matrix = *world_transform;

        // 1. View Matrix = World Inverse
        self.view_matrix = Mat4::from(*world_transform).inverse();
        
        // 2. VP
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
        
        // 3. Frustum
        self.frustum = Frustum::from_matrix(self.view_projection_matrix);
    }


    // pub fn look_at(&mut self, target: Vec3, up: Vec3) {
    //     let position = Vec3::from(self.transform.translation);        
    //     let forward = (target - position).normalize();
    //     let right = forward.cross(up).normalize();
    //     let new_up = right.cross(forward); // 正交化

    //     // 构建旋转矩阵 (列向量)
    //     let rotation = Mat4::from_cols(
    //         right.extend(0.0),
    //         new_up.extend(0.0),
    //         -forward.extend(0.0), // Camera looks down -Z
    //         Vec3::ZERO.extend(1.0),
    //     );
        
    //     // 重新组合 Affine3A
    //     let mat = Mat4::from_translation(position) * rotation;
    //     self.transform = Affine3A::from_mat4(mat);
    // }

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