use glam::{Affine3A, Mat4, Vec3, Vec3A, Vec4};
use std::borrow::Cow;
use uuid::Uuid;


/// [新增] 纯栈上渲染相机对象 (POD)
/// todo : 考虑直接满足std140对齐要求?
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RenderCamera {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub view_projection_matrix: Mat4,
    pub position: Vec3A, // 世界坐标位置，Lighting 需要
    pub frustum: Frustum, // 剔除需要
    pub near: f32,
    pub far: f32,
}


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
    pub fn new_perspective(fov: f32, aspect: f32, near: f32) -> Self {
        let mut cam = Self {
            uuid: Uuid::new_v4(),
            name: Cow::Owned("Camera".to_string()),
            projection_type: ProjectionType::Perspective,
            fov: fov.to_radians(),
            aspect,
            near,
            far: f32::INFINITY,
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
                Mat4::perspective_infinite_reverse_rh(self.fov, self.aspect, self.near)
            }
            ProjectionType::Orthographic => {
                let w = self.ortho_size * self.aspect;
                let h = self.ortho_size;
                // Reverse Z, swap near and far
                Mat4::orthographic_rh(-w, w, -h, h, self.far, self.near)
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

    pub fn extract_render_camera(&self) -> RenderCamera {
        RenderCamera {
            view_matrix: self.view_matrix,
            projection_matrix: self.projection_matrix,
            view_projection_matrix: self.view_projection_matrix,
            // 从世界矩阵提取位置 (Translation)
            position: self.world_matrix.translation.into(),
            frustum: self.frustum, // Frustum 也是 Copy 的
            near: self.near,
            far: self.far,
        }
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

        // [Reverse-Z] Near Plane 对应 NDC z = 1.0
        // 剔除条件: z_c / w_c > 1.0 (比近平面更近)
        // 保留条件: z_c <= w_c => w_c - z_c >= 0
        planes[4] = rows[3] - rows[2]; // Near

        // 无限远
        planes[5] = Vec4::new(0.0, 0.0, 0.0, 0.0);

        // Normalize
        // for plane in &mut planes {
        //     let length = Vec3::new(plane.x, plane.y, plane.z).length();
        //     if length > 0.0 {
        //         *plane /= length;
        //     }
        // }

        // Normalize with Safety Check
        for (i, plane) in planes.iter_mut().enumerate() {
            // 跳过 Far Plane 的归一化，防止 NaN
            if i == 5 { continue; }

            let length = Vec3::new(plane.x, plane.y, plane.z).length();
            // 增加 epsilon 检查防止除以 0
            if length > 1e-6 {
                *plane /= length;
            } else {
                // 如果法线长度为0（异常情况），让该平面失效（永远不剔除）
                // 设置为 0,0,0,0，这样 dot(center) + 0 < -r 永远为 false (0 < negative) 吗？
                // 不，0 < -r (r>0) 是 false。所以物体可见。安全。
                *plane = Vec4::ZERO; 
            }
        }

        Self { planes }
    }

    // 简单的球体相交检测
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for (i, plane) in self.planes.iter().enumerate() {
            // [可选] 显式跳过 Far Plane (index 5)
            if i == 5 { continue; }
            
            // 如果平面是零向量（无效平面），直接跳过
            if plane.x == 0.0 && plane.y == 0.0 && plane.z == 0.0 {
                continue;
            }

            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false;
            }
        }
        true
    }

    /// AABB 与视锥体相交检测
    /// 使用平面-AABB 测试，如果 AABB 完全在任意平面外侧则返回 false
    pub fn intersects_box(&self, min: Vec3, max: Vec3) -> bool {
        for (i, plane) in self.planes.iter().enumerate() {
            // 跳过 Far Plane
            if i == 5 { continue; }
            
            // 如果平面是零向量（无效平面），直接跳过
            if plane.x == 0.0 && plane.y == 0.0 && plane.z == 0.0 {
                continue;
            }

            // 找到 AABB 上距离平面最近的点（p-vertex）
            // 如果这个点在平面外侧，则整个 AABB 都在外侧
            let p = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );

            let dist = plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
            if dist < 0.0 {
                return false;
            }
        }
        true
    }
}