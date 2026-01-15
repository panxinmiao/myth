use glam::{Affine3A, Mat4, Quat, Vec3, Mat3, EulerRot};

/// Transform 组件
/// 封装了位置、旋转、缩放（TRS）以及矩阵缓存和脏检查逻辑。
#[derive(Debug, Clone)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,

    // 矩阵缓存（供渲染器读取）
    pub(crate) local_matrix: Affine3A,
    pub(crate) world_matrix: Affine3A,

    // 脏检查状态
    last_position: Vec3,
    last_rotation: Quat,
    last_scale: Vec3,
    force_update: bool,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,

            local_matrix: Affine3A::IDENTITY,
            world_matrix: Affine3A::IDENTITY,

            last_position: Vec3::ZERO,
            last_rotation: Quat::IDENTITY,
            last_scale: Vec3::ONE,
            force_update: true,
        }
    }

    /// 检查并更新局部矩阵，返回是否发生了变化
    pub fn update_local_matrix(&mut self) -> bool {
        let changed = self.position != self.last_position
            || self.rotation != self.last_rotation
            || self.scale != self.last_scale
            || self.force_update;

        if changed {
            self.local_matrix = Affine3A::from_scale_rotation_translation(
                self.scale,
                self.rotation,
                self.position,
            );

            self.last_position = self.position;
            self.last_rotation = self.rotation;
            self.last_scale = self.scale;
            self.force_update = false;
        }

        changed
    }

    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) {
        self.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }

    pub fn rotation_euler(&self) -> Vec3 {
        let (x, y, z) = self.rotation.to_euler(EulerRot::XYZ);
        Vec3::new(x, y, z)
    }

    pub fn set_rotation_euler_with_order(&mut self, x: f32, y: f32, z: f32, order: EulerRot) {
        self.rotation = Quat::from_euler(order, x, y, z);
    }

    #[inline]
    pub fn local_matrix(&self) -> &Affine3A {
        &self.local_matrix
    }

    #[inline]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.world_matrix
    }

    #[inline]
    pub fn world_matrix_as_mat4(&self) -> Mat4 {
        Mat4::from(self.world_matrix)
    }

    pub fn set_world_matrix(&mut self, mat: Affine3A) {
        self.world_matrix = mat;
    }

    /// 直接设置局部矩阵（例如来自 glTF 或物理引擎）
    /// 会分解为 TRS 并同步状态
    pub fn apply_local_matrix(&mut self, mat: Affine3A) {
        self.local_matrix = mat;
        let (scale, rotation, translation) = mat.to_scale_rotation_translation();
        self.scale = scale;
        self.rotation = rotation;
        self.position = translation;
        self.last_scale = scale;
        self.last_rotation = rotation;
        self.last_position = translation;
        self.force_update = false;
    }

    pub fn apply_local_matrix_from_mat4(&mut self, mat: Mat4) {
        let affine = Affine3A::from_mat4(mat);
        self.apply_local_matrix(affine);
    }

    /// 将变换面向目标点（在父坐标系中）
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = (target - self.position).normalize();
        if forward.cross(up).length_squared() < 1e-4 {
            return;
        }
        let right = forward.cross(up).normalize();
        let new_up = right.cross(forward).normalize();
        let rot_mat = Mat3::from_cols(right, new_up, -forward);
        self.rotation = Quat::from_mat3(&rot_mat);
    }

    pub fn mark_dirty(&mut self) {
        self.force_update = true;
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}
