use glam::{Affine3A, Mat4, Quat, Vec3, Mat3, EulerRot};

/// Transform 组件
/// 
/// 封装了节点的位置、旋转、缩放（TRS）以及矩阵缓存和脏检查逻辑。
/// 这是一个独立的数据组件，可以被 Node 组合，也可以独立使用。
#[derive(Debug, Clone)]
pub struct Transform {
    // === Public 属性 ===
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,

    // === 矩阵缓存 (Internal) ===
    // 依然保留 pub(crate) 供渲染器读取，但对用户隐藏细节
    pub(crate) local_matrix: Affine3A,
    pub(crate) world_matrix: Affine3A,

    // === 脏检查状态 (Private) ===
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

    // ========================================================================
    // 核心逻辑：智能更新 (Shadow State Check)
    // ========================================================================

    /// 检查并更新局部矩阵
    /// 返回值: bool (是否发生了变化)
    pub fn update_local_matrix(&mut self) -> bool {
        // 1. 脏检查：对比当前 pub 属性和 last 私有属性
        let changed = self.position != self.last_position 
                   || self.rotation != self.last_rotation 
                   || self.scale != self.last_scale
                   || self.force_update;

        if changed {
            // 2. 只有变了才重算矩阵
            self.local_matrix = Affine3A::from_scale_rotation_translation(
                self.scale,
                self.rotation,
                self.position,
            );

            // 3. 同步影子状态
            self.last_position = self.position;
            self.last_rotation = self.rotation;
            self.last_scale = self.scale;
            self.force_update = false;
        }

        changed
    }

    // ========================================================================
    // Getters & Helpers
    // ========================================================================

    /// Helper：设置欧拉角旋转
    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) {
        self.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }

    /// 获取当前的欧拉角 (XYZ 顺序)
    pub fn rotation_euler(&self) -> Vec3 {
        let (x, y, z) = self.rotation.to_euler(EulerRot::XYZ);
        Vec3::new(x, y, z)
    }

    /// 高级：支持指定旋转顺序
    pub fn set_rotation_euler_with_order(&mut self, x: f32, y: f32, z: f32, order: EulerRot) {
        self.rotation = Quat::from_euler(order, x, y, z);
    }

    /// 获取局部矩阵 (Affine3A)
    #[inline]
    pub fn local_matrix(&self) -> &Affine3A {
        &self.local_matrix
    }

    /// 获取世界矩阵 (Affine3A) - 供 CPU 端物理/逻辑计算使用
    #[inline]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.world_matrix
    }

    /// 获取世界矩阵 (Mat4) - 供 Renderer 上传 GPU 使用
    #[inline]
    pub fn world_matrix_as_mat4(&self) -> Mat4 {
        Mat4::from(self.world_matrix)
    }

    /// 供 Scene 更新完矩阵后写入
    pub fn set_world_matrix(&mut self, mat: Affine3A) {
        self.world_matrix = mat;
    }

    /// 直接设置局部矩阵 (适用于 glTF 加载或物理引擎同步)
    /// 
    /// 注意：这会触发矩阵分解，反向更新 position/rotation/scale
    /// 警告：如果矩阵包含切变，分解过程会丢失切变信息
    pub fn apply_local_matrix(&mut self, mat: Affine3A) {
        // 1. 直接应用矩阵
        self.local_matrix = mat;

        // 2. 矩阵分解
        let (scale, rotation, translation) = mat.to_scale_rotation_translation();

        // 3. 更新 Public 属性
        self.scale = scale;
        self.rotation = rotation;
        self.position = translation;

        // 4. 同步影子状态
        self.last_scale = scale;
        self.last_rotation = rotation;
        self.last_position = translation;

        // 5. 标记脏
        self.mark_dirty();
    }

    /// 针对 Mat4 的辅助版本
    pub fn apply_local_matrix_from_mat4(&mut self, mat: Mat4) {
        let affine = Affine3A::from_mat4(mat);
        self.apply_local_matrix(affine);
    }

    /// LookAt 变换
    /// 
    /// `target` 和 `up` 应该处于该变换的父坐标系中。
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        // 1. 计算前向矢量
        let forward = (target - self.position).normalize();

        // 2. 检查退化情况
        if forward.cross(up).length_squared() < 1e-4 {
            return;
        }

        // 3. 构建正交基
        let right = forward.cross(up).normalize();
        let new_up = right.cross(forward).normalize(); 

        let rot_mat = Mat3::from_cols(
            right, 
            new_up, 
            -forward
        );
        self.rotation = Quat::from_mat3(&rot_mat);
    }
    
    /// 手动标记脏（例如用于强制刷新）
    pub fn mark_dirty(&mut self) {
        self.force_update = true;
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}
