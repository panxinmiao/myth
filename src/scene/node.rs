use glam::{Affine3A, Mat4, Quat, Vec3, Mat3, EulerRot};
use uuid::Uuid;
use thunderdome::Index;

#[derive(Debug, Clone)]
pub struct Node {
    // === Public 属性 (API 友好，直接修改) ===
    pub id: Uuid,
    pub name: String,
    
    // === 场景图结构 ===
    pub parent: Option<Index>,
    pub children: Vec<Index>,

    // === 组件关联 ===
    pub mesh: Option<Index>,   // 指向 meshes Arena 中的索引
    pub camera: Option<Index>, // 指向 cameras Arena 中的索引
    pub light: Option<Index>,  // 指向 lights Arena 中的索引

    // 变换属性 (TRS)
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    
    pub visible: bool,

    // === 私有状态 (Private) ===
    
    // 1. 矩阵缓存 (使用 Affine3A 进行极致优化)
    // 内存中占 48字节 (3x4 float)，运算跳过最后一行
    local_matrix: Affine3A,
    world_matrix: Affine3A,

    // 2. 影子状态 (Shadow State)
    // 用于检测 pub 属性是否被用户修改过
    last_position: Vec3,
    last_rotation: Quat,
    last_scale: Vec3,
    
    // 强制更新标记
    force_update: bool,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            parent: None,
            children: Vec::new(),

            mesh: None,
            camera: None,
            light: None,
            
            // 默认 TRS
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            
            visible: true,
            
            // 默认矩阵 (Identity)
            local_matrix: Affine3A::IDENTITY,
            world_matrix: Affine3A::IDENTITY,
            
            // 影子状态初始值
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
        // glam 类型支持快速的 SIMD 比较
        let changed = self.position != self.last_position 
                   || self.rotation != self.last_rotation 
                   || self.scale != self.last_scale
                   || self.force_update;

        if changed {
            // 2. 只有变了才重算矩阵 (Affine3A 构造比 Mat4 快)
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

    /// 这是一个 Helper，本质上是在修改 self.rotation
    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) {
        self.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }

    /// 获取当前的欧拉角 (XYZ 顺序)
    pub fn rotation_euler(&self) -> Vec3 {
        let (x, y, z) = self.rotation.to_euler(EulerRot::XYZ);
        Vec3::new(x, y, z)
    }

    /// 高级：支持指定旋转顺序 (例如相机有时用 YXZ)
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

    /// 获取世界矩阵 (Mat4) - 【关键】供 Renderer 上传 GPU 使用
    /// 因为 Shader 需要完整的 4x4 矩阵，这里做一次转换
    #[inline]
    pub fn world_matrix_as_mat4(&self) -> Mat4 {
        Mat4::from(self.world_matrix)
    }

    /// 供 Scene 更新完矩阵后写入
    pub fn set_world_matrix(&mut self, mat: Affine3A) {
        self.world_matrix = mat;
    }

    /// 直接设置局部矩阵 (适用于 glTF 加载或物理引擎同步)
    /// 注意：这会触发矩阵分解 (Decomposition)，反向更新 position/rotation/scale
    /// 警告：如果矩阵包含切变 (Shear)，分解过程会丢失切变信息，因为 TRS 无法表示切变。
    pub fn apply_local_matrix(&mut self, mat: Affine3A) {
        // 1. 直接应用矩阵
        self.local_matrix = mat;

        // 2. 矩阵分解 (Decompose) -> T, R, S
        // glam 的这一步极其高效
        let (scale, rotation, translation) = mat.to_scale_rotation_translation();

        // 3. 更新 Public 属性 (保持单一事实来源)
        self.scale = scale;
        self.rotation = rotation;
        self.position = translation;

        // 4. 【关键】同步影子状态 (Shadow State Sync)
        // 告诉引擎："这就是最新的状态，不需要在 update_local_matrix 里再算一遍了"
        self.last_scale = scale;
        self.last_rotation = rotation;
        self.last_position = translation;

        // 5. 标记脏
        // 虽然 local_matrix 已经是新的了，但我们需要通知 Scene 更新 world_matrix
        self.mark_dirty();
    }

    /// 针对 Mat4 的辅助版本 (方便 glTF 使用)
    pub fn apply_local_matrix_from_mat4(&mut self, mat: Mat4) {
        // 尝试转换为 Affine3A (丢弃最后一行)
        let affine = Affine3A::from_mat4(mat);
        self.apply_local_matrix(affine);
    }

    /// `target` 和 `up` 应该处于该节点的父节点坐标系中（如果没有父节点，则是世界坐标系）。
    /// 对于绝大多数场景根节点下的相机，这就是世界坐标。
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        // 1. 计算前向矢量 (从自身指向目标)
        let forward = (target - self.position).normalize();

        // 2. 检查退化情况 (Forward 平行于 Up)
        // 如果平行，cross 结果为零向量，会导致 NaN。这里做一个简单保护。
        if forward.cross(up).length_squared() < 1e-4 {
            return;
        }

        // 3. 构建正交基 (Gram-Schmidt process)
        // Right = Forward x Up
        let right = forward.cross(up).normalize();
        // Up' = Right x Forward
        let new_up = right.cross(forward).normalize(); 

        let rot_mat = Mat3::from_cols(
            right, 
            new_up, 
            -forward
        );
        self.rotation = Quat::from_mat3(&rot_mat);
    }
    
    /// 手动标记脏 (例如用于强制刷新)
    pub fn mark_dirty(&mut self) {
        self.force_update = true;
    }
}