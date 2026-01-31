use glam::{Affine3A, Mat4, Quat, Vec3, Mat3, EulerRot};

/// Transform component for scene nodes.
///
/// Encapsulates position, rotation, and scale (TRS) along with
/// cached matrices and dirty-checking logic for efficient updates.
///
/// # Coordinate System
///
/// Uses a right-handed coordinate system:
/// - +X: Right
/// - +Y: Up
/// - +Z: Forward (out of screen)
///
/// # Matrix Updates
///
/// The transform maintains two cached matrices:
/// - `local_matrix`: Transform relative to parent
/// - `world_matrix`: Transform relative to world origin
///
/// These are updated lazily when TRS values change, tracked by
/// dirty checking against previous frame values.
///
/// # Example
///
/// ```rust,ignore
/// let mut transform = Transform::new();
/// transform.position = Vec3::new(1.0, 2.0, 3.0);
/// transform.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
/// transform.scale = Vec3::splat(2.0);
/// ```
#[derive(Debug, Clone)]
pub struct Transform {
    /// Local position relative to parent
    pub position: Vec3,
    /// Local rotation as quaternion
    pub rotation: Quat,
    /// Local scale factor
    pub scale: Vec3,

    // Cached matrices (read by renderer)
    pub(crate) local_matrix: Affine3A,
    pub(crate) world_matrix: Affine3A,

    // Dirty checking state
    last_position: Vec3,
    last_rotation: Quat,
    last_scale: Vec3,
    force_update: bool,
}

impl Transform {
    /// Creates a new transform with identity values.
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

    /// Checks and updates the local matrix if TRS values changed.
    ///
    /// Returns `true` if the matrix was updated, `false` otherwise.
    /// This method is called by the transform system during hierarchy traversal.
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

    /// Sets rotation from Euler angles (XYZ order).
    ///
    /// # Arguments
    ///
    /// * `x` - Rotation around X axis in radians
    /// * `y` - Rotation around Y axis in radians  
    /// * `z` - Rotation around Z axis in radians
    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) {
        self.rotation = Quat::from_euler(EulerRot::XYZ, x, y, z);
    }

    /// Returns the rotation as Euler angles (XYZ order) in radians.
    pub fn rotation_euler(&self) -> Vec3 {
        let (x, y, z) = self.rotation.to_euler(EulerRot::XYZ);
        Vec3::new(x, y, z)
    }

    /// Sets rotation from Euler angles with a custom rotation order.
    pub fn set_rotation_euler_with_order(&mut self, x: f32, y: f32, z: f32, order: EulerRot) {
        self.rotation = Quat::from_euler(order, x, y, z);
    }

    /// Returns a reference to the local transformation matrix.
    #[inline]
    pub fn local_matrix(&self) -> &Affine3A {
        &self.local_matrix
    }

    /// Returns a reference to the world transformation matrix.
    #[inline]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.world_matrix
    }

    /// Returns the world matrix as a 4x4 matrix.
    #[inline]
    pub fn world_matrix_as_mat4(&self) -> Mat4 {
        Mat4::from(self.world_matrix)
    }

    /// Sets the world matrix directly.
    pub fn set_world_matrix(&mut self, mat: Affine3A) {
        self.world_matrix = mat;
    }

    /// Sets the position and marks the transform as dirty.
    pub fn set_position(&mut self, pos: Vec3) {
        self.position = pos;
        self.mark_dirty();
    }

    /// Directly sets the local matrix (e.g., from glTF or physics engine).
    ///
    /// Decomposes the matrix into TRS and synchronizes state.
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

    /// Applies a local matrix from a Mat4 (converts to Affine3A first).
    pub fn apply_local_matrix_from_mat4(&mut self, mat: Mat4) {
        let affine = Affine3A::from_mat4(mat);
        self.apply_local_matrix(affine);
    }

    /// Orients the transform to face a target point in parent space.
    ///
    /// # Arguments
    ///
    /// * `target` - The point to look at in parent-local coordinates
    /// * `up` - The up vector (typically `Vec3::Y`)
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

    /// Marks the transform as needing update (forces matrix recalculation).
    pub fn mark_dirty(&mut self) {
        self.force_update = true;
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}
