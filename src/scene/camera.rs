use glam::{Affine3A, Mat4, Vec3, Vec3A, Vec4};
use std::borrow::Cow;
use uuid::Uuid;

use crate::resources::BoundingBox;

/// [New] Pure stack-based render camera object (POD)
/// TODO: Consider directly satisfying std140 alignment requirements?
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RenderCamera {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub view_projection_matrix: Mat4,
    pub position: Vec3A,  // World position, needed for lighting
    pub frustum: Frustum, // Needed for culling
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub uuid: Uuid,
    pub name: Cow<'static, str>,

    // === Projection Properties (Projection Only) ===
    pub projection_type: ProjectionType,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub ortho_size: f32,

    // Cached matrices (read-only for renderer)
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
    #[must_use]
    pub fn new_perspective(fov_degrees: f32, aspect: f32, near: f32) -> Self {
        let mut cam = Self {
            uuid: Uuid::new_v4(),
            name: Cow::Owned("Camera".to_string()),
            projection_type: ProjectionType::Perspective,
            fov: fov_degrees.to_radians(),
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

    pub fn update_projection_matrix(&mut self) {
        self.projection_matrix = match self.projection_type {
            ProjectionType::Perspective => {
                // glam's perspective_rh is designed for WGPU/Vulkan (0 to 1) by default
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

    #[must_use]
    pub fn extract_render_camera(&self) -> RenderCamera {
        RenderCamera {
            view_matrix: self.view_matrix,
            projection_matrix: self.projection_matrix,
            view_projection_matrix: self.view_projection_matrix,
            // Extract position from world matrix (Translation)
            position: self.world_matrix.translation,
            frustum: self.frustum, // Frustum is also Copy
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
    /// Extract frustum planes from a **reverse-Z, infinite-far** VP matrix.
    ///
    /// This is the default for the main camera (using `perspective_infinite_reverse_rh`).
    /// Far plane is set to zero (disabled) since the camera has infinite far.
    #[must_use]
    pub fn from_matrix(m: Mat4) -> Self {
        let rows = [m.row(0), m.row(1), m.row(2), m.row(3)];

        let mut planes = [Vec4::ZERO; 6];
        // Extraction formula: https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
        // Gribb-Hartmann method

        // Left:   row4 + row1
        planes[0] = rows[3] + rows[0];
        // Right:  row4 - row1
        planes[1] = rows[3] - rows[0];
        // Bottom: row4 + row2
        planes[2] = rows[3] + rows[1];
        // Top:    row4 - row2
        planes[3] = rows[3] - rows[1];

        // [Reverse-Z] Near Plane corresponds to NDC z = 1.0
        // Cull condition: z_c / w_c > 1.0 (closer than near plane)
        // Keep condition: z_c <= w_c => w_c - z_c >= 0
        planes[4] = rows[3] - rows[2]; // Near

        // Infinite far — disabled
        planes[5] = Vec4::ZERO;

        Self::normalize_planes(&mut planes);
        Self { planes }
    }

    /// Extract frustum planes from a **standard-Z [0, 1]** VP matrix.
    ///
    /// Use this for shadow projection matrices (both orthographic and perspective)
    /// where the depth range is standard (near → 0, far → 1).
    /// Both near and far planes are active.
    #[must_use]
    pub fn from_matrix_standard_z(m: Mat4) -> Self {
        let rows = [m.row(0), m.row(1), m.row(2), m.row(3)];

        let mut planes = [Vec4::ZERO; 6];

        // Left/Right/Bottom/Top: identical to reverse-Z
        planes[0] = rows[3] + rows[0];
        planes[1] = rows[3] - rows[0];
        planes[2] = rows[3] + rows[1];
        planes[3] = rows[3] - rows[1];

        // Standard Z [0, 1]:
        // Near:  z_ndc >= 0 → z_c >= 0 → row3
        // Far:   z_ndc <= 1 → w_c - z_c >= 0 → row4 - row3
        planes[4] = rows[2];
        planes[5] = rows[3] - rows[2];

        Self::normalize_planes(&mut planes);
        Self { planes }
    }

    /// Extract frustum planes for shadow caster culling from a **standard-Z** VP matrix.
    ///
    /// Like [`Self::from_matrix_standard_z`] but disables the near plane so that
    /// shadow casters towards the light source are never clipped.
    /// The Left/Right/Bottom/Top/Far planes still provide tight XY and depth culling.
    #[must_use]
    pub fn from_matrix_shadow_caster(m: Mat4) -> Self {
        let mut f = Self::from_matrix_standard_z(m);
        // Disable near plane — include all casters towards the light
        f.planes[4] = Vec4::ZERO;
        f
    }

    /// Normalize all planes, setting degenerate planes to zero.
    fn normalize_planes(planes: &mut [Vec4; 6]) {
        for plane in planes.iter_mut() {
            let length = Vec3::new(plane.x, plane.y, plane.z).length();
            if length > 1e-6 {
                *plane /= length;
            } else {
                *plane = Vec4::ZERO;
            }
        }
    }

    // Simple sphere intersection test
    #[must_use]
    #[inline]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            // Zero-normal planes are disabled (e.g. infinite far, or disabled near for shadow casters)
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

    /// AABB vs frustum intersection test
    /// Uses plane-AABB test, returns false if AABB is completely outside any plane
    #[must_use]
    #[inline]
    pub fn intersects_box(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Zero-normal planes are disabled (e.g. infinite far, or disabled near for shadow casters)
            if plane.x == 0.0 && plane.y == 0.0 && plane.z == 0.0 {
                continue;
            }

            // Find the point on AABB closest to the plane (p-vertex)
            // If this point is outside the plane, the entire AABB is outside
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

    /// AABB vs frustum intersection test
    #[must_use]
    #[inline]
    pub fn intersects_aabb(&self, aabb: &BoundingBox) -> bool {
        self.intersects_box(aabb.min, aabb.max)
    }
}
