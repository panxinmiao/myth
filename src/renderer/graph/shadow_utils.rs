//! Shadow Utilities
//!
//! Pure math functions for shadow mapping, extracted from render passes
//! for reuse and testability.
//!
//! # Provided Functions
//!
//! - Cascade split computation (Practical Split Scheme)
//! - Frustum corner extraction in world space
//! - Orthographic VP matrix construction for CSM cascades
//! - Perspective VP matrix construction for spot light shadows
//! - High-level view builders for directional and spot lights

use glam::{Mat4, Vec3};

use crate::renderer::core::view::RenderView;
use crate::scene::camera::{Frustum, RenderCamera};
use crate::scene::light::{ShadowConfig, SpotLight};

/// Maximum cascade count per directional light.
pub const MAX_CASCADES: u32 = 4;

// ============================================================================
// Cascade Split Computation
// ============================================================================

/// Computes cascade split distances using the Practical Split Scheme.
///
/// `lambda` blends between uniform (`0.0`) and logarithmic (`1.0`) distribution.
/// Returns an array of far distances for each cascade (in view space).
#[must_use]
pub fn compute_cascade_splits(
    cascade_count: u32,
    near: f32,
    far: f32,
    lambda: f32,
) -> [f32; MAX_CASCADES as usize] {
    let mut splits = [0.0f32; MAX_CASCADES as usize];
    let n = cascade_count.min(MAX_CASCADES) as usize;

    for i in 0..n {
        let p = (i + 1) as f32 / n as f32;
        let log_split = near * (far / near).powf(p);
        let uni_split = near + (far - near) * p;
        splits[i] = lambda * log_split + (1.0 - lambda) * uni_split;
    }

    // Ensure the last split reaches the far plane
    if n > 0 {
        splits[n - 1] = far;
    }

    splits
}

// ============================================================================
// Frustum Corners in World Space
// ============================================================================

/// Computes the 8 frustum corners of a view-space frustum slice in world space.
///
/// Uses the camera's projection matrix to extract FOV and aspect ratio,
/// then constructs corners for the given near/far slice in view space,
/// and transforms them to world space using the inverse view matrix.
#[must_use]
pub fn compute_frustum_corners_world(
    camera: &RenderCamera,
    slice_near: f32,
    slice_far: f32,
) -> [Vec3; 8] {
    let proj = camera.projection_matrix;
    let tan_half_fov = 1.0 / proj.y_axis.y;
    let aspect = proj.y_axis.y / proj.x_axis.x;

    let h_near = tan_half_fov * slice_near;
    let w_near = h_near * aspect;
    let h_far = tan_half_fov * slice_far;
    let w_far = h_far * aspect;

    // Corners in view space (RH: -Z is forward)
    let corners_view = [
        // Near face (z = -slice_near)
        Vec3::new(-w_near, -h_near, -slice_near),
        Vec3::new(w_near, -h_near, -slice_near),
        Vec3::new(w_near, h_near, -slice_near),
        Vec3::new(-w_near, h_near, -slice_near),
        // Far face (z = -slice_far)
        Vec3::new(-w_far, -h_far, -slice_far),
        Vec3::new(w_far, -h_far, -slice_far),
        Vec3::new(w_far, h_far, -slice_far),
        Vec3::new(-w_far, h_far, -slice_far),
    ];

    let inv_view = camera.view_matrix.inverse();
    let mut corners_world = [Vec3::ZERO; 8];
    for (i, c) in corners_view.iter().enumerate() {
        corners_world[i] = inv_view.transform_point3(*c);
    }
    corners_world
}

// ============================================================================
// CSM: Build Cascade VP Matrix
// ============================================================================

/// Builds an orthographic VP matrix for one CSM cascade.
///
/// Calculates the light-space AABB of the frustum slice,
/// applies texel alignment to prevent shimmer when the camera moves.
#[must_use]
pub fn build_cascade_vp(
    light_direction: Vec3,
    frustum_corners: &[Vec3; 8],
    shadow_map_size: u32,
    caster_extension: f32,
) -> Mat4 {
    let safe_dir = if light_direction.length_squared() > 1e-6 {
        light_direction.normalize()
    } else {
        -Vec3::Z
    };

    // Compute frustum center
    let mut center = Vec3::ZERO;
    for c in frustum_corners {
        center += *c;
    }
    center /= 8.0;

    let up = if safe_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let light_view = Mat4::look_at_rh(center - safe_dir, center, up);

    // Compute light-space AABB of frustum corners
    let mut ls_min = Vec3::splat(f32::MAX);
    let mut ls_max = Vec3::splat(f32::MIN);
    for c in frustum_corners {
        let ls = light_view.transform_point3(*c);
        ls_min = ls_min.min(ls);
        ls_max = ls_max.max(ls);
    }

    // Expand Z to include potential casters between camera and light.
    // In RH light view, ls_max.z is near (towards light), ls_min.z is far.
    let base_z_range = (ls_max.z - ls_min.z).max(1.0);
    let near_extension = caster_extension.max(base_z_range);
    let far_extension = base_z_range.max(50.0);
    ls_max.z += near_extension;
    ls_min.z -= far_extension;

    // Texel alignment: snap the ortho bounds to texel grid to prevent shimmer
    let world_units_per_texel_x = (ls_max.x - ls_min.x) / shadow_map_size as f32;
    let world_units_per_texel_y = (ls_max.y - ls_min.y) / shadow_map_size as f32;

    if world_units_per_texel_x > 0.0 {
        ls_min.x = (ls_min.x / world_units_per_texel_x).floor() * world_units_per_texel_x;
        ls_max.x = (ls_max.x / world_units_per_texel_x).ceil() * world_units_per_texel_x;
    }
    if world_units_per_texel_y > 0.0 {
        ls_min.y = (ls_min.y / world_units_per_texel_y).floor() * world_units_per_texel_y;
        ls_max.y = (ls_max.y / world_units_per_texel_y).ceil() * world_units_per_texel_y;
    }

    let proj = Mat4::orthographic_rh(
        ls_min.x, ls_max.x, ls_min.y, ls_max.y, -ls_max.z,
        -ls_min.z, // glam orthographic_rh: near/far are positive distances
    );

    proj * light_view
}

// ============================================================================
// Spot Light VP Matrix
// ============================================================================

/// Builds a perspective VP matrix for a spot light shadow.
#[must_use]
pub fn build_spot_vp(position: Vec3, direction: Vec3, spot: &SpotLight) -> Mat4 {
    let safe_dir = if direction.length_squared() > 1e-6 {
        direction.normalize()
    } else {
        -Vec3::Z
    };
    let up = if safe_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(position, position + safe_dir, up);
    let fov = (spot.outer_cone * 2.0).clamp(0.1, std::f32::consts::PI - 0.01);
    let far = spot.range.max(1.0);
    let proj = Mat4::perspective_rh(fov, 1.0, 0.1, far);
    proj * view
}

// ============================================================================
// High-level View Builders
// ============================================================================

/// Builds all cascade [`RenderView`]s for a single directional light.
///
/// # Arguments
///
/// - `light_id`: Unique ID of the light.
/// - `light_direction`: World-space direction the light is shining.
/// - `light_buffer_index`: Index into the GPU light storage buffer.
/// - `camera`: The main camera (used to extract frustum slices).
/// - `shadow_cfg`: Shadow configuration for this light.
/// - `shadow_far`: Effective shadow far distance (clamped to camera far).
/// - `caster_extension`: Maximum scene caster extent (for Z extension).
/// - `base_layer`: Starting layer index in the shadow map array.
///
/// Returns a `Vec<RenderView>` with one entry per cascade, plus the
/// computed split distances (indexed by cascade in `[f32; MAX_CASCADES]`).
#[must_use]
pub fn build_directional_views(
    light_id: u64,
    light_direction: Vec3,
    light_buffer_index: usize,
    camera: &RenderCamera,
    shadow_cfg: &ShadowConfig,
    shadow_far: f32,
    caster_extension: f32,
    base_layer: u32,
) -> (Vec<RenderView>, [f32; MAX_CASCADES as usize]) {
    let cascade_count = shadow_cfg.cascade_count.clamp(1, MAX_CASCADES);
    let cam_near = camera.near.max(0.1);

    let splits = compute_cascade_splits(
        cascade_count,
        cam_near,
        shadow_far,
        shadow_cfg.cascade_split_lambda,
    );

    let mut views = Vec::with_capacity(cascade_count as usize);
    let mut prev_split = cam_near;

    for c in 0..cascade_count as usize {
        let slice_near = prev_split;
        let slice_far = splits[c];
        prev_split = slice_far;

        let corners = compute_frustum_corners_world(camera, slice_near, slice_far);
        let vp = build_cascade_vp(
            light_direction,
            &corners,
            shadow_cfg.map_size,
            caster_extension,
        );

        // Build culling frustum from the cascade VP.
        // Uses standard-Z extraction since the ortho projection uses standard depth [0, 1].
        // The near plane is disabled (set to zero) so that shadow casters towards the light
        // direction are never culled — only the XY bounds and far plane limit the set.
        let frustum = Frustum::from_matrix_shadow_caster(vp);

        views.push(RenderView::new_shadow(
            light_id,
            base_layer + c as u32,
            light_buffer_index,
            format!("DirLight_{light_id}_Cascade_{c}"),
            vp,
            frustum,
            (shadow_cfg.map_size, shadow_cfg.map_size),
            Some(splits[c]),
        ));
    }

    (views, splits)
}

/// Builds a single [`RenderView`] for a spot light shadow.
#[must_use]
pub fn build_spot_view(
    light_id: u64,
    light_buffer_index: usize,
    position: Vec3,
    direction: Vec3,
    spot: &SpotLight,
    shadow_cfg: &ShadowConfig,
    base_layer: u32,
) -> RenderView {
    let vp = build_spot_vp(position, direction, spot);
    let frustum = Frustum::from_matrix_standard_z(vp);

    RenderView::new_shadow(
        light_id,
        base_layer,
        light_buffer_index,
        format!("SpotLight_{light_id}"),
        vp,
        frustum,
        (shadow_cfg.map_size, shadow_cfg.map_size),
        None,
    )
}
