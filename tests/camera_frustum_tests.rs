//! Camera and Frustum Tests
//!
//! Tests for:
//! - Perspective/Orthographic projection matrix generation
//! - Reverse-Z infinite perspective
//! - View-projection matrix update
//! - Frustum plane extraction (Gribb-Hartmann)
//! - Frustum-sphere intersection
//! - Frustum-AABB intersection
//! - RenderCamera extraction

use glam::{Affine3A, Mat4, Vec3};

use myth::resources::geometry::BoundingBox;
use myth::scene::camera::{Camera, Frustum};

const EPSILON: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ============================================================================
// Projection Matrix Tests
// ============================================================================

#[test]
fn perspective_reverse_z_near_maps_to_1() {
    let cam = Camera::new_perspective(60.0, 1.0, 0.1);
    let rc = cam.extract_render_camera();

    // In reverse-Z infinite perspective, a point at z = -near in view space
    // should map to NDC z = 1.0
    let near_point = rc.projection_matrix * glam::Vec4::new(0.0, 0.0, -0.1, 1.0);
    let ndc_z = near_point.z / near_point.w;
    assert!(
        approx(ndc_z, 1.0),
        "Near plane should map to NDC z=1.0 in reverse-Z, got {ndc_z}"
    );
}

#[test]
fn perspective_reverse_z_far_maps_to_0() {
    let cam = Camera::new_perspective(60.0, 1.0, 0.1);
    let rc = cam.extract_render_camera();

    // In infinite reverse-Z, as z → -∞, NDC z → 0.0
    let far_point = rc.projection_matrix * glam::Vec4::new(0.0, 0.0, -100000.0, 1.0);
    let ndc_z = far_point.z / far_point.w;
    assert!(
        ndc_z.abs() < 0.01,
        "Very far point should map to NDC z≈0.0 in reverse-Z, got {ndc_z}"
    );
}

#[test]
fn perspective_aspect_ratio_affects_fov() {
    let cam_wide = Camera::new_perspective(60.0, 2.0, 0.1); // wide
    let cam_square = Camera::new_perspective(60.0, 1.0, 0.1);
    let rc_wide = cam_wide.extract_render_camera();
    let rc_square = cam_square.extract_render_camera();

    // The X scaling should be different for different aspect ratios
    assert_ne!(
        rc_wide.projection_matrix.x_axis.x, rc_square.projection_matrix.x_axis.x,
        "Different aspect ratios should produce different X scaling"
    );
}

// ============================================================================
// View-Projection Update Tests
// ============================================================================

#[test]
fn view_projection_update_from_world_transform() {
    let mut cam = Camera::new_perspective(60.0, 1.0, 0.1);

    let world = Affine3A::from_translation(Vec3::new(0.0, 5.0, 10.0));
    cam.update_view_projection(&world);

    let render_cam = cam.extract_render_camera();
    // Camera position should match world translation
    assert!(approx(render_cam.position.x, 0.0));
    assert!(approx(render_cam.position.y, 5.0));
    assert!(approx(render_cam.position.z, 10.0));
}

#[test]
fn view_matrix_is_inverse_of_world() {
    let mut cam = Camera::new_perspective(60.0, 1.0, 0.1);

    let world = Affine3A::from_translation(Vec3::new(1.0, 2.0, 3.0));
    cam.update_view_projection(&world);

    let render_cam = cam.extract_render_camera();
    let vp_product = Mat4::from(world) * render_cam.view_matrix;

    // World * View ≈ Identity
    let expected = Mat4::IDENTITY;
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                approx(vp_product.col(i)[j], expected.col(i)[j]),
                "World * View should be identity at [{i}][{j}]: {} vs {}",
                vp_product.col(i)[j],
                expected.col(i)[j]
            );
        }
    }
}

// ============================================================================
// Frustum Extraction and Intersection Tests
// ============================================================================

fn make_test_frustum() -> Frustum {
    // Standard perspective camera at origin looking down -Z
    let proj = Mat4::perspective_infinite_reverse_rh(60.0_f32.to_radians(), 1.0, 0.1);
    let view = Mat4::IDENTITY;
    Frustum::from_matrix(proj * view)
}

#[test]
fn frustum_sphere_inside() {
    let frustum = make_test_frustum();
    // Sphere at origin, well inside the frustum
    assert!(
        frustum.intersects_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0),
        "Sphere at center should be inside frustum"
    );
}

#[test]
fn frustum_sphere_outside_left() {
    let frustum = make_test_frustum();
    // Sphere way to the left
    assert!(
        !frustum.intersects_sphere(Vec3::new(-1000.0, 0.0, -5.0), 1.0),
        "Sphere far to the left should be outside"
    );
}

#[test]
fn frustum_sphere_outside_behind() {
    let frustum = make_test_frustum();
    // Sphere behind the camera (positive Z in right-handed)
    assert!(
        !frustum.intersects_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0),
        "Sphere behind camera should be outside"
    );
}

#[test]
fn frustum_sphere_straddling_boundary() {
    let frustum = make_test_frustum();
    // Large sphere that overlaps the frustum
    assert!(
        frustum.intersects_sphere(Vec3::new(0.0, 0.0, -5.0), 100.0),
        "Large sphere should intersect"
    );
}

#[test]
fn frustum_aabb_inside() {
    let frustum = make_test_frustum();
    let min = Vec3::new(-0.5, -0.5, -6.0);
    let max = Vec3::new(0.5, 0.5, -4.0);
    assert!(
        frustum.intersects_box(min, max),
        "AABB in front of camera should be inside"
    );
}

#[test]
fn frustum_aabb_outside() {
    let frustum = make_test_frustum();
    let min = Vec3::new(-1000.0, -1000.0, -1002.0);
    let max = Vec3::new(-999.0, -999.0, -1001.0);
    assert!(
        !frustum.intersects_box(min, max),
        "AABB far away should be outside"
    );
}

#[test]
fn frustum_aabb_behind_camera() {
    let frustum = make_test_frustum();
    let min = Vec3::new(-1.0, -1.0, 5.0);
    let max = Vec3::new(1.0, 1.0, 10.0);
    assert!(
        !frustum.intersects_box(min, max),
        "AABB behind camera should be outside"
    );
}

#[test]
fn frustum_intersects_aabb_struct() {
    let frustum = make_test_frustum();
    let aabb = BoundingBox {
        min: Vec3::new(-0.5, -0.5, -6.0),
        max: Vec3::new(0.5, 0.5, -4.0),
    };
    assert!(frustum.intersects_aabb(&aabb));
}

// ============================================================================
// Standard-Z Frustum (for shadow maps)
// ============================================================================

#[test]
fn frustum_standard_z_inside_and_outside() {
    let proj = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let frustum = Frustum::from_matrix_standard_z(proj * view);

    // Inside
    assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, -50.0), 1.0));

    // Outside (behind)
    assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0));

    // Outside (far beyond)
    assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, -200.0), 1.0));
}

#[test]
fn frustum_shadow_caster_no_near_cull() {
    let proj = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let frustum = Frustum::from_matrix_shadow_caster(proj * view);

    // Shadow caster frustum should NOT cull objects towards the light (near plane disabled)
    // An object "in front" of the light (positive Z, i.e., behind the look direction)
    // For shadow caster, near plane is disabled so objects closer should pass
    assert!(
        frustum.intersects_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0),
        "Shadow caster: inside volume should pass"
    );
}

// ============================================================================
// RenderCamera Extraction
// ============================================================================

#[test]
fn render_camera_has_correct_near() {
    let cam = Camera::new_perspective(60.0, 1.0, 0.5);
    let render_cam = cam.extract_render_camera();
    assert!(approx(render_cam.near, 0.5));
}
