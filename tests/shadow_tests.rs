//! Shadow Algorithm Tests
//!
//! Tests for:
//! - CSM cascade split computation (Practical Split Scheme)
//! - Frustum corners extraction in world space
//! - Cascade VP matrix construction correctness
//! - Texel alignment for shimmer prevention
//! - Spot light VP matrix construction

use glam::{Mat4, Vec3, Vec3A};

use myth::renderer::graph::shadow_utils::*;
use myth::scene::camera::{Frustum, RenderCamera};
use myth::scene::light::SpotLight;

const EPSILON: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ============================================================================
// compute_cascade_splits Tests
// ============================================================================

#[test]
fn cascade_splits_last_equals_far() {
    let splits = compute_cascade_splits(4, 0.1, 100.0, 0.5);
    assert!(
        approx(splits[3], 100.0),
        "Last split should equal far plane, got {}",
        splits[3]
    );
}

#[test]
fn cascade_splits_monotonically_increasing() {
    let splits = compute_cascade_splits(4, 0.1, 100.0, 0.5);
    for i in 1..4 {
        assert!(
            splits[i] > splits[i - 1],
            "Splits should be monotonically increasing: splits[{}]={} <= splits[{}]={}",
            i,
            splits[i],
            i - 1,
            splits[i - 1]
        );
    }
}

#[test]
fn cascade_splits_lambda_0_uniform() {
    let splits = compute_cascade_splits(4, 1.0, 100.0, 0.0);
    // lambda=0 → pure uniform distribution: 1 + (100-1) * (i/4)
    let expected = [25.75, 50.5, 75.25, 100.0];
    for i in 0..4 {
        assert!(
            approx(splits[i], expected[i]),
            "splits[{i}]: expected {}, got {}",
            expected[i],
            splits[i]
        );
    }
}

#[test]
fn cascade_splits_lambda_1_logarithmic() {
    let splits = compute_cascade_splits(4, 1.0, 100.0, 1.0);
    // lambda=1 → pure logarithmic: near * (far/near)^(i/n)
    // 1.0 * 100^(1/4) ≈ 3.162, 100^(2/4) = 10, 100^(3/4) ≈ 31.62, 100^(4/4) = 100
    assert!(
        (splits[0] - 3.162).abs() < 0.01,
        "Log split[0] ≈ 3.162, got {}",
        splits[0]
    );
    assert!(
        approx(splits[1], 10.0),
        "Log split[1] = 10, got {}",
        splits[1]
    );
    assert!(
        (splits[2] - 31.62).abs() < 0.1,
        "Log split[2] ≈ 31.62, got {}",
        splits[2]
    );
    assert!(
        approx(splits[3], 100.0),
        "Log split[3] = 100, got {}",
        splits[3]
    );
}

#[test]
fn cascade_splits_single_cascade() {
    let splits = compute_cascade_splits(1, 0.1, 50.0, 0.5);
    assert!(
        approx(splits[0], 50.0),
        "Single cascade should cover entire range, got {}",
        splits[0]
    );
}

#[test]
fn cascade_splits_clamped_to_max() {
    // cascade_count > MAX_CASCADES should be clamped
    let splits = compute_cascade_splits(10, 0.1, 100.0, 0.5);
    // Should only fill first MAX_CASCADES (4) entries
    assert!(
        approx(splits[3], 100.0),
        "Should clamp to 4 cascades, last = {}",
        splits[3]
    );
}

// ============================================================================
// compute_frustum_corners_world Tests
// ============================================================================

fn make_render_camera(
    pos: Vec3,
    look_dir: Vec3,
    fov_deg: f32,
    aspect: f32,
    near: f32,
) -> RenderCamera {
    let proj = Mat4::perspective_infinite_reverse_rh(fov_deg.to_radians(), aspect, near);
    let view = Mat4::look_at_rh(pos, pos + look_dir, Vec3::Y);
    RenderCamera {
        view_matrix: view,
        projection_matrix: proj,
        view_projection_matrix: proj * view,
        position: Vec3A::from(pos),
        frustum: Frustum::from_matrix(proj * view),
        near,
        far: f32::INFINITY,
    }
}

#[test]
fn frustum_corners_count() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let corners = compute_frustum_corners_world(&cam, 1.0, 10.0);
    assert_eq!(corners.len(), 8);
}

#[test]
fn frustum_corners_near_far_distances() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let near = 1.0;
    let far = 10.0;
    let corners = compute_frustum_corners_world(&cam, near, far);

    // Near face corners (indices 0-3) should be at Z ≈ -near
    for i in 0..4 {
        assert!(
            (corners[i].z - (-near)).abs() < 0.1,
            "Near corner {i} z: expected ≈{}, got {}",
            -near,
            corners[i].z
        );
    }

    // Far face corners (indices 4-7) should be at Z ≈ -far
    for i in 4..8 {
        assert!(
            (corners[i].z - (-far)).abs() < 0.1,
            "Far corner {i} z: expected ≈{}, got {}",
            -far,
            corners[i].z
        );
    }
}

#[test]
fn frustum_corners_symmetry() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let corners = compute_frustum_corners_world(&cam, 1.0, 10.0);

    // Near face: corners should be symmetric about Y axis
    // corner[0] = (-w, -h, -near), corner[1] = (w, -h, -near)
    assert!(
        approx(corners[0].x, -corners[1].x),
        "Near face should be X-symmetric"
    );
    // corner[2] = (w, h, -near), corner[3] = (-w, h, -near)
    assert!(
        approx(corners[2].x, -corners[3].x),
        "Near face should be X-symmetric"
    );
}

// ============================================================================
// build_cascade_vp Tests
// ============================================================================

#[test]
fn cascade_vp_is_valid_matrix() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let corners = compute_frustum_corners_world(&cam, 1.0, 50.0);

    let vp = build_cascade_vp(Vec3::new(0.0, -1.0, 0.0), &corners, 2048, 100.0);

    // VP matrix should not have NaN or Inf
    for i in 0..4 {
        for j in 0..4 {
            let val = vp.col(i)[j];
            assert!(!val.is_nan(), "VP contains NaN at [{i}][{j}]");
            assert!(!val.is_infinite(), "VP contains Inf at [{i}][{j}]");
        }
    }
}

#[test]
fn cascade_vp_determinant_nonzero() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let corners = compute_frustum_corners_world(&cam, 1.0, 50.0);
    let vp = build_cascade_vp(Vec3::new(0.0, -1.0, -0.5), &corners, 2048, 100.0);

    let det = vp.determinant();
    assert!(det.abs() > 1e-10, "VP should be invertible (det={})", det);
}

#[test]
fn cascade_vp_frustum_center_inside() {
    let cam = make_render_camera(Vec3::ZERO, Vec3::NEG_Z, 60.0, 1.0, 0.1);
    let corners = compute_frustum_corners_world(&cam, 1.0, 50.0);
    let vp = build_cascade_vp(Vec3::new(0.0, -1.0, 0.0), &corners, 2048, 100.0);

    // Compute frustum center
    let center: Vec3 = corners.iter().copied().sum::<Vec3>() / 8.0;

    // Transform center through VP
    let clip = vp * glam::Vec4::new(center.x, center.y, center.z, 1.0);
    let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);

    // Center should map to near NDC center (approximately within [-1,1] x [-1,1] x [0,1])
    assert!(
        ndc.x.abs() <= 1.5 && ndc.y.abs() <= 1.5,
        "Frustum center should project near NDC center, got {:?}",
        ndc
    );
}

// ============================================================================
// build_spot_vp Tests
// ============================================================================

#[test]
fn spot_vp_determinant_nonzero() {
    let spot = SpotLight {
        range: 20.0,
        inner_cone: 0.3,
        outer_cone: 0.5,
    };

    let vp = build_spot_vp(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), &spot);
    let det = vp.determinant();
    assert!(
        det.abs() > 1e-10,
        "Spot VP should be invertible (det={})",
        det
    );
}

#[test]
fn spot_vp_near_point_maps_correctly() {
    let spot = SpotLight {
        range: 50.0,
        inner_cone: 0.3,
        outer_cone: 0.5,
    };

    let position = Vec3::new(0.0, 10.0, 0.0);
    let direction = Vec3::NEG_Y;
    let vp = build_spot_vp(position, direction, &spot);

    // A point slightly in front of the light (by 0.1 = near distance)
    // should map to NDC z near 0 (standard-Z near plane)
    let near_point = position + direction * 0.1;
    let clip = vp * glam::Vec4::new(near_point.x, near_point.y, near_point.z, 1.0);
    let ndc_z = clip.z / clip.w;
    assert!(
        ndc_z.is_finite() && ndc_z.abs() < 0.1,
        "Near point should map near NDC z=0, got ndc_z={}",
        ndc_z
    );
}
