//! Resource Component Tests
//!
//! Tests for:
//! - SssRegistry: allocator with fixed 256 slots, free list recycling, version tracking
//! - ChangeTracker: version increment, MutGuard auto-version-on-drop
//! - TextureSlot: compute_matrix for UV transforms (identity, rotation, scale, offset)
//! - Mat3Padded / Mat3Uniform: GPU alignment, construction helpers
//! - FpsCounter: frame counting, 1-second update cycle

use glam::{Mat4, Vec2, Vec3, Vec4};

use myth::resources::screen_space::{FeatureId, SssProfile, SssProfileData, SssRegistry};
use myth::resources::uniforms::{Mat3Padded, Mat3Uniform};
use myth::resources::version_tracker::{ChangeTracker, MutGuard};
use myth::resources::{TextureSlot, TextureTransform};
use myth::utils::FpsCounter;

const EPSILON: f32 = 1e-5;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ============================================================================
// SssRegistry Tests
// ============================================================================

#[test]
fn sss_registry_new_starts_at_version_1() {
    let reg = SssRegistry::new();
    assert_eq!(reg.version, 1);
}

#[test]
fn sss_registry_add_returns_id_and_increments_version() {
    let mut reg = SssRegistry::new();
    let profile = SssProfile::new(Vec3::new(1.0, 0.5, 0.2), 0.1);

    let id = reg.add(&profile);
    assert!(id.is_some());
    assert_eq!(reg.version, 2);
}

#[test]
fn sss_registry_add_stores_correct_data() {
    let mut reg = SssRegistry::new();
    let profile = SssProfile::new(Vec3::new(1.0, 0.5, 0.2), 0.3);

    let id = reg.add(&profile).unwrap();
    let data = reg.buffer_data[id.to_u32() as usize];
    assert!(approx(data.scatter_color[0], 1.0));
    assert!(approx(data.scatter_color[1], 0.5));
    assert!(approx(data.scatter_color[2], 0.2));
    assert!(approx(data.scatter_radius, 0.3));
}

#[test]
fn sss_registry_update_changes_data_and_version() {
    let mut reg = SssRegistry::new();
    let profile1 = SssProfile::new(Vec3::ONE, 1.0);
    let id = reg.add(&profile1).unwrap();
    let v_after_add = reg.version;

    let profile2 = SssProfile::new(Vec3::new(0.5, 0.5, 0.5), 2.0);
    reg.update(id, &profile2);

    assert!(
        reg.version > v_after_add,
        "Version should increment on update"
    );
    let data = reg.buffer_data[id.to_u32() as usize];
    assert!(approx(data.scatter_radius, 2.0));
}

#[test]
fn sss_registry_remove_recycles_id() {
    let mut reg = SssRegistry::new();
    let profile = SssProfile::new(Vec3::ONE, 1.0);

    let id1 = reg.add(&profile).unwrap();
    reg.remove(id1);

    // After removal, the ID should be recycled and re-assignable
    let id2 = reg.add(&profile).unwrap();
    assert_eq!(id1.to_u32(), id2.to_u32(), "Should reuse the recycled ID");
}

#[test]
fn sss_registry_remove_clears_data() {
    let mut reg = SssRegistry::new();
    let profile = SssProfile::new(Vec3::new(1.0, 0.5, 0.2), 0.5);
    let id = reg.add(&profile).unwrap();

    reg.remove(id);
    let data = reg.buffer_data[id.to_u32() as usize];
    assert_eq!(
        data,
        SssProfileData::default(),
        "Data should be zeroed after removal"
    );
}

#[test]
fn sss_registry_allocates_up_to_255() {
    let mut reg = SssRegistry::new();
    let profile = SssProfile::new(Vec3::ONE, 1.0);

    let mut ids = Vec::new();
    for _ in 0..255 {
        let id = reg.add(&profile);
        assert!(id.is_some(), "Should be able to allocate 255 profiles");
        ids.push(id.unwrap());
    }

    // 256th should fail (slot 0 is reserved)
    let overflow = reg.add(&profile);
    assert!(overflow.is_none(), "Should return None when full");
}

#[test]
fn sss_feature_id_round_trip() {
    let id = FeatureId::from_u32(42).unwrap();
    assert_eq!(id.to_u32(), 42);

    assert!(FeatureId::from_u32(0).is_none(), "0 maps to None");
}

// ============================================================================
// ChangeTracker / MutGuard Tests
// ============================================================================

#[test]
fn change_tracker_starts_at_zero() {
    let ct = ChangeTracker::new();
    assert_eq!(ct.version(), 0);
}

#[test]
fn change_tracker_increments_on_changed() {
    let mut ct = ChangeTracker::new();
    ct.changed();
    assert_eq!(ct.version(), 1);
    ct.changed();
    assert_eq!(ct.version(), 2);
}

#[test]
fn mut_guard_increments_version_on_drop() {
    let mut data = 42u32;
    let mut version = 0u64;

    {
        let mut guard = MutGuard::new(&mut data, &mut version);
        *guard = 100;
        // version still 0 here
    }
    // After guard drop, version should be 1
    assert_eq!(version, 1);
    assert_eq!(data, 100);
}

#[test]
fn mut_guard_deref_reads_data() {
    let mut data = String::from("hello");
    let mut version = 0u64;

    {
        let guard = MutGuard::new(&mut data, &mut version);
        assert_eq!(&*guard, "hello");
    }
}

// ============================================================================
// TextureTransform / TextureSlot compute_matrix Tests
// ============================================================================

#[test]
fn texture_transform_default_is_identity() {
    let t = TextureTransform::default();
    assert_eq!(t.offset, Vec2::ZERO);
    assert_eq!(t.rotation, 0.0);
    assert_eq!(t.scale, Vec2::ONE);
}

#[test]
fn texture_slot_compute_matrix_identity() {
    let slot = TextureSlot {
        texture: None,
        transform: TextureTransform::default(),
        channel: 0,
    };
    let m = slot.compute_matrix();
    let expected = Mat3Uniform::IDENTITY;
    // Check each column
    assert_eq!(m.col0, expected.col0, "Column 0 should be identity");
    assert_eq!(m.col1, expected.col1, "Column 1 should be identity");
    assert_eq!(m.col2, expected.col2, "Column 2 should be identity");
}

#[test]
fn texture_slot_compute_matrix_scale() {
    let slot = TextureSlot {
        texture: None,
        transform: TextureTransform {
            offset: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::new(2.0, 3.0),
        },
        channel: 0,
    };
    let m = slot.compute_matrix();
    // col0 = (sx*cos0, sx*sin0, 0) = (2, 0, 0)
    // col1 = (-sy*sin0, sy*cos0, 0) = (0, 3, 0)
    // col2 = (tx, ty, 1) = (0, 0, 1)
    assert!(approx(m.col0.x, 2.0));
    assert!(approx(m.col1.y, 3.0));
    assert!(approx(m.col2.z, 1.0));
}

#[test]
fn texture_slot_compute_matrix_offset() {
    let slot = TextureSlot {
        texture: None,
        transform: TextureTransform {
            offset: Vec2::new(0.5, 0.25),
            rotation: 0.0,
            scale: Vec2::ONE,
        },
        channel: 0,
    };
    let m = slot.compute_matrix();
    // Translation is in col2
    assert!(approx(m.col2.x, 0.5));
    assert!(approx(m.col2.y, 0.25));
}

#[test]
fn texture_slot_compute_matrix_rotation_90() {
    let slot = TextureSlot {
        texture: None,
        transform: TextureTransform {
            offset: Vec2::ZERO,
            rotation: std::f32::consts::FRAC_PI_2, // 90° CCW
            scale: Vec2::ONE,
        },
        channel: 0,
    };
    let m = slot.compute_matrix();
    // rotation is applied as -rotation in the code, so -90° = cos(-90°)=0, sin(-90°)=-1
    // col0 = (sx*c, sx*s, 0) = (cos(-90°), sin(-90°), 0) = (0, -1, 0)
    // col1 = (-sy*s, sy*c, 0) = (-sin(-90°), cos(-90°), 0) = (1, 0, 0)
    assert!(approx(m.col0.x, 0.0));
    assert!(approx(m.col0.y, -1.0));
    assert!(approx(m.col1.x, 1.0));
    assert!(approx(m.col1.y, 0.0));
}

// ============================================================================
// Mat3Padded / Mat3Uniform Tests
// ============================================================================

#[test]
fn mat3_padded_identity() {
    let m = Mat3Padded::IDENTITY;
    assert_eq!(m.col0, Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(m.col1, Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(m.col2, Vec4::new(0.0, 0.0, 1.0, 0.0));
}

#[test]
fn mat3_padded_size_is_48_bytes() {
    assert_eq!(std::mem::size_of::<Mat3Padded>(), 48);
}

#[test]
fn mat3_padded_from_cols_array() {
    let m = Mat3Padded::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(m.col0, Vec4::new(1.0, 2.0, 3.0, 0.0));
    assert_eq!(m.col1, Vec4::new(4.0, 5.0, 6.0, 0.0));
    assert_eq!(m.col2, Vec4::new(7.0, 8.0, 9.0, 0.0));
}

#[test]
fn mat3_padded_from_mat4_extracts_upper_left_3x3() {
    let m4 = Mat4::from_cols(
        Vec4::new(1.0, 2.0, 3.0, 0.0),
        Vec4::new(4.0, 5.0, 6.0, 0.0),
        Vec4::new(7.0, 8.0, 9.0, 0.0),
        Vec4::new(10.0, 11.0, 12.0, 1.0),
    );
    let m3 = Mat3Padded::from_mat4(m4);
    assert_eq!(m3.col0, Vec4::new(1.0, 2.0, 3.0, 0.0));
    assert_eq!(m3.col1, Vec4::new(4.0, 5.0, 6.0, 0.0));
    assert_eq!(m3.col2, Vec4::new(7.0, 8.0, 9.0, 0.0));
}

#[test]
fn mat3_padded_from_cols() {
    let m = Mat3Padded::from_cols(Vec3::X, Vec3::Y, Vec3::Z);
    assert_eq!(m, Mat3Padded::IDENTITY);
}

// ============================================================================
// FpsCounter Tests
// ============================================================================

#[test]
fn fps_counter_initial_state() {
    let fps = FpsCounter::new();
    assert_eq!(fps.current_fps, 0.0);
}

#[test]
fn fps_counter_single_frame_returns_none() {
    // Calling update once should not trigger a 1-second report
    let mut fps = FpsCounter::new();
    let result = fps.update();
    // It probably hasn't been 1 second yet
    assert!(
        result.is_none(),
        "Single frame update should not report FPS"
    );
}
