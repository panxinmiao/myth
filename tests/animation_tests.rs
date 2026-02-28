//! Animation System Tests
//!
//! Tests for:
//! - KeyframeTrack linear/step/cubic interpolation
//! - Interpolatable trait implementations (f32, Vec3, Quat, MorphWeightData)
//! - KeyframeCursor O(1) optimization and binary search fallback
//! - AnimationAction loop modes (Once, Loop, PingPong)
//! - AnimationClip duration auto-computation

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use glam::{Quat, Vec3};

use myth::animation::action::{AnimationAction, LoopMode};
use myth::animation::binding::TargetPath;
use myth::animation::clip::{AnimationClip, Track, TrackData, TrackMeta};
use myth::animation::tracks::{InterpolationMode, KeyframeCursor, KeyframeTrack};
use myth::animation::values::{Interpolatable, MorphWeightData};

const EPSILON: f32 = 1e-5;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ============================================================================
// KeyframeTrack: Linear Interpolation (f32)
// ============================================================================

#[test]
fn track_linear_f32_midpoint() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![0.0_f32, 10.0],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(0.5, &mut cursor);
    assert!(approx(val, 5.0), "Expected 5.0, got {val}");
}

#[test]
fn track_linear_f32_exact_keyframe() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0],
        vec![0.0_f32, 10.0, 20.0],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();
    assert!(approx(track.sample_with_cursor(0.0, &mut cursor), 0.0));
    assert!(approx(track.sample_with_cursor(1.0, &mut cursor), 10.0));
    assert!(approx(track.sample_with_cursor(2.0, &mut cursor), 20.0));
}

#[test]
fn track_linear_f32_clamp_beyond_range() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![0.0_f32, 10.0],
        InterpolationMode::Linear,
    );

    // Sampling beyond the last keyframe should clamp to last value
    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(5.0, &mut cursor);
    assert!(approx(val, 10.0), "Expected 10.0, got {val}");
}

#[test]
fn track_linear_f32_before_first() {
    let track = KeyframeTrack::new(
        vec![1.0, 2.0],
        vec![10.0_f32, 20.0],
        InterpolationMode::Linear,
    );

    // Before first keyframe: should clamp to first value
    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(0.5, &mut cursor);
    assert!(approx(val, 10.0), "Expected 10.0, got {val}");
}

// ============================================================================
// KeyframeTrack: Step Interpolation
// ============================================================================

#[test]
fn track_step_holds_value() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0],
        vec![0.0_f32, 100.0, 200.0],
        InterpolationMode::Step,
    );

    // Step should hold the current keyframe value
    let mut cursor = KeyframeCursor::default();
    assert!(approx(track.sample_with_cursor(0.0, &mut cursor), 0.0));
    assert!(approx(track.sample_with_cursor(0.5, &mut cursor), 0.0));
    assert!(approx(track.sample_with_cursor(0.99, &mut cursor), 0.0));
    assert!(approx(track.sample_with_cursor(1.0, &mut cursor), 100.0));
    assert!(approx(track.sample_with_cursor(1.5, &mut cursor), 100.0));
}

// ============================================================================
// KeyframeTrack: Linear Interpolation (Vec3)
// ============================================================================

#[test]
fn track_linear_vec3() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![Vec3::ZERO, Vec3::new(10.0, 20.0, 30.0)],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(0.5, &mut cursor);
    assert!(approx(val.x, 5.0));
    assert!(approx(val.y, 10.0));
    assert!(approx(val.z, 15.0));
}

// ============================================================================
// KeyframeTrack: Linear Interpolation (Quat - slerp)
// ============================================================================

#[test]
fn track_linear_quat_slerp() {
    let q0 = Quat::IDENTITY;
    let q1 = Quat::from_rotation_y(PI);

    let track = KeyframeTrack::new(vec![0.0, 1.0], vec![q0, q1], InterpolationMode::Linear);

    // At t=0.5, should be halfway rotation
    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(0.5, &mut cursor);
    let expected = q0.slerp(q1, 0.5);
    let angle = val.angle_between(expected);
    assert!(angle < 0.01, "Quaternion slerp mismatch: angle={angle}");
}

// ============================================================================
// KeyframeTrack: Cubic Spline Interpolation
// ============================================================================

#[test]
fn track_cubic_f32_endpoints() {
    // CubicSpline: values = [in_tangent0, value0, out_tangent0, in_tangent1, value1, out_tangent1]
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![
            0.0_f32, 0.0, 1.0, // frame 0: in_tangent=0, value=0, out_tangent=1
            1.0, 10.0, 0.0, // frame 1: in_tangent=1, value=10, out_tangent=0
        ],
        InterpolationMode::CubicSpline,
    );

    // At exact keyframes, should return exact value
    let mut cursor = KeyframeCursor::default();
    let v0 = track.sample_with_cursor(0.0, &mut cursor);
    assert!(approx(v0, 0.0), "got {}", v0);
    let v1 = track.sample_with_cursor(1.0, &mut cursor);
    assert!(approx(v1, 10.0), "got {}", v1);
}

#[test]
fn track_cubic_f32_smooth_midpoint() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![
            0.0_f32, 0.0, 0.0, // frame 0: zero tangents, value=0
            0.0, 10.0, 0.0, // frame 1: zero tangents, value=10
        ],
        InterpolationMode::CubicSpline,
    );

    // With zero tangents, Hermite interpolation midpoint should be approximately 5.0
    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(0.5, &mut cursor);
    assert!(
        (val - 5.0).abs() < 1.0,
        "Cubic midpoint expected ~5.0, got {val}"
    );
}

// ============================================================================
// KeyframeTrack::sample() (stateless, no cursor)
// ============================================================================

#[test]
fn sample_linear_f32_midpoint() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![0.0_f32, 10.0],
        InterpolationMode::Linear,
    );
    assert!(approx(track.sample(0.5), 5.0), "got {}", track.sample(0.5));
}

#[test]
fn sample_linear_f32_exact_keyframes() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0],
        vec![0.0_f32, 10.0, 20.0],
        InterpolationMode::Linear,
    );
    assert!(
        approx(track.sample(0.0), 0.0),
        "t=0: got {}",
        track.sample(0.0)
    );
    assert!(
        approx(track.sample(1.0), 10.0),
        "t=1: got {}",
        track.sample(1.0)
    );
    assert!(
        approx(track.sample(2.0), 20.0),
        "t=2: got {}",
        track.sample(2.0)
    );
}

#[test]
fn sample_linear_f32_clamp_beyond() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![0.0_f32, 10.0],
        InterpolationMode::Linear,
    );
    assert!(approx(track.sample(5.0), 10.0));
}

#[test]
fn sample_linear_f32_before_first() {
    let track = KeyframeTrack::new(
        vec![1.0, 2.0],
        vec![10.0_f32, 20.0],
        InterpolationMode::Linear,
    );
    // Before first keyframe, t is clamped to 0 inside sample_at_frame
    assert!(approx(track.sample(0.0), 10.0), "got {}", track.sample(0.0));
}

#[test]
fn sample_step_holds_value() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0],
        vec![0.0_f32, 100.0, 200.0],
        InterpolationMode::Step,
    );
    assert!(approx(track.sample(0.0), 0.0));
    assert!(approx(track.sample(0.5), 0.0));
    assert!(approx(track.sample(1.0), 100.0));
    assert!(approx(track.sample(1.5), 100.0));
    assert!(approx(track.sample(2.0), 200.0));
}

#[test]
fn sample_linear_vec3() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![Vec3::ZERO, Vec3::new(10.0, 20.0, 30.0)],
        InterpolationMode::Linear,
    );
    let val = track.sample(0.5);
    assert!(approx(val.x, 5.0));
    assert!(approx(val.y, 10.0));
    assert!(approx(val.z, 15.0));
}

#[test]
fn sample_cubic_f32_endpoints() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0],
        vec![
            0.0_f32, 0.0, 1.0, // frame 0: in_tangent=0, value=0, out_tangent=1
            1.0, 10.0, 0.0, // frame 1: in_tangent=1, value=10, out_tangent=0
        ],
        InterpolationMode::CubicSpline,
    );
    assert!(
        approx(track.sample(0.0), 0.0),
        "t=0: got {}",
        track.sample(0.0)
    );
    assert!(
        approx(track.sample(1.0), 10.0),
        "t=1: got {}",
        track.sample(1.0)
    );
}

#[test]
fn sample_matches_cursor_across_all_times() {
    // Verify sample() and sample_with_cursor() produce identical results
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0_f32, 10.0, 5.0, 20.0, 15.0],
        InterpolationMode::Linear,
    );
    for i in 0..=40 {
        let t = i as f32 * 0.1;
        let mut cursor = KeyframeCursor::default();
        let val_cursor = track.sample_with_cursor(t, &mut cursor);
        let val_sample = track.sample(t);
        assert!(
            approx(val_sample, val_cursor),
            "t={t}: sample()={val_sample} != sample_with_cursor()={val_cursor}"
        );
    }
}

// ============================================================================
// KeyframeCursor: O(1) Sequential Access
// ============================================================================

#[test]
fn cursor_sequential_forward() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0_f32, 10.0, 20.0, 30.0, 40.0],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();

    // Sequential forward sampling should use O(1) cursor optimization
    for i in 0..=20 {
        let t = i as f32 * 0.2;
        let val = track.sample_with_cursor(t, &mut cursor);
        let expected = t * 10.0;
        assert!(
            approx(val, expected),
            "t={t}: expected {expected}, got {val}"
        );
    }
}

#[test]
fn cursor_forward_then_jump_back() {
    let track = KeyframeTrack::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0_f32, 10.0, 20.0, 30.0],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();

    // Move forward to t=2.5
    let val = track.sample_with_cursor(2.5, &mut cursor);
    assert!(approx(val, 25.0));

    // Jump back to t=0.5 (large jump → binary search fallback)
    let val = track.sample_with_cursor(0.5, &mut cursor);
    assert!(approx(val, 5.0));
}

#[test]
fn cursor_single_keyframe() {
    let track = KeyframeTrack::new(vec![0.0], vec![42.0_f32], InterpolationMode::Linear);

    let mut cursor = KeyframeCursor::default();
    let val = track.sample_with_cursor(5.0, &mut cursor);
    assert!(approx(val, 42.0));
}

#[test]
fn cursor_two_keyframes() {
    let track = KeyframeTrack::new(
        vec![0.0, 2.0],
        vec![0.0_f32, 100.0],
        InterpolationMode::Linear,
    );

    let mut cursor = KeyframeCursor::default();
    assert!(approx(track.sample_with_cursor(0.0, &mut cursor), 0.0));
    assert!(approx(track.sample_with_cursor(1.0, &mut cursor), 50.0));
    assert!(approx(track.sample_with_cursor(2.0, &mut cursor), 100.0));
}

// ============================================================================
// Interpolatable Implementations
// ============================================================================

#[test]
fn interpolatable_f32_linear() {
    let result = f32::interpolate_linear(&0.0, &10.0, 0.25);
    assert!(approx(result, 2.5));
}

#[test]
fn interpolatable_vec3_linear() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(10.0, 20.0, 30.0);
    let result = Vec3::interpolate_linear(&a, &b, 0.5);
    assert!(approx(result.x, 5.0));
    assert!(approx(result.y, 10.0));
    assert!(approx(result.z, 15.0));
}

#[test]
fn interpolatable_quat_linear_is_slerp() {
    let a = Quat::IDENTITY;
    let b = Quat::from_rotation_y(FRAC_PI_2);
    let result = Quat::interpolate_linear(&a, &b, 0.5);

    let expected = a.slerp(b, 0.5);
    let angle = result.angle_between(expected);
    assert!(angle < 1e-4, "Slerp mismatch: angle={angle}");
}

#[test]
fn interpolatable_morph_weight_linear() {
    let mut a = MorphWeightData::allocate(4);
    a.weights[0] = 0.0;
    a.weights[1] = 1.0;
    a.weights[2] = 0.5;
    a.weights[3] = 0.0;

    let mut b = MorphWeightData::allocate(4);
    b.weights[0] = 1.0;
    b.weights[1] = 0.0;
    b.weights[2] = 0.5;
    b.weights[3] = 1.0;

    let result = MorphWeightData::interpolate_linear(&a, &b, 0.5);
    assert!(approx(result.weights[0], 0.5));
    assert!(approx(result.weights[1], 0.5));
    assert!(approx(result.weights[2], 0.5));
    assert!(approx(result.weights[3], 0.5));
}

// ============================================================================
// AnimationAction Loop Modes
// ============================================================================

fn make_simple_clip(duration: f32) -> Arc<AnimationClip> {
    Arc::new(AnimationClip::new(
        "test".to_string(),
        vec![Track {
            meta: TrackMeta {
                node_name: "node".to_string(),
                target: TargetPath::Translation,
            },
            data: TrackData::Vector3(KeyframeTrack::new(
                vec![0.0, duration],
                vec![Vec3::ZERO, Vec3::X],
                InterpolationMode::Linear,
            )),
        }],
    ))
}

#[test]
fn action_loop_mode_once() {
    let clip = make_simple_clip(2.0);
    let mut action = AnimationAction::new(clip);
    action.loop_mode = LoopMode::Once;
    action.time = 0.0;

    // Advance past end
    action.update(3.0);
    assert!(
        approx(action.time, 2.0),
        "Once: should clamp to duration, got {}",
        action.time
    );
    assert!(action.paused, "Once: should auto-pause at end");
}

#[test]
fn action_loop_mode_loop() {
    let clip = make_simple_clip(2.0);
    let mut action = AnimationAction::new(clip);
    action.loop_mode = LoopMode::Loop;
    action.time = 0.0;

    // Advance past end by 0.5
    action.update(2.5);
    assert!(
        approx(action.time, 0.5),
        "Loop: should wrap to 0.5, got {}",
        action.time
    );
    assert!(!action.paused, "Loop: should NOT auto-pause");
}

#[test]
fn action_loop_reverse_playback() {
    let clip = make_simple_clip(2.0);
    let mut action = AnimationAction::new(clip);
    action.loop_mode = LoopMode::Loop;
    action.time_scale = -1.0;
    action.time = 0.5;

    // Advance with negative time_scale: time += 1.0 * -1.0 = -0.5
    // time = 0.5 + (-1.0) = -0.5 → Loop wrap: 2.0 + (-0.5 % 2.0) = 2.0 - 0.5 = 1.5
    action.update(1.0);
    assert!(
        action.time > 0.0 && action.time <= 2.0,
        "Loop reverse: time should be within [0, duration], got {}",
        action.time
    );
}

#[test]
fn action_paused_no_update() {
    let clip = make_simple_clip(2.0);
    let mut action = AnimationAction::new(clip);
    action.paused = true;
    action.time = 0.5;

    action.update(1.0);
    assert!(approx(action.time, 0.5), "Paused action should not advance");
}

#[test]
fn action_disabled_no_update() {
    let clip = make_simple_clip(2.0);
    let mut action = AnimationAction::new(clip);
    action.enabled = false;
    action.time = 0.5;

    action.update(1.0);
    assert!(
        approx(action.time, 0.5),
        "Disabled action should not advance"
    );
}

#[test]
fn action_time_scale() {
    let clip = make_simple_clip(4.0);
    let mut action = AnimationAction::new(clip);
    action.loop_mode = LoopMode::Once;
    action.time_scale = 2.0;
    action.time = 0.0;

    action.update(1.0); // dt=1.0, time_scale=2.0, so effective dt=2.0
    assert!(
        approx(action.time, 2.0),
        "Expected 2.0, got {}",
        action.time
    );
}

// ============================================================================
// AnimationClip Auto-Duration
// ============================================================================

#[test]
fn clip_auto_duration() {
    let clip = AnimationClip::new(
        "test".to_string(),
        vec![
            Track {
                meta: TrackMeta {
                    node_name: "a".to_string(),
                    target: TargetPath::Translation,
                },
                data: TrackData::Vector3(KeyframeTrack::new(
                    vec![0.0, 1.5],
                    vec![Vec3::ZERO, Vec3::X],
                    InterpolationMode::Linear,
                )),
            },
            Track {
                meta: TrackMeta {
                    node_name: "b".to_string(),
                    target: TargetPath::Rotation,
                },
                data: TrackData::Quaternion(KeyframeTrack::new(
                    vec![0.0, 3.0],
                    vec![Quat::IDENTITY, Quat::from_rotation_y(1.0)],
                    InterpolationMode::Linear,
                )),
            },
        ],
    );

    assert!(
        approx(clip.duration, 3.0),
        "Duration should be max of all tracks (3.0), got {}",
        clip.duration
    );
}

#[test]
fn clip_empty_tracks_zero_duration() {
    let clip = AnimationClip::new("empty".to_string(), vec![]);
    assert!(approx(clip.duration, 0.0));
}
