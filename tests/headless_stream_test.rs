//! Integration tests for `ReadbackStream` (async ring buffer).
//!
//! Renders 100 frames in headless mode using a `ReadbackStream` with
//! `buffer_count = 3`. Frames are submitted non-blocking and collected
//! via `try_recv`. Any remaining in-flight frames are drained with
//! `flush` at the end. Verifies exactly 100 frames are received.

use myth::prelude::*;
use myth::render::core::ReadbackStream;

#[test]
fn headless_stream_recording() {
    let mut engine = Engine::default();

    let width: u32 = 256;
    let height: u32 = 256;
    pollster::block_on(engine.init_headless(width, height, None)).expect("headless init failed");

    // ── Minimal scene ────────────────────────────────────────────────
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(1.0, 0.4, 0.1, 1.0));
    let _cube = scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    let cam = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&cam)
        .set_position(0.0, 2.0, 5.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    // ── Create ReadbackStream ────────────────────────────────────────
    let mut stream: ReadbackStream = engine
        .renderer
        .create_readback_stream(3)
        .expect("create_readback_stream failed");

    let total_frames: u64 = 100;
    let expected_bytes = (width * height * 4) as usize;
    let mut received: u64 = 0;

    // ── Hot loop ─────────────────────────────────────────────────────
    for _ in 0..total_frames {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();

        engine
            .submit_to_stream(&mut stream)
            .expect("submit_to_stream failed");

        // Drive GPU callbacks.
        engine.poll_device();

        // Opportunistically pull ready frames.
        while let Some(frame) = stream.try_recv().expect("try_recv failed") {
            assert_eq!(
                frame.pixels.len(),
                expected_bytes,
                "frame {}: unexpected pixel buffer size",
                frame.frame_index
            );
            received += 1;
        }
    }

    // ── Flush remaining ──────────────────────────────────────────────
    let frames = engine.flush_stream(&mut stream).expect("flush_stream failed");
    for frame in frames {
        assert_eq!(
            frame.pixels.len(),
            expected_bytes,
            "flush frame {}: unexpected pixel buffer size",
            frame.frame_index
        );
        received += 1;
    }

    assert_eq!(
        received, total_frames,
        "expected {total_frames} frames, got {received}"
    );
}
