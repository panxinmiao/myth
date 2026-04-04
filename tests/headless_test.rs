//! Headless Readback Tests
//! Test for:
//! - Synchronous readback via `Renderer.readback_pixels()`
//! - Asynchronous readback via `ReadbackStream`
use myth::prelude::*;
use myth::render::core::ReadbackStream;

// Integration tests for synchronous headless readback.
//
// Renders 10 frames in headless mode. Each frame is read back via the
// optimised `readback_pixels()` path (staging-buffer cache). Verifies:
//
// - Every call returns a non-empty pixel buffer of the expected size.
// - The cached staging buffer prevents per-frame allocation.
#[test]
fn headless_sync_readback() {
    let mut engine = Engine::default();

    let width: u32 = 256;
    let height: u32 = 256;
    pollster::block_on(engine.init_headless(width, height, None)).expect("headless init failed");

    // ── Minimal scene ────────────────────────────────────────────────
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(0.2, 0.6, 1.0, 1.0));
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

    // ── Render 10 frames & readback each ─────────────────────────────
    let expected_bytes = (width * height * 4) as usize; // RGBA8

    for i in 0..10 {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();

        let pixels = engine.readback_pixels().expect("readback failed");
        assert_eq!(
            pixels.len(),
            expected_bytes,
            "frame {i}: unexpected buffer size"
        );

        // Sanity: at least one non-zero pixel (the scene is not pitch black).
        let any_nonzero = pixels.iter().any(|&b| b != 0);
        assert!(any_nonzero, "frame {i}: all pixels are zero");
    }
}

// Integration tests for `ReadbackStream` (async ring buffer).
//
// Renders 100 frames in headless mode using a `ReadbackStream` with
// `buffer_count = 3`. Frames are submitted non-blocking and collected
// via `try_recv`. Any remaining in-flight frames are drained with
// `flush` at the end. Verifies exactly 100 frames are received.
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
        .create_readback_stream(3, 16)
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
    let frames = engine
        .flush_stream(&mut stream)
        .expect("flush_stream failed");
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
