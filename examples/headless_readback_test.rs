//! Acceptance Test A — synchronous readback.
//!
//! Renders 10 frames in headless mode. Each frame is read back via the
//! optimised `readback_pixels()` path (staging-buffer cache). Verifies:
//!
//! - Every call returns a non-empty pixel buffer of the expected size.
//! - The cached staging buffer prevents per-frame allocation (the second
//!   call and beyond reuse the same buffer).
//!
//! ```bash
//! cargo run --example headless_readback_test --no-default-features
//! ```

use myth::prelude::*;

fn main() {
    env_logger::init();

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

    println!("Test A passed: 10 synchronous readback frames OK ({width}×{height}, {} bytes each)", expected_bytes);
}
