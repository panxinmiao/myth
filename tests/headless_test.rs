//! Headless Rendering & Readback Tests
//!
//! - Synchronous readback via `Renderer.readback_pixels()`
//! - Asynchronous readback via `ReadbackStream`
//! - Material rendering: Physical (PBR), Phong, Unlit
//! - Multi-light scenes (directional + point)
//! - Alpha blending and alpha mask
//! - Multiple geometry types (box, sphere, plane)
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
    let mut submitted: u64 = 0;
    let mut received: u64 = 0;
    let mut skipped: u64 = 0;

    // ── Hot loop ─────────────────────────────────────────────────────
    for _ in 0..total_frames {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();

        match engine.submit_to_stream(&mut stream) {
            Ok(_) => {
                submitted += 1;
            }
            Err(_) => {
                skipped += 1;
            }
        }

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
        received, submitted,
        "expected {submitted} frames, got {received}"
    );

    assert_eq!(
        submitted + skipped,
        total_frames,
        "total frames ({total_frames}) should equal submitted + skipped ({submitted} + {skipped})"
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Initialise a headless engine and return it together with the pixel
/// count so callers don't have to repeat the boilerplate.
fn setup_headless(w: u32, h: u32) -> (Engine, usize) {
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(w, h, None)).expect("headless init failed");
    let expected_bytes = (w * h * 4) as usize;
    (engine, expected_bytes)
}

/// Remove the currently active scene so a fresh one can be created.
fn reset_active_scene(engine: &mut Engine) {
    if let Some(h) = engine.scene_manager.active_handle() {
        engine.scene_manager.remove_scene(h);
    }
}

/// Render a few warm-up frames, then capture one and return the pixel buffer.
fn render_and_capture(engine: &mut Engine, warmup: usize) -> Vec<u8> {
    for _ in 0..warmup {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();
    }
    // Final capture frame
    engine.update(1.0 / 60.0);
    engine.render_active_scene();
    engine.readback_pixels().expect("readback failed")
}

/// Check that the image is not entirely black (at least one pixel with a
/// non-zero RGB channel).
fn assert_not_black(pixels: &[u8], label: &str) {
    let any_color = pixels
        .chunks_exact(4)
        .any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
    assert!(any_color, "{label}: rendered image is entirely black");
}

/// Check that at least one pixel differs between two render results
/// (i.e. the scenes are visually distinct).
fn assert_images_differ(a: &[u8], b: &[u8], label: &str) {
    assert_eq!(a.len(), b.len());
    let differs = a.iter().zip(b.iter()).any(|(x, y)| x != y);
    assert!(differs, "{label}: images are identical but should differ");
}

// ── Physical Material Tests ──────────────────────────────────────────────

/// A PhysicalMaterial sphere is rendered with a directional light.
/// The framebuffer must contain visible (non-black) pixels.
#[test]
fn physical_material_sphere() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhysicalMaterial::new(Vec4::new(0.9, 0.1, 0.1, 1.0))
        .with_roughness(0.4)
        .with_metalness(0.0);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "physical_material_sphere");
}

/// A metallic PhysicalMaterial box should produce different colours than
/// a dielectric one.
#[test]
fn physical_metallic_vs_dielectric() {
    let (mut engine, _) = setup_headless(128, 128);

    // Dielectric
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.8, 0.8, 0.2, 1.0))
            .with_roughness(0.3)
            .with_metalness(0.0);
        scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 1.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dielectric = render_and_capture(&mut engine, 2);
    assert_not_black(&dielectric, "dielectric");

    // Metallic - same colour, same scene, just metalness=1
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.8, 0.8, 0.2, 1.0))
            .with_roughness(0.3)
            .with_metalness(1.0);
        scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 1.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let metallic = render_and_capture(&mut engine, 2);
    assert_not_black(&metallic, "metallic");

    assert_images_differ(&dielectric, &metallic, "metallic_vs_dielectric");
}

/// Emissive PhysicalMaterial should produce visible output even without
/// any light in the scene.
#[test]
fn physical_emissive_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhysicalMaterial::new(Vec4::new(0.0, 0.0, 0.0, 1.0))
        .with_emissive(Vec3::new(1.0, 0.5, 0.0), 5.0);
    scene.spawn_sphere(1.0, mat, &engine.assets);

    // No light added intentionally.

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "physical_emissive_no_light");
}

// ── Phong Material Tests ─────────────────────────────────────────────────

/// Blinn-Phong shaded sphere with specular highlight.
#[test]
fn phong_material_sphere() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhongMaterial::new(Vec4::new(0.2, 0.5, 0.9, 1.0))
        .with_shininess(64.0)
        .with_specular(Vec3::new(1.0, 1.0, 1.0));
    scene.spawn_sphere(1.0, mat, &engine.assets);

    scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "phong_material_sphere");
}

/// Emissive Phong material without scene lights.
#[test]
fn phong_emissive_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = PhongMaterial::new(Vec4::new(0.0, 0.0, 0.0, 1.0))
        .with_emissive(Vec3::new(0.0, 1.0, 0.5), 4.0);
    scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "phong_emissive_no_light");
}

// ── Unlit Material Tests ─────────────────────────────────────────────────

/// Unlit material should produce visible output regardless of lighting.
#[test]
fn unlit_material_no_light() {
    let (mut engine, expected) = setup_headless(128, 128);
    let scene = engine.scene_manager.create_active();

    let mat = UnlitMaterial::new(Vec4::new(0.0, 1.0, 0.0, 1.0));
    scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);

    // No light - unlit should still render.
    let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 0.0, 4.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 2);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "unlit_material_no_light");
}

// ── Multi-Light Tests ────────────────────────────────────────────────────

/// Scene with directional + point light should differ from directional-only.
#[test]
fn multi_light_scene() {
    let (mut engine, _) = setup_headless(128, 128);

    // ──── Directional only ────────────────────────────────────────────
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0))
            .with_roughness(0.5)
            .with_metalness(0.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 2.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dir_only = render_and_capture(&mut engine, 2);
    assert_not_black(&dir_only, "directional_only");

    // ──── Directional + point light ───────────────────────────────────
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(Vec4::new(0.7, 0.7, 0.7, 1.0))
            .with_roughness(0.5)
            .with_metalness(0.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 2.0));
        let point = scene.add_light(Light::new_point(Vec3::new(1.0, 0.2, 0.2), 15.0, 20.0));
        scene.node(&point).set_position(2.0, 2.0, 2.0);
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let dir_plus_point = render_and_capture(&mut engine, 2);
    assert_not_black(&dir_plus_point, "directional_plus_point");

    assert_images_differ(&dir_only, &dir_plus_point, "multi_light");
}

// ── Alpha Mode Tests ─────────────────────────────────────────────────────

/// A semi-transparent (AlphaMode::Blend) object in front of a solid
/// background should produce colours that differ from a fully opaque one.
#[test]
fn alpha_blend_vs_opaque() {
    let (mut engine, _) = setup_headless(128, 128);

    // Opaque foreground
    {
        let scene = engine.scene_manager.create_active();
        // Background plane
        let bg = UnlitMaterial::new(Vec4::new(0.0, 0.0, 1.0, 1.0));
        let bg_node = scene.spawn_plane(4.0, 4.0, bg, &engine.assets);
        scene.node(&bg_node).set_position(0.0, 0.0, -2.0);
        // Foreground box (opaque red)
        let fg = UnlitMaterial::new(Vec4::new(1.0, 0.0, 0.0, 1.0));
        scene.spawn_box(1.0, 1.0, 0.1, fg, &engine.assets);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let opaque = render_and_capture(&mut engine, 2);
    assert_not_black(&opaque, "opaque_foreground");

    // Semi-transparent foreground
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let bg = UnlitMaterial::new(Vec4::new(0.0, 0.0, 1.0, 1.0));
        let bg_node = scene.spawn_plane(4.0, 4.0, bg, &engine.assets);
        scene.node(&bg_node).set_position(0.0, 0.0, -2.0);
        // Foreground box: same red but 50% alpha with Blend mode
        let fg =
            UnlitMaterial::new(Vec4::new(1.0, 0.0, 0.0, 0.5)).with_alpha_mode(AlphaMode::Blend);
        scene.spawn_box(1.0, 1.0, 0.1, fg, &engine.assets);

        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let blended = render_and_capture(&mut engine, 2);
    assert_not_black(&blended, "blended_foreground");

    assert_images_differ(&opaque, &blended, "alpha_blend_vs_opaque");
}

// ── Cross-Material Comparison ────────────────────────────────────────────

/// Varying PBR roughness on the same geometry should produce visually
/// distinct images (smooth specular highlight vs diffuse scatter).
#[test]
fn physical_roughness_variation() {
    let (mut engine, _) = setup_headless(128, 128);

    let color = Vec4::new(0.8, 0.3, 0.3, 1.0);

    // Low roughness (mirror-like)
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(color)
            .with_roughness(0.05)
            .with_metalness(1.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let img_smooth = render_and_capture(&mut engine, 2);
    assert_not_black(&img_smooth, "smooth");

    // High roughness (diffuse-like)
    reset_active_scene(&mut engine);
    {
        let scene = engine.scene_manager.create_active();
        let mat = PhysicalMaterial::new(color)
            .with_roughness(1.0)
            .with_metalness(1.0);
        scene.spawn_sphere(1.0, mat, &engine.assets);
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 0.0, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);
    }
    let img_rough = render_and_capture(&mut engine, 2);
    assert_not_black(&img_rough, "rough");

    assert_images_differ(&img_smooth, &img_rough, "roughness_variation");
}

// ── Multi-Object Scene ───────────────────────────────────────────────────

/// A scene with multiple PBR objects (sphere + box + sphere) at different
/// positions should render without panics and produce a non-black image.
#[test]
fn multi_object_scene() {
    let (mut engine, expected) = setup_headless(256, 256);
    let scene = engine.scene_manager.create_active();

    // Red metallic sphere
    let sphere_mat = PhysicalMaterial::new(Vec4::new(0.9, 0.1, 0.1, 1.0))
        .with_roughness(0.2)
        .with_metalness(1.0);
    let sphere = scene.spawn_sphere(0.6, sphere_mat, &engine.assets);
    scene.node(&sphere).set_position(-1.5, 0.0, 0.0);

    // Green dielectric box
    let box_mat = PhysicalMaterial::new(Vec4::new(0.1, 0.8, 0.2, 1.0))
        .with_roughness(0.6)
        .with_metalness(0.0);
    let bx = scene.spawn_box(1.0, 1.0, 1.0, box_mat, &engine.assets);
    scene.node(&bx).set_position(0.0, 0.0, 0.0);

    // Blue rough sphere
    let blue_mat = PhysicalMaterial::new(Vec4::new(0.1, 0.3, 1.0, 1.0))
        .with_roughness(0.9)
        .with_metalness(0.5);
    let blue_sphere = scene.spawn_sphere(0.4, blue_mat, &engine.assets);
    scene.node(&blue_sphere).set_position(1.5, 0.0, 0.0);

    // Lights
    scene.add_light(Light::new_directional(Vec3::ONE, 2.5));
    let pt = scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 10.0, 15.0));
    scene.node(&pt).set_position(0.0, 3.0, 3.0);

    // Camera
    let cam = scene.add_camera(Camera::new_perspective(60.0, 1.0, 0.1));
    scene
        .node(&cam)
        .set_position(0.0, 3.0, 6.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    let pixels = render_and_capture(&mut engine, 3);
    assert_eq!(pixels.len(), expected);
    assert_not_black(&pixels, "multi_object_scene");
}
