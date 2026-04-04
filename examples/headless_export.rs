//! Headless (offscreen) rendering example.
//!
//! Initialises the GPU without a window, renders a single frame of a
//! textured cube lit by a directional light, reads back the framebuffer,
//! and saves the result as `output.png`.
//!
//! ```bash
//! cargo run --example headless_export --no-default-features
//! ```

use myth::prelude::*;

fn main() {
    env_logger::init();

    let mut engine = Engine::default();

    // Initialise GPU in headless mode — no window, no surface.
    let width: u32 = 800;
    let height: u32 = 600;
    pollster::block_on(engine.init_headless(width, height)).expect("headless init failed");

    // ── Scene setup ──────────────────────────────────────────────────
    let scene = engine.scene_manager.create_active();

    // Checkerboard material
    let image_handle = engine
        .assets
        .images
        .add(Image::checkerboard(512, 512, 64));
    let tex_handle = engine
        .assets
        .textures
        .add(Texture::new_2d(Some("checker"), image_handle));
    let mat = UnlitMaterial::new(Vec4::ONE).with_map(tex_handle);

    // Spawn a box
    let _cube = scene.spawn_box(2.0, 2.0, 2.0, mat, &engine.assets);

    // Camera
    let cam = scene.add_camera(Camera::new_perspective(
        45.0,
        width as f32 / height as f32,
        0.1,
    ));
    scene
        .node(&cam)
        .set_position(0.0, 3.0, 8.0)
        .look_at(Vec3::ZERO);
    scene.active_camera = Some(cam);

    // Directional light
    let light = scene.add_light(Light::new_directional(Vec3::ONE, 5.0));
    scene
        .node(&light)
        .set_position(5.0, 10.0, 5.0)
        .look_at(Vec3::ZERO);

    // ── Render one frame ─────────────────────────────────────────────
    engine.update(0.016);
    let rendered = engine.render_active_scene();
    assert!(rendered, "render_active_scene returned false");

    // ── Readback & save ──────────────────────────────────────────────
    let pixels = engine.readback_pixels().expect("readback failed");
    image::save_buffer(
        "output.png",
        &pixels,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .expect("failed to save output.png");

    println!("Saved output.png ({width}×{height})");
}
