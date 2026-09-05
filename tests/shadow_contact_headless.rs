//! Shadow Contact Tests
//!
//! - A sealed room is not lit by a sun it cannot see
//! - Which faces the shadow map records is configurable per light

use myth::prelude::*;
use myth::scene::light::{ShadowConfig, ShadowFaces};

const SIZE: u32 = 256;

/// Peak, not mean: a leak is a thin line, which moves the extremes and
/// barely moves the average.
fn peak_delta(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// A closed box of six slabs with the camera inside and the sun outside, so
/// any sunlight reaching the camera came through a solid wall.
fn sealed_room(engine: &mut Engine, faces: ShadowFaces, intensity: f32) -> Vec<u8> {
    let scene = engine.scene_manager.create_active();
    scene.environment.set_ambient_light(Vec3::splat(0.3));

    let mut sun = Light::new_directional(Vec3::ONE, intensity);
    sun.cast_shadows = true;
    sun.shadow = Some(ShadowConfig {
        faces,
        bias: 0.0,
        normal_bias: 0.0,
        map_size: 1024,
        cascade_split_lambda: 0.9,
        max_shadow_distance: 100.0,
        ..Default::default()
    });
    let sun = scene.add_light(sun);
    scene
        .node(&sun)
        .set_position(60.0, 120.0, 40.0)
        .look_at(Vec3::ZERO);

    let wall = PhongMaterial::new(Vec4::new(0.8, 0.8, 0.8, 1.0));
    // Room-sized rather than toy-sized: the artefact is a shadow-map texel
    // wide, so it only shows when the map is spread over a real interior.
    let (inner, thickness, height) = (60.0, 0.67, 12.0);
    let outer = inner + thickness * 2.0;
    // Centre-to-centre from the room's middle out to a wall.
    let reach = inner / 2.0 + thickness / 2.0;
    let slabs = [
        (
            Vec3::new(0.0, -thickness / 2.0, 0.0),
            Vec3::new(outer, thickness, outer),
        ),
        (
            Vec3::new(0.0, height + thickness / 2.0, 0.0),
            Vec3::new(outer, thickness, outer),
        ),
        (
            Vec3::new(0.0, height / 2.0, -reach),
            Vec3::new(outer, height, thickness),
        ),
        (
            Vec3::new(0.0, height / 2.0, reach),
            Vec3::new(outer, height, thickness),
        ),
        (
            Vec3::new(-reach, height / 2.0, 0.0),
            Vec3::new(thickness, height, inner),
        ),
        (
            Vec3::new(reach, height / 2.0, 0.0),
            Vec3::new(thickness, height, inner),
        ),
    ];
    for (centre, size) in slabs {
        let node = scene.spawn_box(size.x, size.y, size.z, wall.clone(), &engine.assets);
        scene.node(&node).set_position(centre.x, centre.y, centre.z);
    }

    let cam = scene.add_camera(Camera::new_perspective(60.0, 1.0, 0.1));
    // Standing in the middle, looking down at where a far wall meets the floor.
    scene
        .node(&cam)
        .set_position(0.0, 5.0, 20.0)
        .look_at(Vec3::new(0.0, 0.0, -inner / 2.0));
    scene.active_camera = Some(cam);

    for _ in 0..3 {
        engine.update(1.0 / 60.0);
        engine.render_active_scene();
    }
    engine.readback_pixels().expect("readback failed")
}

#[test]
fn sealed_room_is_not_lit_by_a_sun_it_cannot_see() {
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(SIZE, SIZE, None)).expect("headless init failed");

    let lit = sealed_room(&mut engine, ShadowFaces::Front, 4.0);
    let unlit = sealed_room(&mut engine, ShadowFaces::Front, 0.0);

    let delta = peak_delta(&lit, &unlit);
    assert!(
        delta <= 8,
        "sealed room changed by {delta}/255 when the sun came up; light is reaching through a wall"
    );
}

#[test]
fn shadow_faces_selects_a_different_pipeline() {
    let mut engine = Engine::default();
    pollster::block_on(engine.init_headless(SIZE, SIZE, None)).expect("headless init failed");

    let back = sealed_room(&mut engine, ShadowFaces::Back, 4.0);
    let front = sealed_room(&mut engine, ShadowFaces::Front, 4.0);

    assert!(
        peak_delta(&back, &front) > 8,
        "the two face modes rendered the same image; the setting is not reaching the pipeline"
    );
}
