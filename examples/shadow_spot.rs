use std::sync::Arc;

use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;
use myth::{PlaneOptions, create_plane};
use winit::window::Window;

struct ShadowSpotDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for ShadowSpotDemo {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        let scene = engine.scene_manager.create_active();

        let sphere_geo = engine.assets.geometries.add(Geometry::new_sphere(1.0));
        let sphere_mat = engine
            .assets
            .materials
            .add(MeshPhysicalMaterial::new(Vec4::new(0.2, 0.7, 1.0, 1.0)));
        let mut sphere = Mesh::new(sphere_geo, sphere_mat);
        sphere.cast_shadows = true;
        sphere.receive_shadows = true;
        let sphere_node = scene.add_mesh(sphere);
        if let Some(node) = scene.get_node_mut(sphere_node) {
            node.transform.position = Vec3::new(0.0, 1.0, 0.0);
        }

        let floor_geo = engine.assets.geometries.add(create_plane(&PlaneOptions {
            width: 25.0,
            height: 25.0,
            ..Default::default()
        }));
        let floor_material = MeshPhysicalMaterial::new(Vec4::new(0.9, 0.9, 0.9, 1.0));
        floor_material.set_side(Side::Double);
        let floor_mat = engine.assets.materials.add(floor_material);
        let mut floor = Mesh::new(floor_geo, floor_mat);
        floor.receive_shadows = true;
        let floor_node = scene.add_mesh(floor);
        if let Some(node) = scene.get_node_mut(floor_node) {
            node.transform.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        }

        let mut spot = Light::new_spot(Vec3::new(1.0, 0.95, 0.9), 30.0, 40.0, 0.35, 0.55);
        spot.cast_shadows = true;
        if let Some(shadow) = spot.shadow.as_mut() {
            shadow.map_size = 1024;
        }
        let spot_node = scene.add_light(spot);
        if let Some(node) = scene.get_node_mut(spot_node) {
            node.transform.position = Vec3::new(6.0, 10.0, 4.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        let camera = Camera::new_perspective(45.0, 16.0 / 9.0, 0.1);
        let cam_node = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node) {
            node.transform.position = Vec3::new(8.0, 6.0, 8.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        scene.active_camera = Some(cam_node);

        Self {
            controls: OrbitControls::new(Vec3::new(8.0, 6.0, 8.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Shadow Spot | FPS: {:.2}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ShadowSpotDemo>()
}
