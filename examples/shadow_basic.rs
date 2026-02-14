use std::sync::Arc;

use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;
use myth::{PlaneOptions, create_plane};
use winit::window::Window;

struct ShadowBasicDemo {
    cube_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for ShadowBasicDemo {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        let scene = engine.scene_manager.create_active();

        let cube_geo = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.5, 1.5, 1.5));
        let cube_mat = engine
            .assets
            .materials
            .add(MeshPhongMaterial::new(Vec4::new(0.9, 0.3, 0.2, 1.0)));
        let mut cube = Mesh::new(cube_geo, cube_mat);
        cube.cast_shadows = true;
        cube.receive_shadows = true;
        let cube_node = scene.add_mesh(cube);

        if let Some(node) = scene.get_node_mut(cube_node) {
            node.transform.position = Vec3::new(0.0, 3.2, 0.0);
        }

        let ground_geo = engine.assets.geometries.add(create_plane(&PlaneOptions {
            width: 30.0,
            height: 30.0,
            ..Default::default()
        }));
        let ground_material = MeshPhongMaterial::new(Vec4::new(0.8, 0.8, 0.85, 1.0));
        ground_material.set_side(Side::Double);
        let ground_mat = engine.assets.materials.add(ground_material);
        let mut ground = Mesh::new(ground_geo, ground_mat);
        ground.receive_shadows = true;
        let ground_node = scene.add_mesh(ground);

        if let Some(node) = scene.get_node_mut(ground_node) {
            node.transform.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            node.transform.position = Vec3::new(0.0, 0.0, 0.0);
        }

        let mut dir_light = Light::new_directional(Vec3::ONE, 5.0);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let light_node = scene.add_light(dir_light);

        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(8.0, 12.0, 6.0);
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
            cube_node,
            controls: OrbitControls::new(Vec3::new(8.0, 6.0, 8.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(node) = scene.get_node_mut(self.cube_node) {
            node.transform.rotation *= Quat::from_rotation_y(1.2 * frame.dt);
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Shadow Basic | FPS: {:.2}", fps));
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
        .run::<ShadowBasicDemo>()
}
