use std::sync::Arc;

use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;
use winit::window::Window;

/// PBR Material Cube Example
struct PbrBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for PbrBox {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        // 1. prepare resources
        let scene = engine.scene_manager.create_active();
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);

        // 2. Create PBR Material
        let physical_mat = MeshPhysicalMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0));
        let tex_handle = engine.assets.textures.add(texture);
        physical_mat.set_map(Some(tex_handle));

        // 3. Create Mesh and add to scene
        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(physical_mat);
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = scene.add_mesh(mesh);

        // 4. Add Light
        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);

        // 5. Load environment map
        let env_texture_handle = engine
            .assets
            .load_cube_texture(
                [
                    "examples/assets/Park2/posx.jpg",
                    "examples/assets/Park2/negx.jpg",
                    "examples/assets/Park2/posy.jpg",
                    "examples/assets/Park2/negy.jpg",
                    "examples/assets/Park2/posz.jpg",
                    "examples/assets/Park2/negz.jpg",
                ],
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load environment map");

        scene.environment.set_env_map(Some(env_texture_handle));

        // 6. Setup Camera
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            cube_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        // Rotate cube
        if let Some(node) = scene.get_node_mut(self.cube_node_id) {
            let rot_y = Quat::from_rotation_y(0.02 * 60.0 * frame.dt);
            let rot_x = Quat::from_rotation_x(0.01 * 60.0 * frame.dt);
            node.transform.rotation = node.transform.rotation * rot_y * rot_x;
        }

        // Orbit controls
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        // FPS display
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Box PBR | FPS: {:.2}", fps));
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
        .run::<PbrBox>()
}
