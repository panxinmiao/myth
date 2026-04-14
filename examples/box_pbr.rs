//! [gallery]
//! name = "PBR Box"
//! category = "Materials"
//! description = "Physically based material test with checker albedo and image-based lighting."
//! order = 120
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

/// PBR Material Cube Example
struct PbrBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for PbrBox {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let image_handle = engine.assets.images.add(Image::checkerboard(512, 512, 64));
        let tex_handle = engine
            .assets
            .textures
            .add(Texture::new_2d(Some("checker"), image_handle));

        // spawn with builder-style PBR material
        let cube_node_id = scene.spawn_box(
            2.0,
            2.0,
            2.0,
            PhysicalMaterial::new(Vec4::ONE).with_map(tex_handle),
            &engine.assets,
        );

        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        // Load environment map
        let env_texture_handle = engine.assets.load_cube_texture(
            [
                format!("{}envs/Park2/posx.jpg", ASSET_PATH),
                format!("{}envs/Park2/negx.jpg", ASSET_PATH),
                format!("{}envs/Park2/posy.jpg", ASSET_PATH),
                format!("{}envs/Park2/negy.jpg", ASSET_PATH),
                format!("{}envs/Park2/posz.jpg", ASSET_PATH),
                format!("{}envs/Park2/negz.jpg", ASSET_PATH),
            ],
            ColorSpace::Srgb,
            true,
        );
        scene.environment.set_env_map(Some(env_texture_handle));

        // Camera
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 3.0, 10.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        Self {
            cube_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
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
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        // FPS display
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Box PBR | FPS: {:.2}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<PbrBox>()
}
