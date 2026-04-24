//! [gallery]
//! name = "3D Gaussian Splatting"
//! category = "Advanced"
//! description = "Loads and renders a 3D Gaussian Splatting point cloud from an NPZ file."
//! order = 500
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct GaussianSplattingDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for GaussianSplattingDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // Load the compressed NPZ point cloud
        let npz_path = format!("{}3dgs/point_cloud.npz", ASSET_PATH);
        let cloud = myth::load_gaussian_npz_from_source(npz_path)
            .expect("Failed to load NPZ Gaussian cloud");

        // Register in the asset server and add to scene
        let cloud_handle = engine.assets.gaussian_clouds.add(cloud);
        let _cloud_node = scene.add_gaussian_cloud("gaussian_cloud", cloud_handle);

        scene.node(&_cloud_node).set_rotation_euler(std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::FRAC_PI_2);

        // Camera — use the first camera from the training data as a starting view
        let camera_pos = Vec3::new(0.0, 2.0, 2.5);
        let target = Vec3::ZERO;

        let cam_node = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        scene.active_camera = Some(cam_node);

        Self {
            controls: OrbitControls::new(camera_pos, target),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        if let Some((transform, camera)) = engine
            .scene_manager
            .active_scene_mut()
            .and_then(|s| s.query_main_camera_bundle())
        {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("3D Gaussian Splatting | FPS: {:.2}", fps));
        }

        // if frame.frame_count == 10 {
        //     if let Some(dump) = engine.renderer.dump_graph_mermaid() {
        //         println!("{}", dump);
        //     }
        // }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Myth Engine — 3D Gaussian Splatting")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<GaussianSplattingDemo>()
}
