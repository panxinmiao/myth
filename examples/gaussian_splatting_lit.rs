//! [gallery]
//! name = "3DGS Deferred Lighting"
//! category = "Advanced"
//! description = "Shows deferred real-time lighting on a 3D Gaussian Splatting asset with moving local lights and mesh occluders."
//! order = 520
//! features = ["3dgs", "gaussian-npz"]
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct GaussianSplattingLitDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    light_handles: [NodeHandle; 2],
    light_orbs: [NodeHandle; 2],
    time: f32,
}

impl AppHandler for GaussianSplattingLitDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let box_geometry = engine
            .assets
            .geometries
            .add(Geometry::new_box(1.0, 1.0, 1.0));
        let sphere_geometry = engine.assets.geometries.add(Geometry::new_sphere(1.0));

        let floor_material = PhysicalMaterial::new(Vec4::new(0.16, 0.18, 0.22, 1.0))
            .with_roughness(0.88)
            .with_metalness(0.0);
        let accent_material = PhysicalMaterial::new(Vec4::new(0.78, 0.46, 0.24, 1.0))
            .with_roughness(0.38)
            .with_metalness(0.02);
        let orb_material = PhysicalMaterial::new(Vec4::new(0.88, 0.92, 1.0, 1.0))
            .with_roughness(0.18)
            .with_metalness(0.0);

        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.02, 0.03, 0.06, 1.0),
            Vec4::new(0.07, 0.05, 0.03, 1.0),
        ));

        let floor = scene.add_mesh(Mesh::new(box_geometry, engine.assets.materials.add(floor_material)));
        scene
            .node(&floor)
            .set_position(0.0, -0.7, 0.0)
            .set_scale_xyz(15.0, 0.25, 15.0);

        let center_block = scene.add_mesh(Mesh::new(box_geometry, engine.assets.materials.add(accent_material)));
        scene
            .node(&center_block)
            .set_position(0.0, 0.1, -1.6)
            .set_scale_xyz(1.8, 1.5, 0.75)
            .set_rotation_euler(0.0, 0.25, 0.0);

        let cloud_handle = engine
            .assets
            .load_gaussian_npz(format!("{}3dgs/point_cloud.npz", ASSET_PATH));
        let cloud_node = scene.add_gaussian_cloud("gaussian_cloud_lit", cloud_handle);
        scene.node(&cloud_node).set_rotation_euler(
            std::f32::consts::FRAC_PI_2,
            0.15,
            std::f32::consts::FRAC_PI_2,
        );

        let warm_light = scene.add_light(Light::new_point(Vec3::new(1.0, 0.82, 0.62), 55.0, 18.0));
        let cool_light = scene.add_light(Light::new_point(Vec3::new(0.52, 0.72, 1.0), 42.0, 16.0));
        scene.add_light(Light::new_directional(Vec3::splat(0.35), 0.9));

        let warm_orb = scene.add_mesh(Mesh::new(sphere_geometry, engine.assets.materials.add(orb_material.clone())));
        let cool_orb = scene.add_mesh(Mesh::new(sphere_geometry, engine.assets.materials.add(orb_material)));
        scene.node(&warm_orb).set_scale_xyz(0.12, 0.12, 0.12);
        scene.node(&cool_orb).set_scale_xyz(0.10, 0.10, 0.10);

        let camera_position = Vec3::new(0.0, 2.0, 6.0);
        let camera_target = Vec3::new(0.0, 0.5, 0.0);
        let camera = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&camera)
            .set_position(camera_position.x, camera_position.y, camera_position.z)
            .look_at(camera_target);
        scene.active_camera = Some(camera);

        Self {
            controls: OrbitControls::new(camera_position, camera_target),
            fps_counter: FpsCounter::new(),
            light_handles: [warm_light, cool_light],
            light_orbs: [warm_orb, cool_orb],
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.time += frame.dt;

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let warm_pos = Vec3::new(
            2.8 * (self.time * 0.7).cos(),
            1.2 + 0.3 * (self.time * 1.3).sin(),
            1.8 * (self.time * 0.7).sin(),
        );
        let cool_pos = Vec3::new(
            2.1 * (self.time * -0.9).cos(),
            0.9 + 0.25 * (self.time * 1.8).sin(),
            2.6 * (self.time * -0.9).sin(),
        );

        scene
            .node(&self.light_handles[0])
            .set_position(warm_pos.x, warm_pos.y, warm_pos.z);
        scene
            .node(&self.light_orbs[0])
            .set_position(warm_pos.x, warm_pos.y, warm_pos.z);
        scene
            .node(&self.light_handles[1])
            .set_position(cool_pos.x, cool_pos.y, cool_pos.z);
        scene
            .node(&self.light_orbs[1])
            .set_position(cool_pos.x, cool_pos.y, cool_pos.z);

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("3DGS Deferred Lighting | FPS: {:.2}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Myth Engine — 3DGS Deferred Lighting")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<GaussianSplattingLitDemo>()
}