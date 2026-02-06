use std::sync::Arc;

use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;
use winit::window::Window;

/// Earth Example
struct Earth {
    earth_node_id: NodeHandle,
    cloud_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for Earth {
    fn init(engine: &mut MythEngine, _window: &Arc<Window>) -> Self {
        // 1. Prepare resources
        let geometry = myth::create_sphere(&myth::resources::primitives::SphereOptions {
            radius: 63.71,
            width_segments: 100,
            height_segments: 50,
            ..Default::default()
        });

        let mut mat = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // Load textures
        let earth_tex_handle = engine
            .assets
            .load_texture(
                "examples/assets/planets/earth_atmos_4096.jpg",
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load earth texture");
        let specular_tex_handle = engine
            .assets
            .load_texture(
                "examples/assets/planets/earth_specular_2048.jpg",
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load specular texture");
        let emssive_tex_handle = engine
            .assets
            .load_texture(
                "examples/assets/planets/earth_lights_2048.png",
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load emissive texture");
        let normal_map_handle = engine
            .assets
            .load_texture(
                "examples/assets/planets/earth_normal_2048.jpg",
                ColorSpace::Linear,
                true,
            )
            .expect("Failed to load normal map");
        let clouds_tex_handle = engine
            .assets
            .load_texture(
                "examples/assets/planets/earth_clouds_1024.png",
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load clouds texture");

        if let Some(phong) = mat.as_phong_mut() {
            phong.set_map(Some(earth_tex_handle));
            phong.set_specular_map(Some(specular_tex_handle));
            phong.set_emissive_map(Some(emssive_tex_handle));
            phong.set_normal_map(Some(normal_map_handle));

            phong.set_normal_scale(Vec2::new(0.85, -0.85));
            phong.set_shininess(10.0);
            phong.set_emissive(Vec3::new(0.0962, 0.0962, 0.0512));
            phong.set_emissive_intensity(3.0);
        }

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(mat);

        // Cloud layer material
        let mut cloud_material = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));
        if let Some(phong) = cloud_material.as_phong_mut() {
            phong.set_map(Some(clouds_tex_handle));
            phong.set_opacity(0.8);
            phong.set_alpha_mode(AlphaMode::Blend);
            phong.set_depth_write(false);
            phong.set_side(Side::Front);
        }
        let cloud_material_handle = engine.assets.materials.add(cloud_material);

        // 2. Create Meshes and add to scene
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cloud_mesh = Mesh::new(geo_handle, cloud_material_handle);

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        let earth_node_id = scene.add_mesh(mesh);
        if let Some(earth) = scene.get_node_mut(earth_node_id) {
            earth.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, -1.0, 0.0);
        }

        let cloud_node_id = scene.add_mesh(cloud_mesh);
        if let Some(clouds) = scene.get_node_mut(cloud_node_id) {
            clouds.transform.scale = Vec3::splat(1.005);
            clouds.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.0, 0.41);
        }

        // 3. Add Sun Light
        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        let light_handle = scene.add_light(light);
        scene
            .environment
            .set_ambient_color(Vec3::new(0.0001, 0.0001, 0.0001));

        if let Some(light_node) = scene.get_node_mut(light_handle) {
            light_node.transform.position = Vec3::new(3.0, 0.0, 1.0);
            light_node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 4. Setup Camera
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 250.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            earth_node_id,
            cloud_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 250.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut MythEngine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.001 * 60.0 * frame.dt, 0.0);
        let rot_clouds = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.00125 * 60.0 * frame.dt, 0.0);

        // Earth self-rotation
        if let Some(node) = scene.get_node_mut(self.earth_node_id) {
            node.transform.rotation = rot * node.transform.rotation;
        }

        // Cloud layer self-rotation
        if let Some(clouds) = scene.get_node_mut(self.cloud_node_id) {
            clouds.transform.rotation = rot_clouds * clouds.transform.rotation;
        }

        // Orbit controls
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        // FPS Display
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Earth | FPS: {:.2}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_title("Earth")
        .with_settings(RenderSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<Earth>()
}
