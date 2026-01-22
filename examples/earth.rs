use glam::{Vec2, Vec3, Vec4, Quat};
use three::app::{App, AppContext, AppHandler};
use three::resources::Material;
use three::scene::{Camera, NodeHandle, light};
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::renderer::settings::RenderSettings;

/// 地球渲染示例
struct Earth {
    earth_node_id: NodeHandle,
    cloud_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for Earth {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 准备资源
        let geometry = three::create_sphere(three::resources::primitives::SphereOptions {
            radius: 63.71,
            width_segments: 100,
            height_segments: 50,
            ..Default::default()
        });

        let mut mat = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // 加载纹理
        let earth_tex_handle = ctx.assets.load_texture_from_file(
            "examples/assets/planets/earth_atmos_4096.jpg", three::ColorSpace::Srgb
        ).expect("Failed to load earth texture");
        let specular_tex_handle = ctx.assets.load_texture_from_file(
            "examples/assets/planets/earth_specular_2048.jpg", three::ColorSpace::Srgb
        ).expect("Failed to load specular texture");
        let emssive_tex_handle = ctx.assets.load_texture_from_file(
            "examples/assets/planets/earth_lights_2048.png", three::ColorSpace::Srgb
        ).expect("Failed to load emissive texture");
        let normal_map_handle = ctx.assets.load_texture_from_file(
            "examples/assets/planets/earth_normal_2048.jpg", three::ColorSpace::Linear
        ).expect("Failed to load normal map");
        let clouds_tex_handle = ctx.assets.load_texture_from_file(
            "examples/assets/planets/earth_clouds_1024.png", three::ColorSpace::Srgb
        ).expect("Failed to load clouds texture");

        if let Some(phong) = mat.as_phong_mut() {
            {
                let bindings = phong.bindings_mut();
                bindings.map = Some(earth_tex_handle.into());
                bindings.specular_map = Some(specular_tex_handle.into());
                bindings.emissive_map = Some(emssive_tex_handle.into());
                bindings.normal_map = Some(normal_map_handle.into());
            }

            let mut uniforms = phong.uniforms_mut();
            uniforms.normal_scale = Vec2::new(0.85, -0.85);
            uniforms.shininess = 10.0;
            uniforms.emissive = Vec3::new(0.0962, 0.0962, 0.0512);
            uniforms.emissive_intensity = 3.0;
        }
            
        let geo_handle = ctx.assets.add_geometry(geometry);
        let mat_handle = ctx.assets.add_material(mat);

        // 云层材质
        let mut cloud_material = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));
        if let Some(phong) = cloud_material.as_phong_mut() {
            phong.set_map(Some(clouds_tex_handle.into()));
            phong.uniforms_mut().opacity = 0.8;
            
            {
                let mut settings = phong.settings_mut();
                settings.transparent = true;
                settings.depth_write = false;
                settings.side = three::Side::Front;
            }
        }
        let cloud_material_handle = ctx.assets.add_material(cloud_material);

        // 2. 创建 Mesh 并加入场景
        let mesh = three::resources::Mesh::new(geo_handle, mat_handle);
        let cloud_mesh = three::resources::Mesh::new(geo_handle, cloud_material_handle);

        let earth_node_id = ctx.scene.add_mesh(mesh);
        if let Some(earth) = ctx.scene.get_node_mut(earth_node_id) {
            earth.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, -1.0, 0.0);
        }

        let cloud_node_id = ctx.scene.add_mesh(cloud_mesh);
        if let Some(clouds) = ctx.scene.get_node_mut(cloud_node_id) {
            clouds.transform.scale = Vec3::splat(1.005);
            clouds.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.0, 0.41);
        }

        // 3. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        let light_index = ctx.scene.add_light(light);
        ctx.scene.environment.set_ambient_color(Vec3::new(0.0001, 0.0001, 0.0001));

        if let Some(light_node) = ctx.scene.get_node_mut(light_index) {
            light_node.transform.position = Vec3::new(3.0, 0.0, 1.0);
            light_node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = ctx.scene.add_camera(camera);
        
        if let Some(node) = ctx.scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 250.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        ctx.scene.active_camera = Some(cam_node_id);

        Self {
            earth_node_id,
            cloud_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 250.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.001 * 60.0 * ctx.dt, 0.0);
        let rot_clouds = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.00125 * 60.0 * ctx.dt, 0.0);

        // 地球自转
        if let Some(node) = ctx.scene.get_node_mut(self.earth_node_id) {
            node.transform.rotation = rot * node.transform.rotation;
        }
        
        // 云层自转
        if let Some(clouds) = ctx.scene.get_node_mut(self.cloud_node_id) {
            clouds.transform.rotation = rot_clouds * clouds.transform.rotation;
        }

        // 轨道控制器
        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            ctx.window.set_title(&format!("Earth | FPS: {:.2}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_title("Earth")
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<Earth>()
}