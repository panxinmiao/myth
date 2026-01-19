use glam::{Vec2, Vec3, Vec4, Quat};
use three::app::App;
use three::resources::{Material, Mesh};
use three::scene::{Camera};
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::{FpsCounter};
use three::renderer::settings::RenderSettings;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_title("Earth").with_settings(RenderSettings {
        vsync: false,
        ..Default::default()
    });

    let mut fps_counter = FpsCounter::new();

    // 2. 准备资源 (Geometry, Texture, Material)
    let geometry = three::create_sphere(three::resources::primitives::SphereOptions {
        radius: 63.71,
        width_segments: 100,
        height_segments: 50,
        ..Default::default()
    });
    // let geometry = Geometry::new_box(2.0, 2.0, 2.0);
    let mut mat = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));

    let earth_tex_handle = app.assets.load_texture_from_file("examples/assets/planets/earth_atmos_4096.jpg", three::ColorSpace::Srgb)?;
    let specular_tex_handle = app.assets.load_texture_from_file("examples/assets/planets/earth_specular_2048.jpg", three::ColorSpace::Srgb)?;
    let emssive_tex_handle = app.assets.load_texture_from_file("examples/assets/planets/earth_lights_2048.png", three::ColorSpace::Srgb)?;
    let normal_map_handle = app.assets.load_texture_from_file("examples/assets/planets/earth_normal_2048.jpg", three::ColorSpace::Linear)?;
    let clouds_tex_handle = app.assets.load_texture_from_file("examples/assets/planets/earth_clouds_1024.png", three::ColorSpace::Srgb)?;
    if let Some(phong) = mat.as_phong_mut() {

        {
            let mut bindings = phong.bindings_mut();
            bindings.map = Some(earth_tex_handle);
            bindings.specular_map = Some(specular_tex_handle);
            bindings.emissive_map = Some(emssive_tex_handle);
            bindings.normal_map = Some(normal_map_handle);
        }

        let mut uniforms = phong.uniforms_mut();
        uniforms.normal_scale = Vec2::new(0.85, -0.85);
        uniforms.shininess = 10.0;
        uniforms.emissive = Vec3::new(0.0962, 0.0962, 0.0512);
        uniforms.emissive_intensity = 3.0;
    }
        
    
    let geo_handle = app.assets.add_geometry(geometry);
    let mat_handle = app.assets.add_material(mat.into());

    let mut cloud_material = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));
    if let Some(phong) = cloud_material.as_phong_mut() {
        phong.set_map(Some(clouds_tex_handle));
        phong.uniforms_mut().opacity = 0.8;
        
        {
            let mut settings = phong.settings_mut();
            settings.transparent = true;
            settings.depth_write = false;
            settings.side = three::Side::Front;
        }
    }

    let cloud_material_handle = app.assets.add_material(cloud_material.into());

    // 4. 创建 Mesh 并加入场景
    let mesh = Mesh::new(geo_handle, mat_handle);
    let cloud_mesh = Mesh::new(geo_handle, cloud_material_handle);

    let earth_node_id = app.scene.add_mesh(mesh);

    if let Some(earth) = app.scene.get_node_mut(earth_node_id) {
        earth.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, -1.0, 0.0);
    }

    let cloud_node_id = app.scene.add_mesh(cloud_mesh);
    if let Some(clouds) = app.scene.get_node_mut(cloud_node_id) {
        clouds.transform.scale = Vec3::splat(1.005);
        clouds.transform.rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.0, 0.41);
    }

    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);

    let light_index = app.scene.add_light(light);

    app.scene.environment.set_ambient_color(Vec3::new(0.0001, 0.0001, 0.0001));

    if let Some(light_node) = app.scene.get_node_mut(light_index) {
        light_node.transform.position = Vec3::new(3.0, 0.0, 1.0);
        light_node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }

    // 5. 设置相机
    let camera = Camera::new_perspective(
        45.0, 
        1280.0 / 720.0, // 默认窗口大小的长宽比
        0.1, 
        1000.0
    );
    
    // 5.2 将相机加入场景 (自动创建 Node)
    let cam_node_id = app.scene.add_camera(camera);
    
    // 5.3 设置相机节点的位置和朝向
    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.transform.position = Vec3::new(0.0, 0.0, 250.0);
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    
    // 5.4 激活相机
    app.scene.active_camera = Some(cam_node_id);


    let mut controls = OrbitControls::new(Vec3::new(0.0, 0.0, 250.0), Vec3::ZERO);

    app.set_update_fn(move |ctx| {

        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.001 * 60.0 * ctx.dt, 0.0);
        let rot_clouds = Quat::from_euler(glam::EulerRot::XYZ, 0.0, 0.00125 * 60.0 * ctx.dt, 0.0);

        // 1. 地球自转
        if let Some(node) = ctx.scene.get_node_mut(earth_node_id) {
            node.transform.rotation = rot * node.transform.rotation;
        }
        // 2. 云层自转
        if let Some(clouds) = ctx.scene.get_node_mut(cloud_node_id) {
            clouds.transform.rotation = rot_clouds * clouds.transform.rotation;
        }

        // 3. 相机控制
        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        if let Some(fps) = fps_counter.update() {
            let title = format!("Earth | FPS: {:.2}", fps);
            ctx.window.set_title(&title);
        }
    });

    // 7. 运行
    app.run()?;
    Ok(())
}