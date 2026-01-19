use std::sync::Arc;
use glam::Vec3;
use three::app::App;
use three::scene::Camera;
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::{AnimationMixer, AnimationAction, Binder};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut app = App::new().with_settings(three::renderer::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    let env_texture_handle = app.assets.load_cube_texture_from_files(
        [
            "examples/assets/Park2/posx.jpg",
            "examples/assets/Park2/negx.jpg",
            "examples/assets/Park2/posy.jpg",
            "examples/assets/Park2/negy.jpg",
            "examples/assets/Park2/posz.jpg",
            "examples/assets/Park2/negz.jpg",
        ],
        three::ColorSpace::Srgb
    )?;

    let env_texture = app.assets.get_texture_mut(env_texture_handle).unwrap();
    env_texture.generate_mipmaps = true;
    app.scene.environment.set_env_map(Some((env_texture_handle, &env_texture)));

    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
    app.scene.add_light(light);

    let gltf_path = std::path::Path::new("D:\\glTF\\girl\\tifa_piss.glb");
    println!("Loading glTF model from: {}", gltf_path.display());
    
    let (loaded_nodes, animations) = GltfLoader::load(
        gltf_path,
        &mut app.assets,
        &mut app.scene
    )?;

    println!("Successfully loaded {} root nodes", loaded_nodes.len());
    println!("Total animations found: {}", animations.len());

    let mut mixer = AnimationMixer::new();
    
    if let Some(clip) = animations.into_iter().next() {
        println!("Playing animation: {} (duration: {:.2}s)", clip.name, clip.duration);
        
        let root_node = loaded_nodes.first().copied().unwrap();
        let clip = Arc::new(clip);
        let bindings = Binder::bind(&app.scene, root_node, &clip);
        
        println!("Created {} bindings", bindings.len());
        
        let mut action = AnimationAction::new(clip);
        action.bindings = bindings;
        mixer.add_action(action);
    }

    let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 100.0);

    let cam_node_id = app.scene.add_camera(camera);
    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.transform.position = Vec3::new(0.0, 1.0, 3.0);
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    app.scene.active_camera = Some(cam_node_id);

    let mut controls = OrbitControls::new(Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 1.0, 0.0));
    let mut fps_counter = FpsCounter::new();

    app.set_update_fn(move |ctx| {
        mixer.update(ctx.dt, ctx.scene);

        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        if let Some(fps) = fps_counter.update() {
            let title = format!("Skinning Animation | FPS: {:.2}", fps);
            ctx.window.set_title(&title);
        }
    });

    println!("Starting render loop...");
    app.run()?;
    Ok(())
}