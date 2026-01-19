use std::sync::Arc;
use glam::Vec3;
use three::app::App;
use three::scene::Camera;
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::{AnimationMixer, AnimationAction, Binder};
use three::animation::binding::TargetPath;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut app = App::new().with_settings(three::renderer::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 2.0);
    app.scene.add_light(light);
    app.scene.environment.set_ambient_color(Vec3::splat(0.3));

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

    let gltf_path = std::path::Path::new("examples/assets/facecap.glb");
    println!("Loading glTF model from: {}", gltf_path.display());
    
    let (loaded_nodes, animations) = GltfLoader::load(
        gltf_path,
        &mut app.assets,
        &mut app.scene
    )?;

    println!("Successfully loaded {} root nodes", loaded_nodes.len());
    println!("Total animations found: {}", animations.len());

    for (i, anim) in animations.iter().enumerate() {
        println!("Animation {}: '{}' with {} tracks", i, anim.name, anim.tracks.len());
        
        let morph_tracks: Vec<_> = anim.tracks.iter()
            .filter(|t| t.meta.target == TargetPath::Weights)
            .collect();
        
        if !morph_tracks.is_empty() {
            println!("  - Found {} morph target weight tracks", morph_tracks.len());
        }
    }

    for (mesh_key, mesh) in app.scene.meshes.iter() {
        if let Some(geometry) = app.assets.get_geometry(mesh.geometry) {
            if geometry.has_morph_targets() {
                println!("Mesh {:?} has {} morph targets, {} vertices per target",
                    mesh_key,
                    geometry.morph_target_count,
                    geometry.morph_vertex_count
                );
            }
        }
    }

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
        node.transform.position = Vec3::new(0.0, 0.0, 4.0);
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    app.scene.active_camera = Some(cam_node_id);

    let mut controls = OrbitControls::new(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO);
    let mut fps_counter = FpsCounter::new();

    app.set_update_fn(move |window, scene, _assets, input, _time, dt| {
        mixer.update(dt, scene);

        if let Some(cam_idx) = scene.active_camera {
            if let Some(cam_node) = scene.get_node_mut(cam_idx) {
                controls.update(&mut cam_node.transform, input, 45.0, dt);
            }
        }

        if let Some(fps) = fps_counter.update() {
            window.set_title(&format!("Morph Target Demo - FPS: {:.1}", fps));
        }
    });

    app.run()
}
