//! Morph Target (Face Cap) 测试用例
//! 
//! 演示 Morph Target 动画功能，使用 facecap.glb 模型

use glam::Vec3;
use three::app::App;
use three::scene::Camera;
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::assets::loaders::gltf::AnimationProperty;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_settings(three::renderer::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    // 2. 添加灯光
    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 2.0);
    app.scene.add_light(light);
    
    // 添加环境光
    app.scene.environment.uniforms_mut().ambient_light = Vec3::splat(0.3);

    // 加载环境贴图
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

    // 3. 加载 facecap.glb 模型
    let gltf_path = std::path::Path::new("examples/assets/facecap.glb");
    println!("Loading glTF model from: {}", gltf_path.display());
    
    let (loaded_nodes, animations) = GltfLoader::load(
        gltf_path,
        &mut app.assets,
        &mut app.scene
    )?;

    println!("Successfully loaded {} root nodes", loaded_nodes.len());
    println!("Total animations found: {}", animations.len());

    // 打印动画信息
    for (i, anim) in animations.iter().enumerate() {
        println!("Animation {}: '{}' with {} channels", i, anim.name, anim.channels.len());
        
        // 统计 morph target 通道
        let morph_channels: Vec<_> = anim.channels.iter()
            .filter(|c| c.property == AnimationProperty::MorphTargetWeights)
            .collect();
        
        if !morph_channels.is_empty() {
            println!("  - Found {} morph target weight channels", morph_channels.len());
            for channel in &morph_channels {
                println!("    Node: {:?}, {} keyframes, {} weight values", 
                    channel.node_index, 
                    channel.inputs.len(),
                    channel.outputs.len()
                );
            }
        }
    }

    // 4. 打印 Mesh 的 morph target 信息 (已由 glTF 加载器自动初始化)
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

    // 5. 准备动画测试数据
    let active_animation = animations.first().cloned();
    let mut anim_time = 0.0f32;

    // 6. 设置相机
    let camera = Camera::new_perspective(
        45.0,
        1280.0 / 720.0,
        0.1,
        100.0
    );

    let cam_node_id = app.scene.add_camera(camera);
    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.transform.position = Vec3::new(0.0, 0.0, 4.0);
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    app.scene.active_camera = Some(cam_node_id);

    // 7. 轨道控制器
    let mut controls = OrbitControls::new(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO);
    let mut fps_counter = FpsCounter::new();

    // 8. 更新回调
    app.set_update_fn(move |window, scene, _assets, input, _time, dt| {
        
        // === 动画播放逻辑 ===
        if let Some(anim) = &active_animation {
            // 1. 计算循环时间
            let duration = anim.channels.first()
                .and_then(|c| c.inputs.last())
                .copied()
                .unwrap_or(1.0);
            
            anim_time += dt;
            if anim_time > duration {
                anim_time %= duration;
            }

            // 2. 遍历所有通道并应用数据
            for channel in &anim.channels {
                if channel.inputs.is_empty() { continue; }

                // --- 查找关键帧 ---
                let mut frame_idx = 0;
                for (i, time) in channel.inputs.iter().enumerate() {
                    if *time > anim_time {
                        break;
                    }
                    frame_idx = i;
                }
                
                let next_frame_idx = if frame_idx + 1 < channel.inputs.len() {
                    frame_idx + 1
                } else {
                    frame_idx
                };

                // --- 计算插值因子 t ---
                let t0 = channel.inputs[frame_idx];
                let t1 = channel.inputs[next_frame_idx];
                let factor = if t1 > t0 {
                    (anim_time - t0) / (t1 - t0)
                } else {
                    0.0
                };

                // --- 处理 Morph Target Weights ---
                if channel.property == AnimationProperty::MorphTargetWeights {
                    // 获取节点对应的 Mesh
                    if let Some(node) = scene.get_node(channel.node_index) {
                        if let Some(mesh_key) = node.mesh {
                            if let Some(mesh) = scene.meshes.get_mut(mesh_key) {
                                // 计算每帧的权重数量
                                let num_targets = mesh.morph_target_influences.len();
                                if num_targets == 0 { continue; }
                                
                                let weights_per_frame = num_targets;
                                let idx0 = frame_idx * weights_per_frame;
                                let idx1 = next_frame_idx * weights_per_frame;
                                
                                // 确保索引不越界
                                if idx0 + weights_per_frame <= channel.outputs.len() 
                                    && idx1 + weights_per_frame <= channel.outputs.len() 
                                {
                                    // 线性插值权重
                                    for i in 0..weights_per_frame {
                                        let w0 = channel.outputs[idx0 + i];
                                        let w1 = channel.outputs[idx1 + i];
                                        let weight = w0 + (w1 - w0) * factor;
                                        mesh.set_morph_target_influence(i, weight);
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }

        // === 轨道控制器更新 ===
        if let Some(cam_idx) = scene.active_camera {
            if let Some(cam_node) = scene.get_node_mut(cam_idx) {
                controls.update(&mut cam_node.transform, input, 45.0, dt);
            }
        }

        // === FPS 显示 ===
        if let Some(fps) = fps_counter.update() {
            window.set_title(&format!("Morph Target Demo - FPS: {:.1}", fps));
        }
    });

    // 9. 运行
    app.run()
}
