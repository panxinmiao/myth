use glam::{Vec3, Quat}; // 需要引入 Quat
use three::app::App;
use three::scene::Camera;
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::assets::loaders::gltf::AnimationProperty; // 确保能引用到这个枚举

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_settings(three::renderer::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    // 2. 加载环境贴图
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

    // 3. 添加灯光
    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
    app.scene.add_light(light);

    // 4. 加载 Michelle.glb 模型
    let gltf_path = std::path::Path::new("examples/assets/Michelle.glb");
    println!("Loading glTF model from: {}", gltf_path.display());
    
    // 注意：这里我们需要接收 animations
    let (loaded_nodes, animations) = GltfLoader::load(
        gltf_path,
        &mut app.assets,
        &mut app.scene
    )?;

    println!("Successfully loaded {} root nodes", loaded_nodes.len());
    println!("Total animations found: {}", animations.len());

    // 5. 准备动画测试数据
    // 我们取第一个动画进行播放
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
        node.transform.position = Vec3::new(0.0, 1.0, 3.0);
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    app.scene.active_camera = Some(cam_node_id);

    // 7. 轨道控制器
    let mut controls = OrbitControls::new(Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 1.0, 0.0));
    let mut fps_counter = FpsCounter::new();

    // 8. 更新回调
    // move 关键字将 active_animation 和 anim_time 移入闭包所有权
    app.set_update_fn(move |window, scene, _assets, input, _time, dt| {
        
        // === 动画播放逻辑 (Simple Hardcoded Interpolator) ===
        if let Some(anim) = &active_animation {
            // 1. 计算循环时间
            // 假设所有通道时长一致，取第一个通道的最后时间作为总时长
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

                // --- 查找关键帧 (Linear Search) ---
                // 找到 i，使得 inputs[i] <= anim_time < inputs[i+1]
                let mut frame_idx = 0;
                for (i, time) in channel.inputs.iter().enumerate() {
                    if *time > anim_time {
                        break;
                    }
                    frame_idx = i;
                }
                
                // 防止越界 (如果是最后一帧，保持最后一帧)
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

                // --- 获取节点并修改 Transform ---
                if let Some(node) = scene.get_node_mut(channel.node_index) {
                    match channel.property {
                        AnimationProperty::Translation => {
                            // 从 flat array 中读取 Vec3 (stride = 3)
                            let idx0 = frame_idx * 3;
                            let idx1 = next_frame_idx * 3;
                            
                            let v0 = Vec3::from_slice(&channel.outputs[idx0..idx0+3]);
                            let v1 = Vec3::from_slice(&channel.outputs[idx1..idx1+3]);
                            
                            // 线性插值
                            node.transform.position = v0.lerp(v1, factor);
                        },
                        AnimationProperty::Rotation => {
                            // 从 flat array 中读取 Quat (stride = 4)
                            let idx0 = frame_idx * 4;
                            let idx1 = next_frame_idx * 4;
                            
                            let q0 = Quat::from_slice(&channel.outputs[idx0..idx0+4]);
                            let q1 = Quat::from_slice(&channel.outputs[idx1..idx1+4]);
                            
                            // 球面线性插值 (Slerp) - 骨骼旋转必须用这个
                            node.transform.rotation = q0.slerp(q1, factor).normalize();
                        },
                        AnimationProperty::Scale => {
                            let idx0 = frame_idx * 3;
                            let idx1 = next_frame_idx * 3;
                            
                            let v0 = Vec3::from_slice(&channel.outputs[idx0..idx0+3]);
                            let v1 = Vec3::from_slice(&channel.outputs[idx1..idx1+3]);
                            
                            node.transform.scale = v0.lerp(v1, factor);
                        },
                        _ => {}
                    }
                }
            }
        }
        // =================================================

        // 更新轨道控制器
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            controls.update(transform, input, camera.fov.to_degrees(), dt);
        }

        // 显示 FPS
        if let Some(fps) = fps_counter.update() {
            let title = format!("Michelle glTF (Animation Test) | FPS: {:.2}", fps);
            window.set_title(&title);
        }
    });

    println!("Starting render loop...");
    app.run()?;
    Ok(())
}