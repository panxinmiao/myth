use std::collections::HashMap;
use std::path::Path;
use std::fs;
use glam::{Vec3, Vec4, Quat, Mat4, Affine3A};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::material::{Material, MeshStandardMaterial};
use crate::resources::texture::Texture;
use crate::scene::{Scene, Node, NodeIndex, SkeletonKey};
use crate::scene::skeleton::{Skeleton, BindMode};
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use crate::animation::clip::{AnimationClip, Track, TrackMeta, TrackData, MorphWeightsTrack};
use crate::animation::tracks::{KeyframeTrack, InterpolationMode};
use crate::animation::binding::TargetPath;
use wgpu::{VertexFormat, TextureFormat};
use anyhow::Context;
use serde_json::Value;

// ============================================================================
// 1. 插件化架构定义 (Plugin Architecture)
// ============================================================================

/// glTF 加载扩展 Trait
/// Plugin 机制，允许拦截加载过程的不同阶段
/// 

pub struct LoadContext<'a> {
    pub assets: &'a mut AssetServer,
    pub texture_map: &'a [TextureHandle],
    pub material_map: &'a [MaterialHandle],
    // ...
}

pub trait GltfExtensionParser {
    fn name(&self) -> &str;

    /// 当材质被加载时调用
    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, _engine_mat: &mut Material, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    /// 当纹理被加载时调用
    fn on_load_texture(&mut self, _ctx: &mut LoadContext, _gltf_tex: &gltf::Texture, _engine_tex: &mut Texture, _extension_value: &serde_json::Value) -> anyhow::Result<()> {
        Ok(())
    }

    /// 当节点被创建时调用 (可用于处理 KHR_lights_punctual 等挂载到节点的扩展)
    fn on_load_node(&mut self, _ctx: &mut LoadContext, _gltf_node: &gltf::Node, _engine_node: &mut Node, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }
}

// ============================================================================
// 2. GltfLoader 实现
// ============================================================================

pub struct GltfLoader<'a> {
    assets: &'a mut AssetServer,
    scene: &'a mut Scene,
    base_path: std::path::PathBuf,
    
    // 资源映射表
    texture_map: Vec<TextureHandle>,
    material_map: Vec<MaterialHandle>,
    // glTF Node Index -> Engine Node Index
    node_mapping: Vec<NodeIndex>,

    // 扩展列表
    extensions: HashMap<String, Box<dyn GltfExtensionParser>>,
}

impl<'a> GltfLoader<'a> {
    /// 入口函数
    pub fn load(
        path: &Path,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<(Vec<NodeIndex>, Vec<AnimationClip>)> {
        
        // 1. 读取文件和 Buffer
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open glTF file: {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader(reader)
            .context("Failed to parse glTF file")?;

        let base_path = path.parent().unwrap_or(Path::new("./")).to_path_buf();
        let buffers = Self::load_buffers(&gltf, &base_path)?;

        // 2. 初始化加载器
        let mut loader = Self {
            assets,
            scene,
            base_path,
            texture_map: Vec::new(),
            material_map: Vec::new(),
            node_mapping: Vec::with_capacity(gltf.nodes().count()),
            extensions: HashMap::new(),
        };

        // 注册默认扩展 (可以在这里添加更多)
        // loader.register_extension(Box::new(KhrMaterialsUnlit));

        // 3. 资源加载流程
        loader.load_textures(&gltf, &buffers)?;
        loader.load_materials(&gltf)?;

        // 4. 节点处理流程 (四步走)
        
        // Step 4.1: 创建所有节点 (Node & Transform)
        for node in gltf.nodes() {
            let idx = loader.create_node_shallow(&node)?;
            loader.node_mapping.push(idx);
        }

        // Step 4.2: 建立层级关系 (Hierarchy)
        let mut root_indices = Vec::new();
        for node in gltf.nodes() {
            let parent_idx = loader.node_mapping[node.index()];
            
            if node.children().len() > 0 {
                for child in node.children() {
                    let child_idx = loader.node_mapping[child.index()];
                    loader.scene.attach(child_idx, parent_idx);
                }
            }
        }
        
        // 找出场景根节点
        if let Some(default_scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            for node in default_scene.nodes() {
                root_indices.push(loader.node_mapping[node.index()]);
            }
        }

        // Step 4.3: 加载骨架 (Skins) - 此时所有 Node 已存在，可以安全引用的
        let skeleton_keys = loader.load_skins(&gltf, &buffers)?;

        // Step 4.4: 绑定 Mesh 和 Skin
        for node in gltf.nodes() {
            loader.bind_node_mesh_and_skin(&node, &buffers, &skeleton_keys)?;
        }

        // 5. 加载动画数据 (可选)
        let animations = loader.load_animations(&gltf, &buffers)?;

        Ok((root_indices, animations))
    }

    fn _register_extension(&mut self, ext: Box<dyn GltfExtensionParser>) {
        self.extensions.insert(ext.name().to_string(), ext);
    }

    // --- Helpers ---

    fn load_buffers(gltf: &gltf::Gltf, base_path: &Path) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut buffer_data = Vec::new();
        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Bin => {
                    if let Some(blob) = gltf.blob.as_deref() {
                        buffer_data.push(blob.to_vec());
                    } else {
                        return Err(anyhow::anyhow!("Missing GLB binary chunk"));
                    }
                }
                gltf::buffer::Source::Uri(uri) => {
                    let buffer_path = base_path.join(uri);
                    let data = fs::read(&buffer_path)
                        .with_context(|| format!("Failed to read buffer file: {}", buffer_path.display()))?;
                    buffer_data.push(data);
                }
            }
        }
        Ok(buffer_data)
    }

    // --- Loading Logic ---

    fn load_textures(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<()> {
        for texture in gltf.textures() {
            let img = texture.source();
            let img_data = match img.source() {
                gltf::image::Source::Uri { uri, .. } => {
                    let path = self.base_path.join(uri);
                    image::open(&path)
                        .with_context(|| format!("Failed to load image: {}", path.display()))?
                        .to_rgba8()
                },
                gltf::image::Source::View { view, .. } => {
                    let start = view.offset();
                    let end = start + view.length();
                    let bytes = &buffers[view.buffer().index()][start..end];
                    image::load_from_memory(bytes)
                        .context("Failed to load embedded image")?
                        .to_rgba8()
                }
            };

            let width = img_data.width();
            let height = img_data.height();
            let tex_name = texture.name().unwrap_or("gltf_texture");
            
            let mut engine_tex = Texture::new_2d(
                Some(tex_name),
                width,
                height,
                Some(img_data.into_vec()),
                TextureFormat::Rgba8Unorm
            );

            if let Some(extensions_map) = texture.extensions() {

                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    texture_map: &self.texture_map, 
                    material_map: &self.material_map,
                };

                for (name, value) in extensions_map {
                    // 只调用匹配的插件
                    if let Some(handler) = self.extensions.get_mut(name) {
                        handler.on_load_texture(&mut ctx, &texture, &mut engine_tex, value)?;
                    }
                }
            }

            let handle = self.assets.add_texture(engine_tex);
            self.texture_map.push(handle);
        }
        Ok(())
    }

    fn load_materials(&mut self, gltf: &gltf::Gltf) -> anyhow::Result<()> {
        for material in gltf.materials() {
            let pbr = material.pbr_metallic_roughness();

            let base_color_factor = Vec4::from_array(pbr.base_color_factor());
            let mut mat = MeshStandardMaterial::new(base_color_factor);

            mat.set_metalness(pbr.metallic_factor());
            mat.set_roughness(pbr.roughness_factor());

            if let Some(info) = pbr.base_color_texture() {
                let tex_handle = self.texture_map[info.texture().index()];
                mat.set_map(Some(tex_handle));
                
                if let Some(texture) = self.assets.get_texture(tex_handle) {
                    texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
                }
            }

            if let Some(info) = pbr.metallic_roughness_texture() {
                let tex_handle = self.texture_map[info.texture().index()];
                mat.set_roughness_map(Some(tex_handle));
                mat.set_metalness_map(Some(tex_handle));
            }

            if let Some(info) = material.normal_texture() {
                let tex_handle = self.texture_map[info.texture().index()];
                mat.set_normal_map(Some(tex_handle));
            }

            // 转换为通用 Material 枚举
            let mut engine_mat = Material::from(mat);

            // extensions 处理
            if let Some(extensions_map) = material.extensions() {
                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    texture_map: &self.texture_map, 
                    material_map: &self.material_map,
                };

                for (name, value) in extensions_map {
                    // 只调用匹配的插件
                    if let Some(handler) = self.extensions.get_mut(name) {
                        handler.on_load_material(&mut ctx, &material, &mut engine_mat, value)?;
                    }
                }
            }

            let handle = self.assets.add_material(engine_mat);
            self.material_map.push(handle);
        }
        Ok(())
    }

    // 仅创建节点和 Transform
    fn create_node_shallow(&mut self, node: &gltf::Node) -> anyhow::Result<NodeIndex> {
        let mut engine_node = Node::new(node.name().unwrap_or("Node"));

        let (t, r, s) = node.transform().decomposed();
        engine_node.transform.position = Vec3::from_array(t);
        engine_node.transform.rotation = Quat::from_array(r);
        engine_node.transform.scale = Vec3::from_array(s);


        if let Some(extensions_map) = node.extensions() {
            let mut ctx = LoadContext {
                assets: &mut self.assets,       // 可变借用 assets
                texture_map: &self.texture_map, // 不可变借用 texture_map
                material_map: &self.material_map,
            };
            // 只遍历该节点实际拥有的扩展
            for (name, value) in extensions_map {
                if let Some(handler) = self.extensions.get_mut(name) {
                    handler.on_load_node(&mut ctx, &node, &mut engine_node, value)?;
                }
            }
        }
        Ok(self.scene.add_node(engine_node))
    }

    fn load_skins(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<Vec<SkeletonKey>> {
        let mut skeleton_keys = Vec::new();

        for skin in gltf.skins() {
            let name = skin.name().unwrap_or("Skeleton");
            
            // 1. 读取 Inverse Bind Matrices (IBM)
            let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
            let ibms: Vec<Affine3A> = if let Some(iter) = reader.read_inverse_bind_matrices() {
                iter.map(|m| {
                    let mat = Mat4::from_cols_array_2d(&m);
                    Affine3A::from_mat4(mat)
                }).collect()
            } else {
                // 默认单位矩阵
                vec![Affine3A::IDENTITY; skin.joints().count()]
            };

            // 2. 映射 glTF joint indices 到 Engine Node Indices
            let bones: Vec<NodeIndex> = skin.joints()
                .map(|node| self.node_mapping[node.index()]) 
                .collect();

            // 3. 创建资源
            let skeleton = Skeleton::new(name, bones, ibms);
    
            let key = self.scene.skins.insert(skeleton); 
            skeleton_keys.push(key);
        }

        Ok(skeleton_keys)
    }

    fn bind_node_mesh_and_skin(
        &mut self,
        node: &gltf::Node,
        buffers: &[Vec<u8>],
        skeleton_keys: &[SkeletonKey]
    ) -> anyhow::Result<()> {
        let engine_node_idx = self.node_mapping[node.index()];

        // 1. 加载 Mesh
        if let Some(mesh) = node.mesh() {
            for (i, primitive) in mesh.primitives().enumerate() {
                let geo_handle = self.load_primitive_geometry(&primitive, buffers)?;
                
                let mat_idx = primitive.material().index();
                let mat_handle = if let Some(idx) = mat_idx {
                    self.material_map[idx]
                } else {
                    self.assets.add_material(Material::new_standard(Vec4::ONE))
                };

                let mut engine_mesh = crate::resources::mesh::Mesh::new(geo_handle, mat_handle);
                
                // 如果 geometry 有 morph targets，初始化 mesh 的 morph target influences
                if let Some(geometry) = self.assets.get_geometry(geo_handle) {
                    if geometry.has_morph_targets() {
                        engine_mesh.init_morph_targets(
                            geometry.morph_target_count,
                            geometry.morph_vertex_count
                        );
                    }
                }
                
                let mesh_key = self.scene.meshes.insert(engine_mesh);

                // 简化处理：目前只支持将第一个 primitive 挂载到节点
                // 如果有多 primitive，标准做法是创建子节点挂载
                if i == 0 {
                    let engine_node = &mut self.scene.nodes[engine_node_idx];
                    engine_node.mesh = Some(mesh_key);
                } else {
                    println!("Warning: Multi-primitive meshes not fully supported on single node yet.");
                }
            }
        }

        // 2. 绑定 Skin
        if let Some(skin) = node.skin() {
            let skeleton_key = skeleton_keys[skin.index()];
            let engine_node = &mut self.scene.nodes[engine_node_idx];
            // glTF 默认是 Attached 模式
            engine_node.bind_skeleton(skeleton_key, BindMode::Attached);
        }

        Ok(())
    }

    fn load_primitive_geometry(
        &mut self,
        primitive: &gltf::Primitive,
        buffers: &[Vec<u8>]
    ) -> anyhow::Result<GeometryHandle> {
        let mut geometry = Geometry::new();
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        // Positions
        if let Some(iter) = reader.read_positions() {
            let positions: Vec<[f32; 3]> = iter.collect();
            geometry.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));
        }

        // Normals
        if let Some(iter) = reader.read_normals() {
            let normals: Vec<[f32; 3]> = iter.collect();
            geometry.set_attribute("normal", Attribute::new_planar(&normals, VertexFormat::Float32x3));
        }

        // UVs
        if let Some(iter) = reader.read_tex_coords(0) {
            let uvs: Vec<[f32; 2]> = iter.into_f32().collect();
            geometry.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
        }

        // Tangents
        if let Some(iter) = reader.read_tangents() {
            let tangents: Vec<[f32; 4]> = iter.collect();
            geometry.set_attribute("tangent", Attribute::new_planar(&tangents, VertexFormat::Float32x4));
        }

        // Joints (Skinning)
        if let Some(iter) = reader.read_joints(0) {
            // 将 u8/u16 统一转为 u16 以匹配 shader 常见的 Uint16x4
            let joints: Vec<[u16; 4]> = iter.into_u16().collect();
            geometry.set_attribute("joints", Attribute::new_planar(&joints, VertexFormat::Uint16x4));
        }

        // Weights (Skinning)
        if let Some(iter) = reader.read_weights(0) {
            let weights: Vec<[f32; 4]> = iter.into_f32().collect();
            geometry.set_attribute("weights", Attribute::new_planar(&weights, VertexFormat::Float32x4));
        }

        // Indices
        if let Some(iter) = reader.read_indices() {
            let indices: Vec<u32> = iter.into_u32().collect();
            geometry.set_indices_u32(&indices);
        }


        // 定义一个闭包，用于从 buffers 中获取数据切片
        // 这是 gltf::accessor::Iter 所需要的
        let get_buffer_data = |buffer: gltf::Buffer| -> Option<&[u8]> {
            buffers.get(buffer.index()).map(|v| v.as_slice())
        };

        // === 2.加载 Morph Targets 数据 ===
        for target in primitive.morph_targets() {

            // 2.1 Morph Positions (Displacement) - Vec3
            if let Some(accessor) = target.positions() {
                // 使用 gltf::accessor::Iter 直接读取
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
                    let attr = Attribute::new_planar(&data, VertexFormat::Float32x3);
                    geometry.morph_attributes.entry("position".to_string())
                        .or_default()
                        .push(attr);
                }
            }

            // 2.2 Morph Normals (Displacement) - Vec3
            if let Some(accessor) = target.normals() {
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
                    let attr = Attribute::new_planar(&data, VertexFormat::Float32x3);
                    geometry.morph_attributes.entry("normal".to_string())
                        .or_default()
                        .push(attr);
                }
            }

            // 2.3 Morph Tangents (Displacement) - Vec3
            // Morph Target 的切线数据只有 XYZ 位移，没有 W 分量？
            if let Some(accessor) = target.tangents() {
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
                    // 这里我们依然可以用 Float32x3 存储
                    let attr = Attribute::new_planar(&data, VertexFormat::Float32x3);
                    geometry.morph_attributes.entry("tangent".to_string())
                        .or_default()
                        .push(attr);
                }
            }
        }

        // === 3. 构建 Morph Storage Buffers ===
        geometry.build_morph_storage_buffers();

        geometry.compute_bounding_volume();

        Ok(self.assets.add_geometry(geometry))
    }

    fn load_animations(
        &self, 
        gltf: &gltf::Gltf, 
        buffers: &[Vec<u8>]
    ) -> anyhow::Result<Vec<AnimationClip>> {
        let mut animations = Vec::new();

        for anim in gltf.animations() {
            let mut tracks = Vec::new();

            for channel in anim.channels() {
                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
                let target = channel.target();
                let gltf_node = target.node();
                
                // 获取节点名称用于绑定
                let node_name = gltf_node.name().unwrap_or("Node").to_string();
         
                let times: Vec<f32> = reader.read_inputs().unwrap().collect();
                
                let interpolation = match channel.sampler().interpolation() {
                    gltf::animation::Interpolation::Linear => InterpolationMode::Linear,
                    gltf::animation::Interpolation::Step => InterpolationMode::Step,
                    gltf::animation::Interpolation::CubicSpline => InterpolationMode::CubicSpline,
                };

                // 根据属性类型创建不同的 Track
                let track = match target.property() {
                    gltf::animation::Property::Translation => {
                        let outputs = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::Translations(iter) => {
                                iter.map(|t| Vec3::from_array(t)).collect::<Vec<_>>()
                            },
                            _ => continue,
                        };
                        
                        Track {
                            meta: TrackMeta {
                                node_name,
                                target: TargetPath::Translation,
                            },
                            data: TrackData::Vector3(KeyframeTrack::new(times, outputs, interpolation)),
                        }
                    },
                    gltf::animation::Property::Rotation => {
                        let outputs = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::Rotations(iter) => {
                                iter.into_f32().map(|r| Quat::from_array(r)).collect::<Vec<_>>()
                            },
                            _ => continue,
                        };
                        
                        Track {
                            meta: TrackMeta {
                                node_name,
                                target: TargetPath::Rotation,
                            },
                            data: TrackData::Quaternion(KeyframeTrack::new(times, outputs, interpolation)),
                        }
                    },
                    gltf::animation::Property::Scale => {
                        let outputs = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::Scales(iter) => {
                                iter.map(|s| Vec3::from_array(s)).collect::<Vec<_>>()
                            },
                            _ => continue,
                        };
                        
                        Track {
                            meta: TrackMeta {
                                node_name,
                                target: TargetPath::Scale,
                            },
                            data: TrackData::Vector3(KeyframeTrack::new(times, outputs, interpolation)),
                        }
                    },
                    gltf::animation::Property::MorphTargetWeights => {
                        let outputs: Vec<f32> = match reader.read_outputs().unwrap() {
                            gltf::animation::util::ReadOutputs::MorphTargetWeights(iter) => {
                                iter.into_f32().collect()
                            },
                            _ => continue,
                        };
                        
                        // 计算每帧的权重数量
                        let weights_per_frame = if !times.is_empty() {
                            outputs.len() / times.len()
                        } else {
                            0
                        };
                        
                        Track {
                            meta: TrackMeta {
                                node_name,
                                target: TargetPath::Weights,
                            },
                            data: TrackData::MorphWeights(MorphWeightsTrack::new(times, outputs, weights_per_frame)),
                        }
                    },
                };
                
                tracks.push(track);
            }
            
            let clip = AnimationClip::new(
                anim.name().unwrap_or("anim").to_string(),
                tracks,
            );
            animations.push(clip);
        }
        
        Ok(animations)
    }
}