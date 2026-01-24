use std::collections::HashMap;
use std::path::Path;
use std::fs;
use std::sync::atomic::Ordering;
use glam::{Affine3A, Mat4, Quat, Vec3, Vec4};
use crate::resources::{Material, MeshPhysicalMaterial, TextureSampler};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::texture::Texture;
use crate::scene::{Scene, NodeHandle, SkeletonKey};
use crate::scene::skeleton::{Skeleton, BindMode};
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use crate::animation::clip::{AnimationClip, Track, TrackMeta, TrackData};
use crate::animation::tracks::{KeyframeTrack, InterpolationMode};
use crate::animation::binding::TargetPath;
use crate::animation::values::MorphWeightData;
use crate::resources::mesh::MAX_MORPH_TARGETS;
use wgpu::{VertexFormat, TextureFormat};
use anyhow::Context;
use serde_json::Value;

// ============================================================================
// 1. 插件化架构定义 (Plugin Architecture)
// ============================================================================

/// glTF 加载扩展 Trait
/// Plugin 机制，允许拦截加载过程的不同阶段

pub struct LoadContext<'a> {
    pub assets: &'a mut AssetServer,
    pub texture_map: &'a [TextureHandle],
    pub material_map: &'a [MaterialHandle],
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
    fn on_load_node(&mut self, _ctx: &mut LoadContext, _gltf_node: &gltf::Node, _scene: &mut Scene, _node_handle: NodeHandle, _extension_value: &Value) -> anyhow::Result<()> {
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
    // glTF Node Index -> Engine NodeHandle
    node_mapping: Vec<NodeHandle>,

    // 扩展列表
    extensions: HashMap<String, Box<dyn GltfExtensionParser>>,
}

impl<'a> GltfLoader<'a> {
    /// 入口函数
    pub fn load(
        path: &Path,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<(Vec<NodeHandle>, Vec<AnimationClip>)> {
        
        // 1. 读取文件和 Buffer
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open glTF file: {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader_without_validation(reader)
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

        // 注册默认扩展
        loader._register_extension(Box::new(KhrMaterialsPbrSpecularGlossiness));
        loader._register_extension(Box::new(KhrRMaterialsClearcoat));

        let mut supported_ext = loader.extensions.keys().cloned().collect::<Vec<_>>();

        // buiit-in supported extensions
        supported_ext.push("KHR_materials_emissive_strength".to_string());
        supported_ext.push("KHR_materials_ior".to_string());
        supported_ext.push("KHR_materials_specular".to_string());

        let require_not_supported: Vec<_> = gltf.extensions_required().filter(
            |ext| !supported_ext.contains(&ext.to_string())
        ).collect();

        if !require_not_supported.is_empty() {
            println!("glTF file requires unsupported extensions: {:?}", require_not_supported);
        }

        let used_not_supported: Vec<_> = gltf.extensions_used().filter(
            |ext| !supported_ext.contains(&ext.to_string())
        ).collect();

        if !used_not_supported.is_empty() {
            println!("This GLTF uses extensions: {:?}, which are not supported yet, so the display may not be so correct.", used_not_supported);
        }

        // 3. 资源加载流程
        loader.load_textures(&gltf, &buffers)?;
        loader.load_materials(&gltf)?;

        // 4. 节点处理流程 (四步走)
        
        // Step 4.1: 创建所有节点 (Node & Transform)
        for node in gltf.nodes() {
            let handle = loader.create_node_shallow(&node)?;
            loader.node_mapping.push(handle);
        }

        // Step 4.2: 建立层级关系 (Hierarchy)
        let mut root_handles = Vec::new();
        for node in gltf.nodes() {
            let parent_handle = loader.node_mapping[node.index()];
            
            if node.children().len() > 0 {
                for child in node.children() {
                    let child_handle = loader.node_mapping[child.index()];
                    loader.scene.attach(child_handle, parent_handle);
                }
            }
        }
        
        // 找出场景根节点
        if let Some(default_scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            for node in default_scene.nodes() {
                let node_handle = loader.node_mapping[node.index()];
                root_handles.push(node_handle);
                loader.scene.root_nodes.push(node_handle);
            }
        }


        // Step 4.3: 加载骨架 (Skins) - 此时所有 Node 已存在，可以安全引用
        let skeleton_keys = loader.load_skins(&gltf, &buffers)?;

        // Step 4.4: 绑定 Mesh 和 Skin
        for node in gltf.nodes() {
            loader.bind_node_mesh_and_skin(&node, &buffers, &skeleton_keys)?;
        }

        // 5. 加载动画数据 (可选)
        let animations = loader.load_animations(&gltf, &buffers)?;

        Ok((root_handles, animations))
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

            let sampler = texture.sampler();

            let engine_sampler = TextureSampler{
                mag_filter: sampler.mag_filter().map(|f| match f {
                    gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
                }).unwrap_or(wgpu::FilterMode::Linear),
                min_filter: sampler.min_filter().map(|f| match f {
                    gltf::texture::MinFilter::Nearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MinFilter::Linear => wgpu::FilterMode::Linear,
                    gltf::texture::MinFilter::NearestMipmapNearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MinFilter::LinearMipmapNearest => wgpu::FilterMode::Linear,
                    gltf::texture::MinFilter::NearestMipmapLinear => wgpu::FilterMode::Nearest,
                    gltf::texture::MinFilter::LinearMipmapLinear => wgpu::FilterMode::Linear,
                }).unwrap_or(wgpu::FilterMode::Linear),
                address_mode_u: match sampler.wrap_s() {
                    gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                address_mode_v: match sampler.wrap_t() {
                    gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                address_mode_w: wgpu::AddressMode::ClampToEdge, // glTF 不支持 3D 纹理，这里默认
                mipmap_filter: match sampler.min_filter() {
                    Some(gltf::texture::MinFilter::NearestMipmapNearest) | Some(gltf::texture::MinFilter::LinearMipmapNearest) => wgpu::MipmapFilterMode::Nearest,
                    Some(gltf::texture::MinFilter::NearestMipmapLinear) | Some(gltf::texture::MinFilter::LinearMipmapLinear) => wgpu::MipmapFilterMode::Linear,
                    _ => wgpu::MipmapFilterMode::Linear,
                },
                ..Default::default()
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

            engine_tex.sampler = engine_sampler;
            engine_tex.generate_mipmaps = true;

            if let Some(extensions_map) = texture.extensions() {
                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    texture_map: &self.texture_map, 
                    material_map: &self.material_map,
                };

                for (name, value) in extensions_map {
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
            let mut mat = MeshPhysicalMaterial::new(base_color_factor);

            mat.set_metalness(pbr.metallic_factor());
            mat.set_roughness(pbr.roughness_factor());
            mat.set_emissive(Vec3::from_array(material.emissive_factor()));
            
            {
                let bindings = mat.bindings_mut();
                if let Some(info) = pbr.base_color_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    bindings.map = Some(tex_handle.into());
                    
                    if let Some(texture) = self.assets.get_texture(tex_handle) {
                        texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
                    }
                }

                if let Some(info) = pbr.metallic_roughness_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    bindings.roughness_map = Some(tex_handle.into());
                    bindings.metalness_map = Some(tex_handle.into());
                }

                if let Some(info) = material.normal_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    bindings.normal_map = Some(tex_handle.into());
                }

                if let Some(info) = material.occlusion_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    bindings.ao_map = Some(tex_handle.into());
                }

                if let Some(info) = material.emissive_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    
                    if let Some(texture) = self.assets.get_texture(tex_handle) {
                        texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
                    }
                    bindings.emissive_map = Some(tex_handle.into());
                }
            }

            {
                let mut settings = mat.settings_mut();

                settings.side = if material.double_sided() { crate::resources::material::Side::Double } else { crate::resources::material::Side::Front };
                settings.transparent = match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => false,
                    gltf::material::AlphaMode::Mask => false,
                    gltf::material::AlphaMode::Blend => true,
                };
            }


            //=========================Material Extensions=========================//


            // KHR_materials_emissive_strength
            if let Some(info) = material.emissive_strength() {
                mat.set_emissive_intensity(info);
            }

            // KHR_materials_ior
            if let Some(info) =  material.ior() {
                mat.set_ior(info);
            }

            // KHR_materials_specular
            if let Some(specular) = material.specular() {
                mat.set_specular_color(Vec3::from_array(specular.specular_color_factor()));
                mat.set_specular_intensity(specular.specular_factor());

                if let Some(info) = specular.specular_color_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];

                    if let Some(texture) = self.assets.get_texture(tex_handle) {
                        texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
                    }

                    mat.set_specular_map(Some(tex_handle.into()));
                }

                if let Some(info) = specular.specular_texture() {
                    let tex_handle = self.texture_map[info.texture().index()];
                    mat.set_specular_intensity_map(Some(tex_handle.into()));
                }
            }

            // 转换为通用 Material 枚举
            let mut engine_mat = Material::from(mat);

            // 处理 KHR_materials_pbrSpecularGlossiness
            if material.pbr_specular_glossiness().is_some() {
                if let Some(handler) = self.extensions.get_mut("KHR_materials_pbrSpecularGlossiness") {
                    let mut ctx = LoadContext {
                        assets: &mut self.assets, 
                        texture_map: &self.texture_map, 
                        material_map: &self.material_map,
                    };
                    handler.on_load_material(&mut ctx, &material, &mut engine_mat, &Value::Null)?;
                }
            }

            // extensions 处理
            if let Some(extensions_map) = material.extensions() {
                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    texture_map: &self.texture_map, 
                    material_map: &self.material_map,
                };

                for (name, value) in extensions_map {
                    // println!("Processing material extension: {}, {}", name, value);
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

    // 仅创建节点和 Transform，设置名称组件
    fn create_node_shallow(&mut self, node: &gltf::Node) -> anyhow::Result<NodeHandle> {
        let node_name = node.name().unwrap_or("Node");
        let handle = self.scene.create_node_with_name(node_name);

        // 设置变换
        if let Some(engine_node) = self.scene.get_node_mut(handle) {
            let (t, r, s) = node.transform().decomposed();
            engine_node.transform.position = Vec3::from_array(t);
            engine_node.transform.rotation = Quat::from_array(r);
            engine_node.transform.scale = Vec3::from_array(s);
        }

        // 处理节点扩展
        if let Some(extensions_map) = node.extensions() {
            let mut ctx = LoadContext {
                assets: &mut self.assets,
                texture_map: &self.texture_map,
                material_map: &self.material_map,
            };
            for (name, value) in extensions_map {
                if let Some(handler) = self.extensions.get_mut(name) {
                    handler.on_load_node(&mut ctx, &node, self.scene, handle, value)?;
                }
            }
        }
        
        Ok(handle)
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
                vec![Affine3A::IDENTITY; skin.joints().count()]
            };

            // 2. 映射 glTF joint indices 到 Engine NodeHandles
            let bones: Vec<NodeHandle> = skin.joints()
                .map(|node| self.node_mapping[node.index()]) 
                .collect();

            // =========================================================
            // 计算真正的 root_bone_index
            // =========================================================
            let joints: Vec<_> = skin.joints().collect();
            let joint_indices: std::collections::HashSet<usize> = joints.iter().map(|n| n.index()).collect();
            
            // 步骤 1: 找出所有"有父节点"的骨骼 (即被其他骨骼作为子节点引用的)
            let mut child_joint_indices = std::collections::HashSet::new();
            for node in &joints {
                for child in node.children() {
                    // 如果子节点也在 joints 列表中，将其标记为"有父节点"
                    if joint_indices.contains(&child.index()) {
                        child_joint_indices.insert(child.index());
                    }
                }
            }

            let root_bone_index = 'block: {
                // A. 优先尝试 glTF 显式定义的 skeleton root
                if let Some(skeleton_root) = skin.skeleton() {
                    if let Some(index) = joints.iter().position(|n| n.index() == skeleton_root.index()) {
                        break 'block index;
                    }
                }

                // B. 自动查找：在 joints 列表中，没在 child_joint_indices 里的就是根
                // (因为它在 joints 范围内没有父节点)
                for (i, node) in joints.iter().enumerate() {
                    if !child_joint_indices.contains(&node.index()) {
                        break 'block i; 
                    }
                }
                
                // C. 实在找不到（比如环形结构异常），回退到 0
                0 
            };
            // 3. 创建资源
            let skeleton = Skeleton::new(name, bones, ibms, root_bone_index);
    
            let key = self.scene.skeleton_pool.insert(skeleton); 
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
        let engine_node_handle = self.node_mapping[node.index()];

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
                
                // 直接将 Mesh 组件挂载到节点
                // 简化处理：目前只支持将第一个 primitive 挂载到节点
                if i == 0 {
                    self.scene.set_mesh(engine_node_handle, engine_mesh);
                } else {
                    println!("Warning: Multi-primitive meshes not fully supported on single node yet.");
                }
            }
        }

        // 2. 绑定 Skin
        if let Some(skin) = node.skin() {
            let skeleton_key = skeleton_keys[skin.index()];
            // glTF 默认是 Attached 模式
            self.scene.bind_skeleton(engine_node_handle, skeleton_key, BindMode::Attached);
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
        let get_buffer_data = |buffer: gltf::Buffer| -> Option<&[u8]> {
            buffers.get(buffer.index()).map(|v| v.as_slice())
        };

        // === 2.加载 Morph Targets 数据 ===
        for target in primitive.morph_targets() {
            // 2.1 Morph Positions (Displacement) - Vec3
            if let Some(accessor) = target.positions() {
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
            if let Some(accessor) = target.tangents() {
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
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
                        
                        let weights_per_frame = if !times.is_empty() {
                            outputs.len() / times.len()
                        } else {
                            0
                        };
                        
                        let mut pod_outputs = Vec::with_capacity(times.len());
                        for i in 0..times.len() {
                            let mut pod = MorphWeightData::default();
                            let start = i * weights_per_frame;
                            let count = weights_per_frame.min(MAX_MORPH_TARGETS);
                            pod.weights[..count].copy_from_slice(&outputs[start..start + count]);
                            pod_outputs.push(pod);
                        }
                        
                        Track {
                            meta: TrackMeta {
                                node_name,
                                target: TargetPath::Weights,
                            },
                            data: TrackData::MorphWeights(KeyframeTrack::new(times, pod_outputs, interpolation)),
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

struct KhrMaterialsPbrSpecularGlossiness;

impl GltfExtensionParser for KhrMaterialsPbrSpecularGlossiness {
    fn name(&self) -> &str {
        "KHR_materials_pbrSpecularGlossiness"
    }

    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, engine_mat: &mut Material, _extension_value: &Value) -> anyhow::Result<()> {
        let sg = _gltf_mat.pbr_specular_glossiness()
            .ok_or_else(|| anyhow::anyhow!("Material missing pbr_specular_glossiness data"))?;

        let physical_mat: &mut MeshPhysicalMaterial = engine_mat.as_any_mut().downcast_mut().ok_or_else(|| anyhow::anyhow!("Material is not MeshStandardMaterial"))?;

        // 1. 设置基础材质参数 (转换为非金属材质)
        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.metalness = 0.0;
            uniforms.roughness = 1.0;
            uniforms.specular_color = Vec3::from_array(sg.specular_factor());
            uniforms.specular_intensity = 1.0;
            uniforms.color = Vec4::from_array(sg.diffuse_factor());
        }

        // 2. 处理 diffuse 纹理 -> base color map
        if let Some(diffuse_tex) = sg.diffuse_texture() {
            let tex_handle = _ctx.texture_map[diffuse_tex.texture().index()];
            let bindings = physical_mat.bindings_mut();
            
            if let Some(texture) = _ctx.assets.get_texture(tex_handle) {
                texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
            }
            bindings.map = Some(tex_handle.into());
        }

        // 3. 处理 specular-glossiness 纹理
        if let Some(sg_tex_info) = sg.specular_glossiness_texture() {
            let tex_handle = _ctx.texture_map[sg_tex_info.texture().index()];

            if let Some(texture) = _ctx.assets.get_texture(tex_handle) {
                texture.image.set_format(TextureFormat::Rgba8UnormSrgb);
            }
            
            let glossiness_factor = sg.glossiness_factor();
            
            let (width, height, source_data) = if let Some(source_texture) = _ctx.assets.get_texture(tex_handle) {
                let source_image = &source_texture.image;
                let w = source_image.width.load(Ordering::Relaxed);
                let h = source_image.height.load(Ordering::Relaxed);
                let data_clone = source_image.data.read().unwrap().clone();
                (w, h, data_clone)
            } else {
                (0, 0, None)
            };
            
            if let Some(data) = source_data {
                let pixel_count = (width * height) as usize;
                
                let mut specular_data = Vec::with_capacity(pixel_count * 4);
                let mut roughness_data = Vec::with_capacity(pixel_count * 4);
                
                for i in 0..pixel_count {
                    let offset = i * 4;
                    let r = data[offset];
                    let g = data[offset + 1];
                    let b = data[offset + 2];
                    let glossiness = data[offset + 3];
                    
                    specular_data.push(r);
                    specular_data.push(g);
                    specular_data.push(b);
                    specular_data.push(255);
                    
                    let glossiness_normalized = (glossiness as f32 / 255.0) * glossiness_factor;
                    let roughness_normalized = 1.0 - glossiness_normalized;
                    let roughness_byte = (roughness_normalized * 255.0) as u8;
                    
                    roughness_data.push(0);
                    roughness_data.push(roughness_byte);
                    roughness_data.push(0);
                    roughness_data.push(255);
                }
                
                let specular_texture = Texture::new_2d(
                    Some("sg_specular"),
                    width,
                    height,
                    Some(specular_data),
                    TextureFormat::Rgba8UnormSrgb
                );
                
                let roughness_texture = Texture::new_2d(
                    Some("sg_roughness"),
                    width,
                    height,
                    Some(roughness_data),
                    TextureFormat::Rgba8Unorm
                );
                
                let specular_handle = _ctx.assets.add_texture(specular_texture);
                let roughness_handle = _ctx.assets.add_texture(roughness_texture);
                
                let bindings = physical_mat.bindings_mut();
                bindings.specular_map = Some(specular_handle.into());
                bindings.roughness_map = Some(roughness_handle.into());
                bindings.metalness_map = Some(roughness_handle.into());
            }
        } else {
            let glossiness_factor = sg.glossiness_factor();
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.roughness = 1.0 - glossiness_factor;
        }

        Ok(())
    }
}


struct KhrRMaterialsClearcoat;

impl GltfExtensionParser for KhrRMaterialsClearcoat {
    fn name(&self) -> &str {
        "KHR_materials_clearcoat"
    }

    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, engine_mat: &mut Material, extension_value: &Value) -> anyhow::Result<()> {
        let clearcoat_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid clearcoat extension data"))?;

        let clearcoat_factor = clearcoat_info.get("clearcoatFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let clearcoat_roughness = clearcoat_info.get("clearcoatRoughnessFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let physical_mat: &mut MeshPhysicalMaterial = engine_mat.as_any_mut().downcast_mut().ok_or_else(|| anyhow::anyhow!("Material is not MeshPhysicalMaterial"))?;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.clearcoat = clearcoat_factor;
            uniforms.clearcoat_roughness = clearcoat_roughness;
        }

        // 处理 clearcoat texture
        if let Some(clearcoat_tex_info) = clearcoat_info.get("clearcoatTexture") {
            if let Some(index) = clearcoat_tex_info.get("index").and_then(|v| v.as_u64()) {
                let tex_handle: TextureHandle = _ctx.texture_map[index as usize];
                physical_mat.set_clearcoat_map(tex_handle);
            }
        }

        // 处理 clearcoat roughness texture
        if let Some(clearcoat_roughness_tex_info) = clearcoat_info.get("clearcoatRoughnessTexture") {
            if let Some(index) = clearcoat_roughness_tex_info.get("index").and_then(|v| v.as_u64()) {
                let tex_handle = _ctx.texture_map[index as usize];
                physical_mat.set_clearcoat_roughness_map(Some(tex_handle.into()));
            }
        }

        // 处理 clearcoat normal texture
        if let Some(clearcoat_normal_tex_info) = clearcoat_info.get("clearcoatNormalTexture") {
            if let Some(index) = clearcoat_normal_tex_info.get("index").and_then(|v| v.as_u64()) {
                let tex_handle = _ctx.texture_map[index as usize];
                physical_mat.set_clearcoat_normal_map(tex_handle);
                // Todo: normal map scale
            }
        }

        Ok(())
    }
    
}