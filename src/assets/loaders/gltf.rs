use std::collections::HashMap;
use std::path::Path;
use std::fs;
use glam::{Affine3A, Mat4, Quat, Vec2, Vec3, Vec4};
use crate::{AnimationAction, AnimationMixer, Binder};
use crate::resources::{Material, MeshPhysicalMaterial, TextureSampler, TextureSlot, TextureTransform};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::texture::Texture;
use crate::scene::{NodeHandle, Scene, SkeletonKey};
use crate::scene::skeleton::{Skeleton, BindMode};
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use crate::animation::clip::{AnimationClip, Track, TrackMeta, TrackData};
use crate::animation::tracks::{KeyframeTrack, InterpolationMode};
use crate::animation::binding::TargetPath;
use crate::animation::values::MorphWeightData;
use crate::resources::mesh::MAX_MORPH_TARGETS;
use wgpu::{VertexFormat, TextureFormat};
use anyhow::Context;
use serde_json::{Value};

// ============================================================================
// 1. 中间数据结构 (Intermediate Data Structures)
// ============================================================================

/// 临时存储 glTF 纹理的源数据，用于延迟创建引擎 Texture
struct IntermediateTexture {
    name: Option<String>,
    image_data: Vec<u8>,   // RGBA8 格式的像素数据
    width: u32,
    height: u32,
    sampler: TextureSampler,
}

/// 纹理缓存的 Key，用于去重
/// 如果同一个 index 的纹理以相同的 sRGB 格式请求，则复用 Handle
#[derive(Hash, PartialEq, Eq, Clone)]
struct TextureCacheKey {
    gltf_texture_index: usize,
    is_srgb: bool,
}

// ============================================================================
// 2. 插件化架构定义 (Plugin Architecture)
// ============================================================================

/// glTF 加载扩展 Trait
/// Plugin 机制，允许拦截加载过程的不同阶段

pub struct LoadContext<'a, 'b> {
    pub assets: &'a mut AssetServer,
    pub material_map: &'a [MaterialHandle],
    /// 用于在扩展中获取纹理（内部会根据 is_srgb 参数缓存和去重）
    intermediate_textures: &'a [IntermediateTexture],
    created_textures: &'a mut HashMap<TextureCacheKey, TextureHandle>,
    _phantom: std::marker::PhantomData<&'b ()>,
}

pub trait GltfExtensionParser {
    fn name(&self) -> &str;

    /// 当材质被加载时调用
    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, _engine_mat: &mut Material, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    /// 当节点被创建时调用 (可用于处理 KHR_lights_punctual 等挂载到节点的扩展)
    fn on_load_node(&mut self, _ctx: &mut LoadContext, _gltf_node: &gltf::Node, _scene: &mut Scene, _node_handle: NodeHandle, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    fn setup_texture_map_from_extension(&mut self, ctx: &mut LoadContext, info: &Value , texture_slot: &mut TextureSlot){

        if let Some(tex_info) = info.get("clearcoatTexture") {
            if let Some(index) = tex_info.get("index").and_then(|v| v.as_u64()) {
                let Some(tex_handle) = ctx.get_or_create_texture(index as usize, false).ok() else {
                    // log::warn!("Failed to create texture for index {}", index);
                    return;
                };
                texture_slot.texture = Some(tex_handle);
                texture_slot.channel = tex_info.get("texCoord").and_then(|v| v.as_u64()).unwrap_or(0) as u8;

                if let Some(transform) = tex_info.get("extensions")
                    .and_then(|exts| exts.get("KHR_texture_transform"))
                {
                    if let Some(offset_array) = transform.get("offset").and_then(|v| v.as_array()) {
                        let offset_x = offset_array.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        let offset_y = offset_array.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        texture_slot.transform.offset = Vec2::new(offset_x, offset_y);
                    }
                    if let Some(scale_array) = transform.get("scale").and_then(|v| v.as_array()) {
                        let scale_x = scale_array.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                        let scale_y = scale_array.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                        texture_slot.transform.scale = Vec2::new(scale_x, scale_y);
                    }
                    if let Some(rotation) = transform.get("rotation").and_then(|v| v.as_f64()) {
                        texture_slot.transform.rotation = rotation as f32;
                    }

                    if let Some(tex_coord) = transform.get("texCoord").and_then(|v| v.as_u64()) {
                        texture_slot.channel = tex_coord as u8;
                    }
                }
            }
        }

    }
}

// LoadContext 的辅助方法
impl<'a, 'b> LoadContext<'a, 'b> {
    /// 获取或创建纹理
    /// - `gltf_texture_index`: glTF 纹理索引
    /// - `is_srgb`: 是否使用 sRGB 颜色空间（baseColor/emissive 等使用 true，normal/roughness 等使用 false）
    pub fn get_or_create_texture(
        &mut self,
        gltf_texture_index: usize,
        is_srgb: bool,
    ) -> anyhow::Result<TextureHandle> {
        let key = TextureCacheKey {
            gltf_texture_index,
            is_srgb,
        };

        // 检查缓存
        if let Some(&handle) = self.created_textures.get(&key) {
            return Ok(handle);
        }

        // 从中间数据创建纹理
        let raw = self.intermediate_textures.get(gltf_texture_index)
            .ok_or_else(|| anyhow::anyhow!("Texture index {} out of bounds", gltf_texture_index))?;

        let format = if is_srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };

        let mut engine_tex = Texture::new_2d(
            raw.name.as_deref(),
            raw.width,
            raw.height,
            Some(raw.image_data.clone()),
            format,
        );

        engine_tex.sampler = raw.sampler.clone();
        engine_tex.generate_mipmaps = true;

        let handle = self.assets.add_texture(engine_tex);
        self.created_textures.insert(key, handle);

        Ok(handle)
    }
}

// ============================================================================
// 2. GltfLoader 实现
// ============================================================================

pub struct GltfLoader<'a> {
    assets: &'a mut AssetServer,
    scene: &'a mut Scene,
    base_path: std::path::PathBuf,
    
    // 纹理中间数据（解析阶段生成，下标对应 glTF texture index）
    intermediate_textures: Vec<IntermediateTexture>,
    // 已创建的引擎纹理缓存（用于去重）
    created_textures: HashMap<TextureCacheKey, TextureHandle>,
    
    // 材质映射表
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
    ) -> anyhow::Result<NodeHandle> {
        
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
            intermediate_textures: Vec::new(),
            created_textures: HashMap::new(),
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
        let root_handle = loader.scene.create_node_with_name("gltf_root");
        loader.scene.root_nodes.push(root_handle);
    

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
                loader.scene.attach(node_handle, root_handle);
            }
        }


        // Step 4.3: 加载骨架 (Skins) - 此时所有 Node 已存在，可以安全引用
        let skeleton_keys = loader.load_skins(&gltf, &buffers)?;

        // Step 4.4: 绑定 Mesh 和 Skin
        for node in gltf.nodes() {
            loader.bind_node_mesh_and_skin(&node, &buffers, &skeleton_keys)?;
        }

        // 5. 加载动画数据
        let animations = loader.load_animations(&gltf, &buffers)?;

        let mut mixer = AnimationMixer::new();
        for clip in animations {
            // 1. 为每个 Clip 创建 Action
            // let clip = clip.clone();
            let bindings = Binder::bind(loader.scene, root_handle, &clip);
            
            let mut action = AnimationAction::new(clip.into());
            action.bindings = bindings;
            
            // 默认不播放，或者权重设为 0
            action.enabled = false; 
            action.weight = 0.0;
            
            // 2. 全部加入同一个 Mixer
            mixer.add_action(action);
        }
        loader.scene.animation_mixers.insert(root_handle, mixer);

        Ok(root_handle)
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

    /// 解析阶段：只加载 Image 数据和 Sampler 配置，存放在中间结构中
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

            // 转换 glTF Sampler 为引擎 TextureSampler
            let engine_sampler = TextureSampler {
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
            let tex_name = texture.name().map(|s| s.to_string());

            // 只存储中间数据，不创建引擎纹理
            self.intermediate_textures.push(IntermediateTexture {
                name: tex_name,
                image_data: img_data.into_vec(),
                width,
                height,
                sampler: engine_sampler,
            });
        }
        Ok(())
    }

    /// 获取或创建纹理的辅助方法
    /// - `gltf_texture_index`: glTF 纹理索引
    /// - `is_srgb`: 是否使用 sRGB 颜色空间
    fn get_or_create_texture(
        &mut self,
        gltf_texture_index: usize,
        is_srgb: bool,
    ) -> anyhow::Result<TextureHandle> {
        let key = TextureCacheKey {
            gltf_texture_index,
            is_srgb,
        };

        // 检查缓存
        if let Some(&handle) = self.created_textures.get(&key) {
            return Ok(handle);
        }

        // 从中间数据创建纹理
        let raw = self.intermediate_textures.get(gltf_texture_index)
            .ok_or_else(|| anyhow::anyhow!("Texture index {} out of bounds", gltf_texture_index))?;

        let format = if is_srgb {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        };

        let mut engine_tex = Texture::new_2d(
            raw.name.as_deref(),
            raw.width,
            raw.height,
            Some(raw.image_data.clone()),
            format,
        );

        engine_tex.sampler = raw.sampler.clone();
        engine_tex.generate_mipmaps = true;

        let handle = self.assets.add_texture(engine_tex);
        self.created_textures.insert(key, handle);

        Ok(handle)
    }

    fn setup_texture_map(&mut self, texture_slot: &mut TextureSlot, info: &gltf::texture::Info) -> anyhow::Result<()> {
        let tex_handle = self.get_or_create_texture(info.texture().index(), true)?;
        texture_slot.texture = Some(tex_handle);
        texture_slot.channel = info.tex_coord() as u8;
        if let Some(transform) = info.texture_transform() {
            texture_slot.transform.offset = Vec2::from_array(transform.offset());
            texture_slot.transform.scale = Vec2::from_array(transform.scale());
            texture_slot.transform.rotation = transform.rotation();

            if let Some(tex_coord) = transform.tex_coord() {
                texture_slot.channel = tex_coord as u8;
            }
        }
        Ok(())
    }

    /// 实例化阶段：加载材质时，按需创建纹理（根据使用上下文决定 colorspace）
    fn load_materials(&mut self, gltf: &gltf::Gltf) -> anyhow::Result<()> {
        for material in gltf.materials() {
            let pbr = material.pbr_metallic_roughness();

            let base_color_factor = Vec4::from_array(pbr.base_color_factor());
            let mut mat = MeshPhysicalMaterial::new(base_color_factor);

            mat.set_metalness(pbr.metallic_factor());
            mat.set_roughness(pbr.roughness_factor());
            mat.set_emissive(Vec3::from_array(material.emissive_factor()));
            
            // Base Color Texture (sRGB)
            if let Some(info) = pbr.base_color_texture() {
                // let tex_handle = self.get_or_create_texture(info.texture().index(), true)?;

                self.setup_texture_map(&mut mat.map, &info)?;
            }

            // Metallic-Roughness Texture (Linear)
            if let Some(info) = pbr.metallic_roughness_texture() {
                self.setup_texture_map(&mut mat.roughness_map, &info)?;
                self.setup_texture_map(&mut mat.metalness_map, &info)?;
            }

            // Normal Texture (Linear)
            if let Some(info) = material.normal_texture() {
                // self.setup_texture_map(&mut mat.normal_map, &info)?;
                let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                mat.normal_map.texture = Some(tex_handle);
                mat.normal_map.channel = info.tex_coord() as u8;

                // normal map don't have transform ?
            }

            // Occlusion Texture (Linear)
            if let Some(info) = material.occlusion_texture() {
                let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                mat.ao_map.texture = Some(tex_handle);
                mat.ao_map.channel = info.tex_coord() as u8;
                mat.set_ao_map_intensity(info.strength());

                // ao map don't have transform ?
            }

            // Emissive Texture (sRGB)
            if let Some(info) = material.emissive_texture() {
                self.setup_texture_map(&mut mat.emissive_map, &info)?;
            }

            {
                let mut settings = mat.settings_mut();

                settings.side = if material.double_sided() { crate::resources::material::Side::Double } else { crate::resources::material::Side::Front };
                settings.transparent = match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => false,
                    gltf::material::AlphaMode::Mask => false,
                    gltf::material::AlphaMode::Blend => {
                        true
                    },
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

                // Specular Color Texture (sRGB)
                if let Some(info) = specular.specular_color_texture() {
                    self.setup_texture_map(&mut mat.specular_map, &info)?;
                }

                // Specular Intensity Texture (Linear)
                if let Some(info) = specular.specular_texture() {
                    self.setup_texture_map(&mut mat.specular_intensity_map, &info)?;
                }
            }

            let mut engine_mat = Material::from(mat);

            // 处理 KHR_materials_pbrSpecularGlossiness
            if material.pbr_specular_glossiness().is_some() {
                if let Some(handler) = self.extensions.get_mut("KHR_materials_pbrSpecularGlossiness") {
                    let mut ctx = LoadContext {
                        assets: &mut self.assets, 
                        material_map: &self.material_map,
                        intermediate_textures: &self.intermediate_textures,
                        created_textures: &mut self.created_textures,
                        _phantom: std::marker::PhantomData,
                    };
                    handler.on_load_material(&mut ctx, &material, &mut engine_mat, &Value::Null)?;
                }
            }

            // extensions 处理
            if let Some(extensions_map) = material.extensions() {
                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    material_map: &self.material_map,
                    intermediate_textures: &self.intermediate_textures,
                    created_textures: &mut self.created_textures,
                    _phantom: std::marker::PhantomData,
                };

                for (name, value) in extensions_map {
                    // println!("Processing material extension: {}, {}", name, value);
                    if let Some(handler) = self.extensions.get_mut(name) {
                        handler.on_load_material(&mut ctx, &material, &mut engine_mat, value)?;
                    }
                }
            }

            // 由于上述代码直接修改了 pub(crate) 字段，需要手动通知版本更新
            let physical_mat = engine_mat.as_any_mut().downcast_mut::<MeshPhysicalMaterial>().unwrap();
            physical_mat.flush_texture_transforms();
            physical_mat.notify_pipeline_dirty();

            // 转换为通用 Material 枚举
            // let mut engine_mat = Material::from(mat);

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
                material_map: &self.material_map,
                intermediate_textures: &self.intermediate_textures,
                created_textures: &mut self.created_textures,
                _phantom: std::marker::PhantomData,
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

    fn on_load_material(&mut self, ctx: &mut LoadContext, gltf_mat: &gltf::Material, engine_mat: &mut Material, _extension_value: &Value) -> anyhow::Result<()> {
        let sg = gltf_mat.pbr_specular_glossiness()
            .ok_or_else(|| anyhow::anyhow!("Material missing pbr_specular_glossiness data"))?;

        let physical_mat: &mut MeshPhysicalMaterial = engine_mat.as_any_mut().downcast_mut().ok_or_else(|| anyhow::anyhow!("Material is not MeshStandardMaterial"))?;

        // 1. 设置基础材质参数 (转换为非金属材质)
        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.metalness = 0.0;
            uniforms.roughness = 1.0;
            uniforms.ior = 1000.0;
            uniforms.specular_color = Vec3::from_array(sg.specular_factor());
            uniforms.specular_intensity = 1.0;
            uniforms.color = Vec4::from_array(sg.diffuse_factor());
        }

        // 2. 处理 diffuse 纹理 -> base color map (sRGB)
        if let Some(diffuse_tex) = sg.diffuse_texture() {
            let tex_handle = ctx.get_or_create_texture(diffuse_tex.texture().index(), true)?;
            physical_mat.map.texture = Some(tex_handle);

            
        }

        // 3. 处理 specular-glossiness 纹理
        if let Some(sg_tex_info) = sg.specular_glossiness_texture() {
            let tex_index = sg_tex_info.texture().index();
            let glossiness_factor = sg.glossiness_factor();
            
            // 从中间数据获取原始图像数据
            let raw = ctx.intermediate_textures.get(tex_index)
                .ok_or_else(|| anyhow::anyhow!("Texture index {} out of bounds", tex_index))?;
            
            let width = raw.width;
            let height = raw.height;
            let data = &raw.image_data;
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
            
            let specular_handle = ctx.assets.add_texture(specular_texture);
            let roughness_handle = ctx.assets.add_texture(roughness_texture);

            let mut uv_channel = sg_tex_info.tex_coord();

            let transform = if let Some(tex_transform) = sg_tex_info.texture_transform() {
                uv_channel = tex_transform.tex_coord().unwrap_or(uv_channel);
                TextureTransform {
                    offset: Vec2::from_array(tex_transform.offset()),
                    scale: Vec2::from_array(tex_transform.scale()),
                    rotation: tex_transform.rotation(),
                }
            } else {
                TextureTransform::default()
            };
            
            physical_mat.specular_map.texture = Some(specular_handle);
            physical_mat.specular_map.channel = uv_channel as u8;
            physical_mat.specular_map.transform = transform.clone();

            physical_mat.roughness_map.texture = Some(roughness_handle);
            physical_mat.roughness_map.channel = uv_channel as u8;
            physical_mat.roughness_map.transform = transform.clone();

            physical_mat.metalness_map.texture = Some(roughness_handle);
            physical_mat.metalness_map.channel = uv_channel as u8;
            physical_mat.metalness_map.transform = transform.clone();
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

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, engine_mat: &mut Material, extension_value: &Value) -> anyhow::Result<()> {
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

        // 处理 clearcoat texture (Linear)
        if let Some(clearcoat_tex_info) = clearcoat_info.get("clearcoatTexture") {
            self.setup_texture_map_from_extension(ctx, &clearcoat_tex_info, &mut physical_mat.clearcoat_map);
        }

        // 处理 clearcoat roughness texture (Linear)
        if let Some(clearcoat_roughness_tex_info) = clearcoat_info.get("clearcoatRoughnessTexture") {
            self.setup_texture_map_from_extension(ctx, &clearcoat_roughness_tex_info, &mut physical_mat.clearcoat_roughness_map);
        }

        // 处理 clearcoat normal texture (Linear)
        if let Some(clearcoat_normal_tex_info) = clearcoat_info.get("clearcoatNormalTexture") {
            self.setup_texture_map_from_extension(ctx, &clearcoat_normal_tex_info, &mut physical_mat.clearcoat_normal_map);
        }

        Ok(())
    }
}