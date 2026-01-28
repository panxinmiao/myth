use std::collections::HashMap;
use std::sync::Arc;
use glam::{Affine3A, Mat4, Quat, Vec2, Vec3, Vec4};
use tokio::runtime::Runtime;
use futures::future::try_join_all;
use crate::resources::material::AlphaMode;
use crate::{AnimationAction, AnimationMixer, Binder};
use crate::resources::{Material, MeshPhysicalMaterial, TextureSampler, TextureSlot, TextureTransform};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::texture::Texture;
use crate::resources::buffer::BufferRef;
use crate::scene::{NodeHandle, Scene, SkeletonKey};
use crate::scene::skeleton::{Skeleton, BindMode};
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use crate::assets::io::AssetReaderVariant;
use crate::animation::clip::{AnimationClip, Track, TrackMeta, TrackData};
use crate::animation::tracks::{KeyframeTrack, InterpolationMode};
use crate::animation::binding::TargetPath;
use crate::animation::values::MorphWeightData;
use crate::resources::mesh::MAX_MORPH_TARGETS;
use wgpu::{BufferUsages, PrimitiveTopology, TextureFormat, VertexFormat, VertexStepMode};
use anyhow::Context;
use serde_json::Value;

use std::sync::OnceLock;

fn get_global_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("Failed to create global asset loader runtime")
    })
}

struct IntermediateTexture {
    name: Option<String>,
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    sampler: TextureSampler,
    generate_mipmaps: bool,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct TextureCacheKey {
    gltf_texture_index: usize,
    is_srgb: bool,
}

struct InterleaveChannel {
    name: String,
    data: Vec<u8>,
    format: VertexFormat,
    item_size: usize,
}

impl InterleaveChannel {
    fn from_iter<T, I>(name: &str, iter: I, format: VertexFormat) -> Self
    where
        T: bytemuck::Pod,
        I: Iterator<Item = T>,
    {
        let data: Vec<u8> = iter
            .flat_map(|v| bytemuck::bytes_of(&v).to_vec())
            .collect();
        
        let item_size = std::mem::size_of::<T>();
        
        Self {
            name: name.to_string(),
            data,
            format,
            item_size,
        }
    }
}

pub struct LoadContext<'a, 'b> {
    pub assets: &'a mut AssetServer,
    pub material_map: &'a [MaterialHandle],
    intermediate_textures: &'a [IntermediateTexture],
    created_textures: &'a mut HashMap<TextureCacheKey, TextureHandle>,
    _phantom: std::marker::PhantomData<&'b ()>,
}

pub trait GltfExtensionParser {
    fn name(&self) -> &str;

    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, _engine_mat: &mut Material, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_load_node(&mut self, _ctx: &mut LoadContext, _gltf_node: &gltf::Node, _scene: &mut Scene, _node_handle: NodeHandle, _extension_value: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    fn setup_texture_map_from_extension(&mut self, ctx: &mut LoadContext, tex_info: &Value, texture_slot: &mut TextureSlot, is_srgb: bool) {
        if let Some(index) = tex_info.get("index").and_then(|v| v.as_u64()) {
            let Some(tex_handle) = ctx.get_or_create_texture(index as usize, is_srgb).ok() else {
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

impl<'a, 'b> LoadContext<'a, 'b> {
    pub fn get_or_create_texture(
        &mut self,
        gltf_texture_index: usize,
        is_srgb: bool,
    ) -> anyhow::Result<TextureHandle> {
        let key = TextureCacheKey {
            gltf_texture_index,
            is_srgb,
        };

        if let Some(&handle) = self.created_textures.get(&key) {
            return Ok(handle);
        }

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
        engine_tex.generate_mipmaps = raw.generate_mipmaps;

        let handle = self.assets.add_texture(engine_tex);
        self.created_textures.insert(key, handle);

        Ok(handle)
    }
}

pub struct GltfLoader<'a> {
    assets: &'a mut AssetServer,
    scene: &'a mut Scene,
    reader: AssetReaderVariant,
    
    intermediate_textures: Vec<IntermediateTexture>,
    created_textures: HashMap<TextureCacheKey, TextureHandle>,
    material_map: Vec<MaterialHandle>,
    node_mapping: Vec<NodeHandle>,
    default_material: Option<MaterialHandle>,
    extensions: HashMap<String, Box<dyn GltfExtensionParser>>,
}

impl<'a> GltfLoader<'a> {
    /// 同步加载入口（向后兼容）
    pub fn load(
        path: &std::path::Path,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<NodeHandle> {
        Self::load_sync(path.to_string_lossy().as_ref(), assets, scene)
    }

    /// 同步加载（内部创建运行时）
    pub fn load_sync(
        source: &str,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<NodeHandle> {
        let rt = get_global_runtime();
        rt.block_on(Self::load_async(source, assets, scene))
    }

    /// 异步加载入口
    pub async fn load_async(
        source: &str,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<NodeHandle> {
        let reader = AssetReaderVariant::from_source(source)?;
        let file_name = AssetReaderVariant::source_filename(source);
        
        let gltf_bytes = reader.read_bytes(file_name).await
            .with_context(|| format!("Failed to read glTF file: {}", source))?;
        
        let gltf = gltf::Gltf::from_slice(&gltf_bytes)
            .context("Failed to parse glTF data")?;

        let buffers = Self::load_buffers_async(&gltf, &reader).await?;

        let mut loader = Self {
            assets,
            scene,
            reader: reader.clone(),
            intermediate_textures: Vec::new(),
            created_textures: HashMap::new(),
            material_map: Vec::new(),
            node_mapping: Vec::with_capacity(gltf.nodes().count()),
            extensions: HashMap::new(),
            default_material: None,
        };

        loader.register_extension(Box::new(KhrMaterialsPbrSpecularGlossiness));
        loader.register_extension(Box::new(KhrMaterialsClearcoat));

        let mut supported_ext = loader.extensions.keys().cloned().collect::<Vec<_>>();
        supported_ext.extend([
            "KHR_materials_emissive_strength".to_string(),
            "KHR_materials_ior".to_string(),
            "KHR_materials_specular".to_string(),
        ]);

        let require_not_supported: Vec<_> = gltf.extensions_required().filter(
            |ext| !supported_ext.contains(&ext.to_string())
        ).collect();

        if !require_not_supported.is_empty() {
            log::warn!("glTF file requires unsupported extensions: {:?}", require_not_supported);
        }

        let used_not_supported: Vec<_> = gltf.extensions_used().filter(
            |ext| !supported_ext.contains(&ext.to_string())
        ).collect();

        if !used_not_supported.is_empty() {
            log::warn!("glTF uses unsupported extensions: {:?}, display may not be correct", used_not_supported);
        }

        loader.load_textures_async(&gltf, &buffers).await?;
        loader.load_materials(&gltf)?;

        for node in gltf.nodes() {
            let handle = loader.create_node_shallow(&node)?;
            loader.node_mapping.push(handle);
        }

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
        
        if let Some(default_scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            for node in default_scene.nodes() {
                let node_handle = loader.node_mapping[node.index()];
                loader.scene.attach(node_handle, root_handle);
            }
        }

        let skeleton_keys = loader.load_skins(&gltf, &buffers)?;

        for node in gltf.nodes() {
            loader.bind_node_mesh_and_skin(&node, &buffers, &skeleton_keys)?;
        }

        let animations = loader.load_animations(&gltf, &buffers)?;

        let mut mixer = AnimationMixer::new();
        for clip in animations {
            let bindings = Binder::bind(loader.scene, root_handle, &clip);
            
            let mut action = AnimationAction::new(clip.into());
            action.bindings = bindings;
            action.enabled = false; 
            action.weight = 0.0;
            
            mixer.add_action(action);
        }
        loader.scene.animation_mixers.insert(root_handle, mixer);

        Ok(root_handle)
    }

    fn register_extension(&mut self, ext: Box<dyn GltfExtensionParser>) {
        self.extensions.insert(ext.name().to_string(), ext);
    }

    fn get_default_material(&mut self) -> MaterialHandle {
        if let Some(mat) = &self.default_material {
            mat.clone()
        } else {
            let mat = self.assets.add_material(Material::new_standard(Vec4::ONE));
            self.default_material = Some(mat.clone());
            mat
        }
    }

    async fn load_buffers_async(gltf: &gltf::Gltf, reader: &AssetReaderVariant) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut tasks = Vec::new();

        for buffer in gltf.buffers() {
            let reader = reader.clone();
            let blob = gltf.blob.clone();
            
            match buffer.source() {
                gltf::buffer::Source::Bin => {
                    tasks.push(tokio::spawn(async move {
                        blob.ok_or_else(|| anyhow::anyhow!("Missing GLB blob"))
                    }));
                }
                gltf::buffer::Source::Uri(uri) => {
                    let uri = uri.to_string();
                    tasks.push(tokio::spawn(async move {
                        reader.read_bytes(&uri).await
                    }));
                }
            }
        }

        let results = try_join_all(tasks).await?;
        
        let mut buffers = Vec::with_capacity(results.len());
        for res in results {
            buffers.push(res?);
        }
        Ok(buffers)
    }

    async fn load_textures_async(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<()> {
        let mut tasks: Vec<_> = Vec::new();

        for (index, texture) in gltf.textures().enumerate() {

            let sampler = texture.sampler();
            let mut generate_mipmaps = false;

            // 1. 准备 Sampler 数据 (这部分逻辑保持不变，为了 Move 进闭包，需要由所有权)
            let engine_sampler = TextureSampler {
                mag_filter: sampler.mag_filter().map(|f| match f {
                    gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
                }).unwrap_or(wgpu::FilterMode::Linear),
                min_filter: sampler.min_filter().map(|f| match f {
                    gltf::texture::MinFilter::Nearest => wgpu::FilterMode::Nearest,
                    gltf::texture::MinFilter::Linear => wgpu::FilterMode::Linear,
                    gltf::texture::MinFilter::NearestMipmapNearest => { generate_mipmaps = true; wgpu::FilterMode::Nearest },
                    gltf::texture::MinFilter::LinearMipmapNearest => { generate_mipmaps = true; wgpu::FilterMode::Linear },
                    gltf::texture::MinFilter::NearestMipmapLinear => { generate_mipmaps = true; wgpu::FilterMode::Nearest },
                    gltf::texture::MinFilter::LinearMipmapLinear => { generate_mipmaps = true; wgpu::FilterMode::Linear },
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
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mipmap_filter: match sampler.min_filter() {
                    Some(gltf::texture::MinFilter::NearestMipmapNearest) | Some(gltf::texture::MinFilter::LinearMipmapNearest) => {
                        generate_mipmaps = true;
                        wgpu::MipmapFilterMode::Nearest
                    },
                    Some(gltf::texture::MinFilter::NearestMipmapLinear) | Some(gltf::texture::MinFilter::LinearMipmapLinear) => {
                        generate_mipmaps = true;
                        wgpu::MipmapFilterMode::Linear
                    },
                    _ => wgpu::MipmapFilterMode::Linear,
                },
                ..Default::default()
            };

            let name = texture.name().map(|s| s.to_string());
            let reader = self.reader.clone();

            // 2. 准备数据源 (Clone 数据以移入任务)
            let img_source = texture.source().source();
            let (uri_opt, buffer_data) = match img_source {
                gltf::image::Source::Uri { uri, .. } => (Some(uri.to_string()), None),
                gltf::image::Source::View { view, .. } => {
                    let start = view.offset();
                    let end = start + view.length();
                    (None, Some(buffers[view.buffer().index()][start..end].to_vec()))
                }
            };

            // 3. Spawn 组合任务：下载 + 解码
            tasks.push(tokio::spawn(async move {
                // A. 下载 / 获取数据
                let img_bytes = if let Some(uri) = uri_opt {
                    reader.read_bytes(&uri).await?
                } else {
                    buffer_data.unwrap()
                };

                // B. 并发解码 (CPU 密集型，使用 spawn_blocking 避免阻塞异步运行时)
                let img_data = tokio::task::spawn_blocking(move || {
                    image::load_from_memory(&img_bytes)
                        .with_context(|| format!("Failed to decode texture {}", index))
                        .map(|img| img.to_rgba8())
                }).await??; // 解两层 Result: JoinError 和 anyhow::Result

                // C. 直接构造结果
                Ok(IntermediateTexture {
                    name,
                    width: img_data.width(),
                    height: img_data.height(),
                    image_data: img_data.into_vec(),
                    sampler: engine_sampler,
                    generate_mipmaps,
                })
            }));

        }

        // 4. 等待所有任务完成
        let results: Vec<anyhow::Result<IntermediateTexture>> = try_join_all(tasks).await?;

        // 5. 填入结果
        for res in results {
            // res 是 anyhow::Result<IntermediateTexture>
            self.intermediate_textures.push(res?);
        }

        Ok(())
    }

    fn get_or_create_texture(
        &mut self,
        gltf_texture_index: usize,
        is_srgb: bool,
    ) -> anyhow::Result<TextureHandle> {
        let key = TextureCacheKey {
            gltf_texture_index,
            is_srgb,
        };

        if let Some(&handle) = self.created_textures.get(&key) {
            return Ok(handle);
        }

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
        engine_tex.generate_mipmaps = raw.generate_mipmaps;

        let handle = self.assets.add_texture(engine_tex);
        self.created_textures.insert(key, handle);

        Ok(handle)
    }

    fn setup_texture_map(&mut self, texture_slot: &mut TextureSlot, info: &gltf::texture::Info, is_srgb: bool) -> anyhow::Result<()> {
        let tex_handle = self.get_or_create_texture(info.texture().index(), is_srgb)?;
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

    fn load_materials(&mut self, gltf: &gltf::Gltf) -> anyhow::Result<()> {
        for material in gltf.materials() {
            let pbr = material.pbr_metallic_roughness();

            let base_color_factor = Vec4::from_array(pbr.base_color_factor());
            let mut mat = MeshPhysicalMaterial::new(base_color_factor);

            mat.set_metalness(pbr.metallic_factor());
            mat.set_roughness(pbr.roughness_factor());
            mat.set_emissive(Vec3::from_array(material.emissive_factor()));
            
            if let Some(info) = pbr.base_color_texture() {
                self.setup_texture_map(&mut mat.map, &info, true)?;
            }

            if let Some(info) = pbr.metallic_roughness_texture() {
                self.setup_texture_map(&mut mat.roughness_map, &info, false)?;
                self.setup_texture_map(&mut mat.metalness_map, &info, false)?;
            }

            if let Some(info) = material.normal_texture() {
                let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                mat.normal_map.texture = Some(tex_handle);
                mat.normal_map.channel = info.tex_coord() as u8;
            }

            if let Some(info) = material.occlusion_texture() {
                let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                mat.ao_map.texture = Some(tex_handle);
                mat.ao_map.channel = info.tex_coord() as u8;
                mat.set_ao_map_intensity(info.strength());
            }

            if let Some(info) = material.emissive_texture() {
                self.setup_texture_map(&mut mat.emissive_map, &info, true)?;
            }

            mat.set_side(if material.double_sided() { crate::resources::material::Side::Double } else { crate::resources::material::Side::Front });
    
            let alpha_mode = match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => {
                    let cut_off = material.alpha_cutoff().unwrap_or(0.5);
                    AlphaMode::Mask(cut_off)
                },
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };

            mat.set_alpha_mode(alpha_mode);

            if let Some(info) = material.emissive_strength() {
                mat.set_emissive_intensity(info);
            }

            if let Some(info) = material.ior() {
                mat.set_ior(info);
            }

            if let Some(specular) = material.specular() {
                mat.set_specular_color(Vec3::from_array(specular.specular_color_factor()));
                mat.set_specular_intensity(specular.specular_factor());

                if let Some(info) = specular.specular_color_texture() {
                    self.setup_texture_map(&mut mat.specular_map, &info, true)?;
                }

                if let Some(info) = specular.specular_texture() {
                    self.setup_texture_map(&mut mat.specular_intensity_map, &info, false)?;
                }
            }

            let mut engine_mat = Material::from(mat);

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

            if let Some(extensions_map) = material.extensions() {
                let mut ctx = LoadContext {
                    assets: &mut self.assets, 
                    material_map: &self.material_map,
                    intermediate_textures: &self.intermediate_textures,
                    created_textures: &mut self.created_textures,
                    _phantom: std::marker::PhantomData,
                };

                for (name, value) in extensions_map {
                    if let Some(handler) = self.extensions.get_mut(name) {
                        handler.on_load_material(&mut ctx, &material, &mut engine_mat, value)?;
                    }
                }
            }

            let physical_mat = engine_mat.as_any_mut().downcast_mut::<MeshPhysicalMaterial>().unwrap();
            physical_mat.flush_texture_transforms();
            physical_mat.notify_pipeline_dirty();

            let handle = self.assets.add_material(engine_mat);
            self.material_map.push(handle);
        }
        Ok(())
    }

    fn create_node_shallow(&mut self, node: &gltf::Node) -> anyhow::Result<NodeHandle> {
        let node_name = node.name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Node_{}", node.index()));
        let handle = self.scene.create_node_with_name(node_name.as_str());

        if let Some(engine_node) = self.scene.get_node_mut(handle) {
            let (t, r, s) = node.transform().decomposed();
            engine_node.transform.position = Vec3::from_array(t);
            engine_node.transform.rotation = Quat::from_array(r);
            engine_node.transform.scale = Vec3::from_array(s);
        }

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
            
            let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
            let ibms: Vec<Affine3A> = if let Some(iter) = reader.read_inverse_bind_matrices() {
                iter.map(|m| {
                    let mat = Mat4::from_cols_array_2d(&m);
                    Affine3A::from_mat4(mat)
                }).collect()
            } else {
                vec![Affine3A::IDENTITY; skin.joints().count()]
            };

            let bones: Vec<NodeHandle> = skin.joints()
                .map(|node| self.node_mapping[node.index()]) 
                .collect();

            let joints: Vec<_> = skin.joints().collect();
            let joint_indices: std::collections::HashSet<usize> = joints.iter().map(|n| n.index()).collect();
            
            let mut child_joint_indices = std::collections::HashSet::new();
            for node in &joints {
                for child in node.children() {
                    if joint_indices.contains(&child.index()) {
                        child_joint_indices.insert(child.index());
                    }
                }
            }

            let root_bone_index = 'block: {
                if let Some(skeleton_root) = skin.skeleton() {
                    if let Some(index) = joints.iter().position(|n| n.index() == skeleton_root.index()) {
                        break 'block index;
                    }
                }

                for (i, node) in joints.iter().enumerate() {
                    if !child_joint_indices.contains(&node.index()) {
                        break 'block i; 
                    }
                }
                
                0 
            };

            let skeleton = Skeleton::new(name, bones, ibms, root_bone_index);
    
            let key = self.scene.skeleton_pool.insert(skeleton); 
            skeleton_keys.push(key);
        }

        Ok(skeleton_keys)
    }
    
    fn build_engine_mesh(
        &mut self,
        primitive: &gltf::Primitive,
        buffers: &[Vec<u8>]
    ) -> anyhow::Result<crate::resources::mesh::Mesh> {
        let geo_handle = self.load_primitive_geometry(primitive, buffers)?;
        
        let mat_idx = primitive.material().index();
        let mat_handle = if let Some(idx) = mat_idx {
            self.material_map[idx]
        } else {
            self.get_default_material()
        };

        let mut engine_mesh = crate::resources::mesh::Mesh::new(geo_handle, mat_handle);
        
        if let Some(geometry) = self.assets.get_geometry(geo_handle) {
            if geometry.has_morph_targets() {
                engine_mesh.init_morph_targets(
                    geometry.morph_target_count,
                    geometry.morph_vertex_count
                );
            }
        }

        Ok(engine_mesh)
    }

    fn bind_node_mesh_and_skin(
        &mut self,
        node: &gltf::Node,
        buffers: &[Vec<u8>],
        skeleton_keys: &[SkeletonKey]
    ) -> anyhow::Result<()> {
        let engine_node_handle = self.node_mapping[node.index()];

        if let Some(mesh) = node.mesh() {
            let primitives: Vec<_> = mesh.primitives().collect();

            match primitives.len() {
                0 => {},
                1 => {
                    let engine_mesh = self.build_engine_mesh(&primitives[0], buffers)?;
                    self.scene.set_mesh(engine_node_handle, engine_mesh);
                }
                _ => {
                    for primitive in primitives {
                        let engine_mesh = self.build_engine_mesh(&primitive, buffers)?;
                        
                        let sub_node_handle = self.scene.create_node();
                        self.scene.attach(sub_node_handle, engine_node_handle);
                        self.scene.set_mesh(sub_node_handle, engine_mesh);
                    }
                }
            }
        }

        if let Some(skin) = node.skin() {
            let skeleton_key = skeleton_keys[skin.index()];
            self.scene.bind_skeleton(engine_node_handle, skeleton_key, BindMode::Attached);
        }

        Ok(())
    }

    fn build_interleaved_buffer(
        label: &str,
        channels: Vec<InterleaveChannel>,
        vertex_count: usize,
    ) -> Option<(BufferRef, Vec<(String, Attribute)>)> {
        if channels.is_empty() || vertex_count == 0 {
            return None;
        }

        let total_stride: usize = channels.iter().map(|c| c.item_size).sum();
        let buffer_size = total_stride * vertex_count;
        
        let mut interleaved_data = vec![0u8; buffer_size];

        let mut offsets = Vec::with_capacity(channels.len());
        let mut current_offset = 0;
        for ch in &channels {
            offsets.push(current_offset);
            current_offset += ch.item_size;
        }

        for i in 0..vertex_count {
            let vertex_start = i * total_stride;
            for (ch_idx, channel) in channels.iter().enumerate() {
                let src_start = i * channel.item_size;
                let src_end = src_start + channel.item_size;
                
                if src_end <= channel.data.len() {
                    let dest_start = vertex_start + offsets[ch_idx];
                    interleaved_data[dest_start..dest_start + channel.item_size]
                        .copy_from_slice(&channel.data[src_start..src_end]);
                }
            }
        }

        let buffer = BufferRef::new(
            buffer_size,
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            Some(label),
        );
        let data_arc = Some(Arc::new(interleaved_data));

        let mut attributes = Vec::new();
        for (i, channel) in channels.into_iter().enumerate() {
            attributes.push((
                channel.name,
                Attribute::new_interleaved(
                    buffer.clone(),
                    data_arc.clone(),
                    channel.format,
                    offsets[i] as u64,
                    vertex_count as u32,
                    total_stride as u64,
                    VertexStepMode::Vertex,
                ),
            ));
        }

        Some((buffer, attributes))
    }

    fn load_primitive_geometry(
        &mut self,
        primitive: &gltf::Primitive,
        buffers: &[Vec<u8>]
    ) -> anyhow::Result<GeometryHandle> {
        let mut geometry = Geometry::new();
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let positions: Vec<[f32; 3]> = reader.read_positions()
            .map(|iter| iter.collect())
            .unwrap_or_default();
        
        let vertex_count = positions.len();
        if vertex_count == 0 {
            return Ok(self.assets.add_geometry(geometry));
        }

        // positions 属性是必须的
        geometry.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));

        // 设置索引
        if let Some(iter) = reader.read_indices() {
            let indices: Vec<u32> = iter.into_u32().collect();
            geometry.set_indices_u32(&indices);
        }

        let mut surface_channels = Vec::new();

        if let Some(iter) = reader.read_normals() {
            surface_channels.push(InterleaveChannel::from_iter("normal", iter, VertexFormat::Float32x3));
        }else{
            // 如果没有法线，则计算法线
            geometry.compute_vertex_normals();
        }
        
        if let Some(iter) = reader.read_tangents() {
            surface_channels.push(InterleaveChannel::from_iter("tangent", iter, VertexFormat::Float32x4));
        }

        for i in 0..4 {
            if let Some(iter) = reader.read_tex_coords(i).map(|r| r.into_f32()) {
                let name = if i == 0 { "uv".to_string() } else { format!("uv{}", i) };
                surface_channels.push(InterleaveChannel::from_iter(&name, iter, VertexFormat::Float32x2));
            }
        }

        if let Some(iter) = reader.read_colors(0).map(|r| r.into_rgba_f32()) {
            surface_channels.push(InterleaveChannel::from_iter("color", iter, VertexFormat::Float32x4));
        }

        if let Some((_, attrs)) = Self::build_interleaved_buffer(
            "SurfaceBuffer", 
            surface_channels, 
            vertex_count
        ) {
            for (name, attr) in attrs {
                geometry.set_attribute(&name, attr);
            }
        }

        let mut skin_channels = Vec::new();

        if let Some(iter) = reader.read_joints(0).map(|r| r.into_u16()) {
            skin_channels.push(InterleaveChannel::from_iter("joints", iter, VertexFormat::Uint16x4));
        }
        
        if let Some(iter) = reader.read_weights(0).map(|r| r.into_f32()) {
            skin_channels.push(InterleaveChannel::from_iter("weights", iter, VertexFormat::Float32x4));
        }

        if let Some((_, attrs)) = Self::build_interleaved_buffer(
            "SkinningBuffer", 
            skin_channels, 
            vertex_count
        ) {
            for (name, attr) in attrs {
                geometry.set_attribute(&name, attr);
            }
        }



        let get_buffer_data = |buffer: gltf::Buffer| -> Option<&[u8]> {
            buffers.get(buffer.index()).map(|v| v.as_slice())
        };

        for target in primitive.morph_targets() {
            if let Some(accessor) = target.positions() {
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
                    let attr = Attribute::new_planar(&data, VertexFormat::Float32x3);
                    geometry.morph_attributes.entry("position".to_string())
                        .or_default()
                        .push(attr);
                }
            }

            if let Some(accessor) = target.normals() {
                if let Some(iter) = gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer_data) {
                    let data: Vec<[f32; 3]> = iter.collect();
                    let attr = Attribute::new_planar(&data, VertexFormat::Float32x3);
                    geometry.morph_attributes.entry("normal".to_string())
                        .or_default()
                        .push(attr);
                }
            }

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

        geometry.topology = match primitive.mode() {
            gltf::mesh::Mode::Points => PrimitiveTopology::PointList,
            gltf::mesh::Mode::Lines => PrimitiveTopology::LineList,
            gltf::mesh::Mode::LineLoop => PrimitiveTopology::LineList, // TODO: LineLoop not directly supported
            gltf::mesh::Mode::LineStrip => PrimitiveTopology::LineStrip,
            gltf::mesh::Mode::Triangles => PrimitiveTopology::TriangleList,
            gltf::mesh::Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
            gltf::mesh::Mode::TriangleFan => PrimitiveTopology::TriangleList, // TODO: TriangleFan not directly supported
        };
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
                
                let node_name = gltf_node.name().map(|s| s.to_string())
                    .unwrap_or_else(|| format!("Node_{}", gltf_node.index()));
                        
                let times: Vec<f32> = reader.read_inputs().unwrap().collect();
                
                let interpolation = match channel.sampler().interpolation() {
                    gltf::animation::Interpolation::Linear => InterpolationMode::Linear,
                    gltf::animation::Interpolation::Step => InterpolationMode::Step,
                    gltf::animation::Interpolation::CubicSpline => InterpolationMode::CubicSpline,
                };

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

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.metalness = 0.0;
            uniforms.roughness = 1.0;
            uniforms.ior = 1000.0;
            uniforms.specular_color = Vec3::from_array(sg.specular_factor());
            uniforms.specular_intensity = 1.0;
            uniforms.color = Vec4::from_array(sg.diffuse_factor());
        }

        if let Some(diffuse_tex) = sg.diffuse_texture() {
            let tex_handle = ctx.get_or_create_texture(diffuse_tex.texture().index(), true)?;
            physical_mat.map.texture = Some(tex_handle);
        }

        if let Some(sg_tex_info) = sg.specular_glossiness_texture() {
            let tex_index = sg_tex_info.texture().index();
            let glossiness_factor = sg.glossiness_factor();
            
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
            physical_mat.metalness_map.transform = transform;
        } else {
            let glossiness_factor = sg.glossiness_factor();
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.roughness = 1.0 - glossiness_factor;
        }

        Ok(())
    }
}

struct KhrMaterialsClearcoat;

impl GltfExtensionParser for KhrMaterialsClearcoat {
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

        if let Some(clearcoat_tex_info) = clearcoat_info.get("clearcoatTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_tex_info, &mut physical_mat.clearcoat_map, false);
        }

        if let Some(clearcoat_roughness_tex_info) = clearcoat_info.get("clearcoatRoughnessTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_roughness_tex_info, &mut physical_mat.clearcoat_roughness_map, false);
        }

        if let Some(clearcoat_normal_tex_info) = clearcoat_info.get("clearcoatNormalTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_normal_tex_info, &mut physical_mat.clearcoat_normal_map, false);
        }

        Ok(())
    }
}
