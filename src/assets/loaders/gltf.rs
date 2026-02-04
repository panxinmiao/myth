use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use glam::{Affine3A, Mat4, Quat, Vec2, Vec3, Vec4};
use futures::future::try_join_all;
use crate::resources::material::AlphaMode;
use crate::resources::{Material, MeshPhysicalMaterial, TextureSampler, TextureSlot, TextureTransform, PhysicalFeatures};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::texture::Texture;
use crate::resources::buffer::BufferRef;
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use crate::assets::io::{AssetReaderVariant, AssetSource};
use crate::assets::prefab::{Prefab, PrefabNode, PrefabSkeleton};
use crate::animation::clip::{AnimationClip, Track, TrackMeta, TrackData};
use crate::animation::tracks::{KeyframeTrack, InterpolationMode};
use crate::animation::binding::TargetPath;
use crate::animation::values::MorphWeightData;
use crate::resources::mesh::MAX_MORPH_TARGETS;
use wgpu::{BufferUsages, PrimitiveTopology, TextureFormat, VertexFormat, VertexStepMode};
use anyhow::Context;
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

#[cfg(not(target_arch = "wasm32"))]
// 仅用于同步加载的全局 Runtime
fn get_global_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("Failed to create global asset loader runtime")
    })
}

fn decode_data_uri(uri: &str) -> anyhow::Result<Vec<u8>> {
    if uri.starts_with("data:") {
        let comma = uri.find(',').ok_or_else(|| anyhow::anyhow!("Invalid Data URI"))?;
        let header = &uri[0..comma];
        let data = &uri[comma + 1..];

        if header.ends_with(";base64") {
            use base64::{Engine as _, engine::general_purpose};
            let bytes = general_purpose::STANDARD.decode(data)?;
            Ok(bytes)
        } else {
            Err(anyhow::anyhow!("Unsupported Data URI encoding (only base64 supported)"))
        }
    } else {
        Err(anyhow::anyhow!("Not a Data URI"))
    }
}

fn parse_transform_from_json(texture_slot: &mut TextureSlot, transform_val: &Value) {
    if let Some(offset) = transform_val.get("offset").and_then(|v| v.as_array()) {
        let x = offset.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let y = offset.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        texture_slot.transform.offset = Vec2::new(x, y);
    }

    if let Some(scale) = transform_val.get("scale").and_then(|v| v.as_array()) {
        let x = scale.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let y = scale.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        texture_slot.transform.scale = Vec2::new(x, y);
    }

    if let Some(rotation) = transform_val.get("rotation").and_then(|v| v.as_f64()) {
        texture_slot.transform.rotation = rotation as f32;
    }

    if let Some(tex_coord) = transform_val.get("texCoord").and_then(|v| v.as_u64()) {
        texture_slot.channel = tex_coord as u8;
    }
}

#[allow(dead_code)]
fn sanitize_gltf_data(data: &[u8]) -> anyhow::Result<Cow<'_, [u8]>> {
    if data.starts_with(b"glTF") {
        sanitize_glb(data)
    } else {
        sanitize_json(data)
    }
}

#[allow(dead_code)]
fn sanitize_json(data: &[u8]) -> anyhow::Result<Cow<'_, [u8]>> {
    let mut root: Value = serde_json::from_slice(data)
        .context("Failed to parse GLTF JSON for sanitization")?;

    if patch_json_value(&mut root) {
        let patched = serde_json::to_vec(&root)?;
        Ok(Cow::Owned(patched))
    } else {
        Ok(Cow::Borrowed(data))
    }
}

#[allow(dead_code)]
fn sanitize_glb(data: &[u8]) -> anyhow::Result<Cow<'_, [u8]>> {
    if data.len() < 12 { return Ok(Cow::Borrowed(data)); }
    
    let version = u32::from_le_bytes(data[4..8].try_into()?);
    if version != 2 { return Ok(Cow::Borrowed(data)); }

    let chunk0_len = u32::from_le_bytes(data[12..16].try_into()?) as usize;
    let chunk0_type = u32::from_le_bytes(data[16..20].try_into()?);
    
    if chunk0_type != 0x4E4F534A { return Ok(Cow::Borrowed(data)); }

    let json_bytes = &data[20..20 + chunk0_len];
    
    let mut root: Value = serde_json::from_slice(json_bytes)?;
    if !patch_json_value(&mut root) {
        return Ok(Cow::Borrowed(data));
    }

    let new_json_bytes = serde_json::to_vec(&root)?;
    let padding = (4 - (new_json_bytes.len() % 4)) % 4;
    let new_chunk0_len = new_json_bytes.len() + padding;
    
    let mut new_glb = Vec::with_capacity(data.len() + (new_chunk0_len as isize - chunk0_len as isize).abs() as usize);
    
    new_glb.extend_from_slice(&data[0..8]);
    new_glb.extend_from_slice(&[0, 0, 0, 0]); 

    new_glb.extend_from_slice(&(new_chunk0_len as u32).to_le_bytes());
    new_glb.extend_from_slice(&chunk0_type.to_le_bytes());

    new_glb.extend_from_slice(&new_json_bytes);
    for _ in 0..padding { new_glb.push(0x20); }

    let rest_offset = 20 + chunk0_len;
    if rest_offset < data.len() {
        new_glb.extend_from_slice(&data[rest_offset..]);
    }

    let total_len = new_glb.len() as u32;
    new_glb[8..12].copy_from_slice(&total_len.to_le_bytes());

    Ok(Cow::Owned(new_glb))
}

#[allow(dead_code)]
fn patch_json_value(root: &mut Value) -> bool {
    let mut changed = false;

    if let Some(anims) = root.get_mut("animations").and_then(|v| v.as_array_mut()) {
        for anim in anims {
            if let Some(channels) = anim.get_mut("channels").and_then(|v| v.as_array_mut()) {
                let old_len = channels.len();
                channels.retain(|ch| {
                    ch.get("target")
                        .and_then(|t| t.get("node"))
                        .is_some()
                });
                if channels.len() != old_len {
                    changed = true;
                    log::warn!("Sanitizer: Removed {} invalid animation channels (missing node)", old_len - channels.len());
                }
            }
        }
    }

    changed
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

// 辅助结构体，方便传递解码结果
struct DecodedImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
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
    pub assets: &'a AssetServer,
    pub material_map: &'a [MaterialHandle],
    intermediate_textures: &'a [IntermediateTexture],
    created_textures: &'a mut HashMap<TextureCacheKey, TextureHandle>,
    _phantom: std::marker::PhantomData<&'b ()>,
}

pub trait GltfExtensionParser {
    fn name(&self) -> &str;

    #[allow(unused_variables)]
    fn on_load_material(&mut self, ctx: &mut LoadContext, gltf_mat: &gltf::Material, engine_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
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
                .and_then(|exts| exts.get("KHR_texture_transform")){
                    parse_transform_from_json(texture_slot, transform);
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

        let handle = self.assets.textures.add(engine_tex);
        self.created_textures.insert(key, handle);

        Ok(handle)
    }
}

/// glTF Loader
/// 
/// Supports synchronous and asynchronous loading, outputs `Prefab` data structure,
/// instantiated into the scene via `Scene::instantiate()`.
pub struct GltfLoader {
    assets: Arc<AssetServer>,
    reader: AssetReaderVariant,
    
    intermediate_textures: Vec<IntermediateTexture>,
    created_textures: HashMap<TextureCacheKey, TextureHandle>,
    material_map: Vec<MaterialHandle>,
    default_material: Option<MaterialHandle>,
    extensions: HashMap<String, Box<dyn GltfExtensionParser + Send>>,
    
    prefab_nodes: Vec<PrefabNode>,
    prefab_skeletons: Vec<PrefabSkeleton>,
}

impl GltfLoader {
    
    fn new_loader(
        assets: Arc<AssetServer>,
        reader: AssetReaderVariant,
        gltf: &gltf::Gltf,
    ) -> Self {
        let mut loader = Self {
            assets,
            reader,
            intermediate_textures: Vec::new(),
            created_textures: HashMap::new(),
            material_map: Vec::new(),
            extensions: HashMap::new(),
            default_material: None,
            prefab_nodes: Vec::with_capacity(gltf.nodes().count()),
            prefab_skeletons: Vec::new(),
        };

        // Register core extensions
        loader.register_extension(Box::new(KhrMaterialsPbrSpecularGlossiness));
        loader.register_extension(Box::new(KhrMaterialsClearcoat));
        loader.register_extension(Box::new(KhrMaterialsSheen));
        loader.register_extension(Box::new(KhrMaterialsIridescence));
        loader.register_extension(Box::new(KhrMaterialsAnisotropy));
        loader.register_extension(Box::new(KhrMaterialsTransmission));
        loader.register_extension(Box::new(KhrMaterialsVolume));
        loader.register_extension(Box::new(KhrMaterialsDispersion));

        // Validation / Logging
        let mut supported_ext = loader.extensions.keys().cloned().collect::<Vec<_>>();
        supported_ext.extend([
            "KHR_materials_emissive_strength".to_string(),
            "KHR_materials_ior".to_string(),
            "KHR_materials_specular".to_string(),
            "KHR_texture_transform".to_string(),
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

        loader
    }

    fn register_extension(&mut self, ext: Box<dyn GltfExtensionParser + Send>) {
        self.extensions.insert(ext.name().to_string(), ext);
    }

    async fn load_inner(mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<Arc<Prefab>> {
        self.load_textures_async(gltf, buffers).await?;
        self.load_materials(gltf)?;
        let prefab = self.build_prefab(gltf, buffers)?;
        Ok(Arc::new(prefab))
    }

    /// Synchronous load entry point (backwards compatible) - Native only
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(
        source: impl AssetSource,
        assets: impl Into<Arc<AssetServer>>
    ) -> anyhow::Result<Arc<Prefab>> {
        Self::load_sync(source, assets)
    }

    /// Synchronous load (creates runtime internally) - Native only
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_sync(
        source: impl AssetSource,
        assets: impl Into<Arc<AssetServer>>
    ) -> anyhow::Result<Arc<Prefab>> {
        let rt = get_global_runtime();
        rt.block_on(Self::load_async(source, assets))
    }

    /// Load asynchronously from a source URI (File path or HTTP URL)
    pub async fn load_async(
        source: impl AssetSource,
        assets: impl Into<Arc<AssetServer>>
    ) -> anyhow::Result<Arc<Prefab>> {
        let reader = AssetReaderVariant::new(&source)?;
        let filename = source.filename().unwrap_or(std::borrow::Cow::Borrowed("unknown"));
        
        let gltf_bytes = reader.read_bytes(&filename).await
            .with_context(|| format!("Failed to read glTF file: {}", source.uri()))?;
        
        // 1. Parse glTF
        let gltf = Self::parse_gltf_bytes(&gltf_bytes)?;

        // 2. Load Buffers
        let buffers = Self::load_buffers_async(&gltf, &reader).await?;

        // 3. Init Loader
        let loader = Self::new_loader(assets.into(), reader, &gltf);

        // 4. Execute common loading pipeline
        loader.load_inner(&gltf, &buffers).await
    }

    /// Load from in-memory bytes (GLB or JSON)
    pub async fn load_from_bytes(
        gltf_bytes: Vec<u8>,
        assets: impl Into<Arc<AssetServer>>
    ) -> anyhow::Result<Arc<Prefab>> {
        // 1. Parse glTF
        let gltf = Self::parse_gltf_bytes(&gltf_bytes)?;

        // 2. Create a dummy reader. 
        // For load_from_bytes, we generally expect resources to be embedded (GLB) or Data URIs.
        // (unless we are in a context where "." makes sense).
        // #[cfg(not(target_arch = "wasm32"))]
        let s = ".".to_string();
        let reader = AssetReaderVariant::new(&s)?;
        // #[cfg(target_arch = "wasm32")]
        // let reader = AssetReaderVariant::new(".")?;

        // 3. Load Buffers (Using common async logic)
        let buffers = Self::load_buffers_async(&gltf, &reader).await?;

        // 4. Init Loader
        let loader = Self::new_loader(assets.into(), reader, &gltf);

        // 5. Execute common loading pipeline
        loader.load_inner(&gltf, &buffers).await
    }

    fn parse_gltf_bytes(bytes: &[u8]) -> anyhow::Result<gltf::Gltf> {
        // let sanitized_bytes = sanitize_gltf_data(bytes)
        //     .context("Failed to sanitize glTF data")?;

        match gltf::Gltf::from_slice_without_validation(bytes) {
            Ok(g) => Ok(g),
            Err(err) => {
                log::error!("GLTF Parse Error Details: {:?}", err);
                Err(anyhow::anyhow!("Failed to parse glTF: {}", err))
            }
        }
    }

    fn get_default_material(&mut self) -> MaterialHandle {
        if let Some(mat) = &self.default_material {
            *mat
        } else {
            let mat = self.assets.materials.add(Material::new_physical(Vec4::ONE));
            self.default_material = Some(mat);
            mat
        }
    }

    /// Load buffers asynchronously
    async fn load_buffers_async(gltf: &gltf::Gltf, reader: &AssetReaderVariant) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut tasks = Vec::new();

        for buffer in gltf.buffers() {
            let reader = reader.clone();
            let blob = gltf.blob.clone();


            let future = async move {
                match buffer.source() {
                    gltf::buffer::Source::Bin => {
                        blob.ok_or_else(|| anyhow::anyhow!("Missing GLB blob"))
                    }
                    gltf::buffer::Source::Uri(uri) => {
                        if uri.starts_with("data:") {
                            decode_data_uri(uri)
                        } else {
                            reader.read_bytes(uri).await
                        }
                    }
                }
            };
            tasks.push(future);
        }

        try_join_all(tasks).await

    }

    /// Load textures asynchronously - Native version using tokio::spawn
    async fn load_textures_async(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<()> {
        let mut futures = Vec::new();

        for (index, texture) in gltf.textures().enumerate() {
            let (engine_sampler, generate_mipmaps) = Self::create_texture_sampler(&texture);

            let name = texture.name().map(|s| s.to_string());
            let reader = self.reader.clone();

            let img_source = texture.source().source();

            let uri_opt = match img_source {
                gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
                _ => None,
            };
            
            let buffer_view_data = match img_source {
                gltf::image::Source::View { view, .. } => {
                    let start = view.offset();
                    let end = start + view.length();
                    Some(buffers[view.buffer().index()][start..end].to_vec())
                },
                _ => None,
            };

            // 创建加载和解码任务
            let future = async move {
                // 1. 获取字节流 (IO)
                let img_bytes = if let Some(uri) = uri_opt {
                    if uri.starts_with("data:") {
                        decode_data_uri(&uri)?
                    } else {
                        reader.read_bytes(&uri).await?
                    }
                } else {
                    buffer_view_data.unwrap()
                };

                // 2. 解码图片 (CPU 密集型)
                // Native: 放入 blocking 线程池
                #[cfg(not(target_arch = "wasm32"))]
                let img_data = tokio::task::spawn_blocking(move || {
                    Self::decode_image_cpu_work(&img_bytes, index)
                }).await??;

                // WASM: 直接在当前线程执行 (或者未来接入 WebWorker)
                #[cfg(target_arch = "wasm32")]
                let img_data = Self::decode_image_cpu_work(&img_bytes, index)?;

                Ok::<IntermediateTexture, anyhow::Error>(IntermediateTexture {
                    name,
                    width: img_data.width,
                    height: img_data.height,
                    image_data: img_data.data,
                    sampler: engine_sampler,
                    generate_mipmaps,
                })
            };

            futures.push(future);

        }

        let results = try_join_all(futures).await?;

        for res in results {
            self.intermediate_textures.push(res);
        }

        Ok(())
    }


    // 纯 CPU 解码逻辑，剥离出来方便在不同上下文调用
    fn decode_image_cpu_work(img_bytes: &[u8], index: usize) -> anyhow::Result<DecodedImage> {
         let img = image::load_from_memory(img_bytes)
             .with_context(|| format!("Failed to decode texture {}", index))?;
         let rgba = img.to_rgba8();
         Ok(DecodedImage {
             width: rgba.width(),
             height: rgba.height(),
             data: rgba.into_vec(),
         })
    }

    /// Helper function to create texture sampler from glTF texture
    fn create_texture_sampler(texture: &gltf::Texture) -> (TextureSampler, bool) {
        let sampler = texture.sampler();
        let mut generate_mipmaps = false;

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

        (engine_sampler, generate_mipmaps)
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

        let handle = self.assets.textures.add(engine_tex);
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
            let mat = MeshPhysicalMaterial::new(base_color_factor);

            {
                let mut uniforms = mat.uniforms.write();
                let mut textures = mat.textures.write();
                let mut settings = mat.settings.write();

                uniforms.metalness = pbr.metallic_factor();
                uniforms.roughness = pbr.roughness_factor();
                uniforms.emissive = Vec3::from_array(material.emissive_factor());

                if let Some(info) = pbr.base_color_texture() {
                    self.setup_texture_map(&mut textures.map, &info, true)?;
                }

                if let Some(info) = pbr.metallic_roughness_texture() {
                    self.setup_texture_map(&mut textures.roughness_map, &info, false)?;
                    self.setup_texture_map(&mut textures.metalness_map, &info, false)?;
                }

                if let Some(info) = material.normal_texture() {
                    let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                    textures.normal_map.texture = Some(tex_handle);
                    textures.normal_map.channel = info.tex_coord() as u8;
                    uniforms.normal_scale = Vec2::splat(info.scale());

                    let json_material = material.index()
                        .and_then(|i| gltf.document.materials().nth(i));

                    if let Some(json_mat) = json_material {
                        if let Some(json_normal) = &json_mat.normal_texture() {
                            if let Some(transform_val) = json_normal.extensions()
                                .and_then(|exts| exts.get("KHR_texture_transform")){
                                    parse_transform_from_json(&mut textures.normal_map, transform_val);
                            }
                        }
                    }
                }

                if let Some(info) = material.occlusion_texture() {
                    let tex_handle = self.get_or_create_texture(info.texture().index(), false)?;
                    textures.ao_map.texture = Some(tex_handle);
                    textures.ao_map.channel = info.tex_coord() as u8;
                    uniforms.ao_map_intensity = info.strength();

                    let json_material = material.index()
                        .and_then(|i| gltf.document.materials().nth(i));

                    if let Some(json_mat) = json_material {
                        if let Some(json_occlusion) = &json_mat.occlusion_texture() {
                            if let Some(transform_val) = json_occlusion.extensions()
                                .and_then(|exts| exts.get("KHR_texture_transform")){
                                    parse_transform_from_json(&mut textures.ao_map, transform_val);
                            }
                        }
                    }
                }

                if let Some(info) = material.emissive_texture() {
                    self.setup_texture_map(&mut textures.emissive_map, &info, true)?;
                }

                settings.side = if material.double_sided() { crate::resources::material::Side::Double } else { crate::resources::material::Side::Front };
        
                let alpha_mode = match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => {
                        let cut_off = material.alpha_cutoff().unwrap_or(0.5);
                        AlphaMode::Mask(cut_off)
                    },
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                };

                settings.alpha_mode = alpha_mode;

                if let Some(info) = material.emissive_strength() {
                    uniforms.emissive_intensity = info;
                }

                if let Some(info) = material.ior() {
                    uniforms.ior = info;
                }

                if let Some(specular) = material.specular() {
                    uniforms.specular_color = Vec3::from_array(specular.specular_color_factor());
                    uniforms.specular_intensity = specular.specular_factor();

                    if let Some(info) = specular.specular_color_texture() {
                        self.setup_texture_map(&mut textures.specular_map, &info, true)?;
                    }

                    if let Some(info) = specular.specular_texture() {
                        self.setup_texture_map(&mut textures.specular_intensity_map, &info, false)?;
                    }
                }
            }



            if material.pbr_specular_glossiness().is_some() {
                if let Some(handler) = self.extensions.get_mut("KHR_materials_pbrSpecularGlossiness") {
                    let mut ctx = LoadContext {
                        assets: &self.assets, 
                        material_map: &self.material_map,
                        intermediate_textures: &self.intermediate_textures,
                        created_textures: &mut self.created_textures,
                        _phantom: std::marker::PhantomData,
                    };
                    handler.on_load_material(&mut ctx, &material, &mat, &Value::Null)?;
                }
            }

            if let Some(extensions_map) = material.extensions() {
                let mut ctx = LoadContext {
                    assets: &self.assets, 
                    material_map: &self.material_map,
                    intermediate_textures: &self.intermediate_textures,
                    created_textures: &mut self.created_textures,
                    _phantom: std::marker::PhantomData,
                };

                for (name, value) in extensions_map {
                    if let Some(handler) = self.extensions.get_mut(name) {
                        handler.on_load_material(&mut ctx, &material, &mat, value)?;
                    }
                }
            }

            mat.flush_texture_transforms();
            mat.notify_pipeline_dirty();

            let mut engine_mat = Material::from(mat);
            engine_mat.name = material.name().map(|s| Cow::Owned(s.to_string()));

            let handle = self.assets.materials.add(engine_mat);
            self.material_map.push(handle);
        }
        Ok(())
    }

    fn build_prefab(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<Prefab> {
        for node in gltf.nodes() {
            let prefab_node = self.create_prefab_node(&node)?;
            self.prefab_nodes.push(prefab_node);
        }

        for node in gltf.nodes() {
            let parent_idx = node.index();
            for child in node.children() {
                self.prefab_nodes[parent_idx].children_indices.push(child.index());
            }
        }

        self.load_skins(gltf, buffers)?;

        for node in gltf.nodes() {
            self.bind_node_mesh_and_skin(&node, buffers)?;
        }

        let root_indices: Vec<usize> = if let Some(default_scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            default_scene.nodes().map(|n| n.index()).collect()
        } else {
            Vec::new()
        };

        let animations = self.load_animations(gltf, buffers)?;

        Ok(Prefab {
            nodes: std::mem::take(&mut self.prefab_nodes),
            root_indices,
            skeletons: std::mem::take(&mut self.prefab_skeletons),
            animations,
        })
    }

    fn create_prefab_node(&self, node: &gltf::Node) -> anyhow::Result<PrefabNode> {
        let node_name = node.name().map(|s| s.to_string());
        
        let mut prefab_node = PrefabNode::new();
        prefab_node.name = node_name;

        let (t, r, s) = node.transform().decomposed();
        prefab_node.transform.position = Vec3::from_array(t);
        prefab_node.transform.rotation = Quat::from_array(r);
        prefab_node.transform.scale = Vec3::from_array(s);

        Ok(prefab_node)
    }

    fn load_skins(&mut self, gltf: &gltf::Gltf, buffers: &[Vec<u8>]) -> anyhow::Result<()> {
        for skin in gltf.skins() {
            let name = skin.name().unwrap_or("Skeleton").to_string();
            
            let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
            let ibms: Vec<Affine3A> = if let Some(iter) = reader.read_inverse_bind_matrices() {
                iter.map(|m| {
                    let mat = Mat4::from_cols_array_2d(&m);
                    Affine3A::from_mat4(mat)
                }).collect()
            } else {
                vec![Affine3A::IDENTITY; skin.joints().count()]
            };

            let bone_indices: Vec<usize> = skin.joints().map(|node| node.index()).collect();

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

            self.prefab_skeletons.push(PrefabSkeleton {
                name,
                root_bone_index,
                bone_indices,
                inverse_bind_matrices: ibms,
            });
        }

        Ok(())
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
        
        if let Some(geometry) = self.assets.geometries.get(geo_handle) {
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
    ) -> anyhow::Result<()> {
        let node_idx = node.index();

        if let Some(mesh) = node.mesh() {
            let primitives: Vec<_> = mesh.primitives().collect();

            match primitives.len() {
                0 => {},
                1 => {
                    let engine_mesh = self.build_engine_mesh(&primitives[0], buffers)?;
                    self.prefab_nodes[node_idx].mesh = Some(engine_mesh);
                }
                _ => {
                    let base_idx = self.prefab_nodes.len();
                    let parent_name = self.prefab_nodes[node_idx].name.clone();
                    
                    for (i, primitive) in primitives.iter().enumerate() {
                        let engine_mesh = self.build_engine_mesh(primitive, buffers)?;
                        
                        let mut sub_node = PrefabNode::new();
                        sub_node.name = Some(format!("{}_{}", 
                            parent_name.as_deref().unwrap_or("node"), i));
                        sub_node.mesh = Some(engine_mesh);
                        
                        self.prefab_nodes.push(sub_node);
                    }
                    
                    for i in 0..primitives.len() {
                        self.prefab_nodes[node_idx].children_indices.push(base_idx + i);
                    }
                }
            }
        }

        if let Some(skin) = node.skin() {
            self.prefab_nodes[node_idx].skin_index = Some(skin.index());
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
            return Ok(self.assets.geometries.add(geometry));
        }

        geometry.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));

        if let Some(iter) = reader.read_indices() {
            let indices: Vec<u32> = iter.into_u32().collect();
            geometry.set_indices_u32(&indices);
        }

        let mut surface_channels = Vec::new();

        if let Some(iter) = reader.read_normals() {
            surface_channels.push(InterleaveChannel::from_iter("normal", iter, VertexFormat::Float32x3));
        } else {
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
            gltf::mesh::Mode::LineLoop => PrimitiveTopology::LineList,
            gltf::mesh::Mode::LineStrip => PrimitiveTopology::LineStrip,
            gltf::mesh::Mode::Triangles => PrimitiveTopology::TriangleList,
            gltf::mesh::Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
            gltf::mesh::Mode::TriangleFan => PrimitiveTopology::TriangleList,
        };
        geometry.build_morph_storage_buffers();
        geometry.compute_bounding_volume();

        Ok(self.assets.geometries.add(geometry))
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
                let Some(gltf_node) = target.node() else { 
                    log::warn!("Animation target node is missing(Maybe use\"KHR_animation_pointer\"),  skipping channel for now.");
                    continue; };
                
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

    fn on_load_material(&mut self, ctx: &mut LoadContext, gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, _extension_value: &Value) -> anyhow::Result<()> {
        let sg = gltf_mat.pbr_specular_glossiness()
            .ok_or_else(|| anyhow::anyhow!("Material missing pbr_specular_glossiness data"))?;

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
            physical_mat.textures.write().map.texture = Some(tex_handle);
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
            
            let specular_handle = ctx.assets.textures.add(specular_texture);
            let roughness_handle = ctx.assets.textures.add(roughness_texture);

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

            let mut textures_set = physical_mat.textures.write();
            
            textures_set.specular_map.texture = Some(specular_handle);
            textures_set.specular_map.channel = uv_channel as u8;
            textures_set.specular_map.transform = transform.clone();
            textures_set.roughness_map.texture = Some(roughness_handle);
            textures_set.roughness_map.channel = uv_channel as u8;
            textures_set.roughness_map.transform = transform.clone();

            textures_set.metalness_map.texture = Some(roughness_handle);
            textures_set.metalness_map.channel = uv_channel as u8;
            textures_set.metalness_map.transform = transform;
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

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let clearcoat_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid clearcoat extension data"))?;

        let clearcoat_factor = clearcoat_info.get("clearcoatFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let clearcoat_roughness = clearcoat_info.get("clearcoatRoughnessFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.clearcoat = clearcoat_factor;
            uniforms.clearcoat_roughness = clearcoat_roughness;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(clearcoat_tex_info) = clearcoat_info.get("clearcoatTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_tex_info, &mut textures.clearcoat_map, false);
        }

        if let Some(clearcoat_roughness_tex_info) = clearcoat_info.get("clearcoatRoughnessTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_roughness_tex_info, &mut textures.clearcoat_roughness_map, false);
        }

        if let Some(clearcoat_normal_tex_info) = clearcoat_info.get("clearcoatNormalTexture") {
            self.setup_texture_map_from_extension(ctx, clearcoat_normal_tex_info, &mut textures.clearcoat_normal_map, false);
        }

        physical_mat.enable_feature(PhysicalFeatures::CLEARCOAT);
        
        Ok(())
    }
}

struct KhrMaterialsSheen;

impl GltfExtensionParser for KhrMaterialsSheen {
    fn name(&self) -> &str {
        "KHR_materials_sheen"
    }

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let sheen_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid sheen extension data"))?;

        let sheen_color_factor = sheen_info.get("sheenColorFactor")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some(Vec3::new(
                        arr[0].as_f64().unwrap_or(0.0) as f32,
                        arr[1].as_f64().unwrap_or(0.0) as f32,
                        arr[2].as_f64().unwrap_or(0.0) as f32,
                    ))
                } else {
                    None
                }
            })
            .unwrap_or(Vec3::ZERO);

        let sheen_roughness_factor = sheen_info.get("sheenRoughnessFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.sheen_color = sheen_color_factor;
            uniforms.sheen_roughness = sheen_roughness_factor;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(sheen_color_tex_info) = sheen_info.get("sheenColorTexture") {
            self.setup_texture_map_from_extension(ctx, sheen_color_tex_info, &mut textures.sheen_color_map, true);
        }

        if let Some(sheen_roughness_tex_info) = sheen_info.get("sheenRoughnessTexture") {
            self.setup_texture_map_from_extension(ctx, sheen_roughness_tex_info, &mut textures.sheen_roughness_map, false);
        }

        physical_mat.enable_feature(PhysicalFeatures::SHEEN);

        Ok(())
    }
}

struct KhrMaterialsIridescence;

impl GltfExtensionParser for KhrMaterialsIridescence {
    fn name(&self) -> &str {
        "KHR_materials_iridescence"
    }

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let iridescence_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid iridescence extension data"))?;

        let iridescence_factor = iridescence_info.get("iridescenceFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let iridescence_ior = iridescence_info.get("iridescenceIor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.3) as f32;

        let iridescence_thickness_min = iridescence_info.get("iridescenceThicknessMin")
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0) as f32;

        let iridescence_thickness_max = iridescence_info.get("iridescenceThicknessMax")
            .and_then(|v| v.as_f64())
            .unwrap_or(400.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.iridescence = iridescence_factor;
            uniforms.iridescence_ior = iridescence_ior;
            uniforms.iridescence_thickness_min = iridescence_thickness_min;
            uniforms.iridescence_thickness_max = iridescence_thickness_max;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(iridescence_tex_info) = iridescence_info.get("iridescenceTexture") {
            self.setup_texture_map_from_extension(ctx, iridescence_tex_info, &mut textures.iridescence_map, false);
        }

        if let Some(iridescence_thickness_tex_info) = iridescence_info.get("iridescenceThicknessTexture") {
            self.setup_texture_map_from_extension(ctx, iridescence_thickness_tex_info, &mut textures.iridescence_thickness_map, false);
        }

        physical_mat.enable_feature(PhysicalFeatures::IRIDESCENCE);

        Ok(())
    }
}

struct KhrMaterialsAnisotropy;

impl GltfExtensionParser for KhrMaterialsAnisotropy {
    fn name(&self) -> &str {
        "KHR_materials_anisotropy"
    }

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let anisotropy_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid anisotropy extension data"))?;

        let anisotropy_strength = anisotropy_info.get("anisotropyStrength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let anisotropy_rotation = anisotropy_info.get("anisotropyRotation")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            let direction = Vec2::new(anisotropy_rotation.cos(), anisotropy_rotation.sin()) * anisotropy_strength;
            uniforms.anisotropy_vector = direction;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(anisotropy_tex_info) = anisotropy_info.get("anisotropyTexture") {
            self.setup_texture_map_from_extension(ctx, anisotropy_tex_info, &mut textures.anisotropy_map, false);
        }

        physical_mat.enable_feature(PhysicalFeatures::ANISOTROPY);

        Ok(())
    }
}

struct KhrMaterialsTransmission;

impl GltfExtensionParser for KhrMaterialsTransmission {
    fn name(&self) -> &str {
        "KHR_materials_transmission"
    }

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let transmission_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid transmission extension data"))?;

        let transmission_factor = transmission_info.get("transmissionFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.transmission = transmission_factor;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(transmission_tex_info) = transmission_info.get("transmissionTexture") {
            self.setup_texture_map_from_extension(ctx, transmission_tex_info, &mut textures.transmission_map, false);
        }
        physical_mat.enable_feature(PhysicalFeatures::TRANSMISSION);
        Ok(())
    }
}

struct KhrMaterialsVolume;

impl GltfExtensionParser for KhrMaterialsVolume {
    fn name(&self) -> &str {
        "KHR_materials_volume"
    }

    fn on_load_material(&mut self, ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let volume_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid volume extension data"))?;

        let thickness_factor = volume_info.get("thicknessFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let attenuation_color = volume_info.get("attenuationColor")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some(Vec3::new(
                        arr[0].as_f64().unwrap_or(1.0) as f32,
                        arr[1].as_f64().unwrap_or(1.0) as f32,
                        arr[2].as_f64().unwrap_or(1.0) as f32,
                    ))
                } else {
                    None
                }
            })
            .unwrap_or(Vec3::ONE);

        let attenuation_distance = volume_info.get("attenuationDistance")
            .and_then(|v| v.as_f64())
            .unwrap_or(-1.0 as f64) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.thickness = thickness_factor;
            uniforms.attenuation_color = attenuation_color;
            uniforms.attenuation_distance = attenuation_distance;
        }

        let mut textures = physical_mat.textures.write();

        if let Some(thickness_tex_info) = volume_info.get("thicknessTexture") {
            self.setup_texture_map_from_extension(ctx, thickness_tex_info, &mut textures.thickness_map, false);
        }

        Ok(())
    }   
    
}

struct KhrMaterialsDispersion;

impl GltfExtensionParser for KhrMaterialsDispersion {
    fn name(&self) -> &str {
        "KHR_materials_dispersion"
    }

    fn on_load_material(&mut self, _ctx: &mut LoadContext, _gltf_mat: &gltf::Material, physical_mat: &MeshPhysicalMaterial, extension_value: &Value) -> anyhow::Result<()> {
        let dispersion_info = extension_value.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid dispersion extension data"))?;

        let dispersion_factor = dispersion_info.get("dispersionFactor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        {
            let mut uniforms = physical_mat.uniforms_mut();
            uniforms.dispersion = dispersion_factor;
        }

        physical_mat.enable_feature(PhysicalFeatures::DISPERSION);

        Ok(())
    }
}