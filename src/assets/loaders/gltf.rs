use std::path::Path;
use std::fs;
use glam::{Vec3, Vec4, Quat};
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::material::{Material, MeshStandardMaterial};
use crate::resources::texture::Texture;
use crate::scene::{Scene, Node};
use crate::assets::{AssetServer, TextureHandle, MaterialHandle, GeometryHandle};
use wgpu::{VertexFormat, TextureFormat};
use anyhow::Context;

pub struct GltfLoader<'a> {
    assets: &'a mut AssetServer,
    scene: &'a mut Scene,
    base_path: std::path::PathBuf,
    texture_map: Vec<TextureHandle>,
    material_map: Vec<MaterialHandle>,
}

impl<'a> GltfLoader<'a> {
    pub fn load(
        path: &Path,
        assets: &'a mut AssetServer,
        scene: &'a mut Scene
    ) -> anyhow::Result<Vec<crate::scene::NodeIndex>> {
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open glTF file: {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader(reader)
            .context("Failed to parse glTF file")?;

        let base_path = path.parent().unwrap_or(Path::new("./")).to_path_buf();

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

        let mut loader = Self {
            assets,
            scene,
            base_path,
            texture_map: Vec::new(),
            material_map: Vec::new(),
        };

        loader.load_textures(&gltf, &buffer_data)?;
        loader.load_materials(&gltf)?;

        let mut root_indices = Vec::new();
        let default_scene = gltf.default_scene().or_else(|| gltf.scenes().next());
        
        if let Some(scene) = default_scene {
            for node in scene.nodes() {
                let node_idx = loader.load_node(&node, &buffer_data, None)?;
                root_indices.push(node_idx);
            }
        }

        Ok(root_indices)
    }

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
            
            let engine_tex = Texture::new_2d(
                Some(tex_name),
                width,
                height,
                Some(img_data.into_vec()),
                TextureFormat::Rgba8Unorm
            );

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

            // Convert to generic Material wrapper
            let handle = self.assets.add_material(Material::from(mat));
            self.material_map.push(handle);
        }
        Ok(())
    }

    fn load_node(
        &mut self,
        node: &gltf::Node,
        buffers: &[Vec<u8>],
        parent_idx: Option<crate::scene::NodeIndex>
    ) -> anyhow::Result<crate::scene::NodeIndex> {

        let mut engine_node = Node::new(node.name().unwrap_or("Node"));

        let (t, r, s) = node.transform().decomposed();
        engine_node.transform.position = Vec3::from_array(t);
        engine_node.transform.rotation = Quat::from_array(r);
        engine_node.transform.scale = Vec3::from_array(s);

        if let Some(mesh) = node.mesh() {
            for (i, primitive) in mesh.primitives().enumerate() {
                let geo_handle = self.load_primitive_geometry(&primitive, buffers)?;
                let mat_idx = primitive.material().index();
                let mat_handle = if let Some(idx) = mat_idx {
                    self.material_map[idx]
                } else {
                    self.assets.add_material(Material::new_standard(Vec4::ONE))
                };

                let engine_mesh = crate::resources::mesh::Mesh::new(geo_handle, mat_handle);
                let mesh_key = self.scene.meshes.insert(engine_mesh);

                if i == 0 {
                    engine_node.mesh = Some(mesh_key);
                }
            }
        }

        let node_idx = if let Some(p) = parent_idx {
            self.scene.add_to_parent(engine_node, p)
        } else {
            self.scene.add_node(engine_node)
        };

        for child in node.children() {
            self.load_node(&child, buffers, Some(node_idx))?;
        }

        Ok(node_idx)
    }

    fn load_primitive_geometry(
        &mut self,
        primitive: &gltf::Primitive,
        buffers: &[Vec<u8>]
    ) -> anyhow::Result<GeometryHandle> {
        let mut geometry = Geometry::new();
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        if let Some(iter) = reader.read_positions() {
            let positions: Vec<[f32; 3]> = iter.collect();
            geometry.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));
        }

        if let Some(iter) = reader.read_normals() {
            let normals: Vec<[f32; 3]> = iter.collect();
            geometry.set_attribute("normal", Attribute::new_planar(&normals, VertexFormat::Float32x3));
        }

        if let Some(iter) = reader.read_tex_coords(0) {
            let uvs: Vec<[f32; 2]> = iter.into_f32().collect();
            geometry.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
        }

        if let Some(iter) = reader.read_tangents() {
            let tangents: Vec<[f32; 4]> = iter.collect();
            geometry.set_attribute("tangent", Attribute::new_planar(&tangents, VertexFormat::Float32x4));
        }

        if let Some(iter) = reader.read_indices() {
            let indices: Vec<u32> = iter.into_u32().collect();
            geometry.set_indices_u32(&indices);
        }

        geometry.compute_bounding_volume();

        Ok(self.assets.add_geometry(geometry))
    }
}
