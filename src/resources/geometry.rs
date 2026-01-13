use std::collections::HashMap;
use uuid::Uuid;
use wgpu::{PrimitiveTopology, VertexFormat, VertexStepMode, BufferUsages};
use glam::Vec3;
use core::ops::Range;
use bitflags::bitflags;

use crate::resources::buffer::{BufferRef};

#[derive(Debug, Clone)]
pub struct Attribute {
    pub buffer: BufferRef,
    pub format: VertexFormat,
    pub offset: u64,
    pub count: u32,
    pub stride: u64,
    pub step_mode: VertexStepMode,
}

impl Attribute {
    pub fn new_planar<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let stride = std::mem::size_of::<T>() as u64;
        let buffer_ref = BufferRef::new(
            data, 
            BufferUsages::VERTEX | BufferUsages::COPY_DST, 
            Some("GeometryVertexAttr")
        );

        Self {
            buffer: buffer_ref,
            format,
            offset: 0,
            count: data.len() as u32,
            stride,
            step_mode: VertexStepMode::Vertex,
        }
    }

    pub fn new_instanced<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let stride = std::mem::size_of::<T>() as u64;
        let buffer_ref = BufferRef::new(
            data, 
            BufferUsages::VERTEX | BufferUsages::COPY_DST, 
            Some("GeometryInstanceAttr")
        );

        Self {
            buffer: buffer_ref,
            format,
            offset: 0,
            count: data.len() as u32,
            stride,
            step_mode: VertexStepMode::Instance,
        }
    }

    pub fn new_interleaved(
        buffer: BufferRef, 
        format: VertexFormat, 
        offset: u64, 
        count: u32,
        stride: u64,
        step_mode: VertexStepMode
    ) -> Self {
        Self { buffer, format, offset, count, stride, step_mode }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    pub fn size(&self) -> Vec3 { self.max - self.min }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct GeometryFeatures: u32 {
        const HAS_NORMAL       = 1 << 0;
        const HAS_UV           = 1 << 1;
        const USE_VERTEX_COLOR  = 1 << 2;
        const USE_TANGENT       = 1 << 3;
        const USE_MORPHING      = 1 << 4; // 变形
        const USE_SKINNING      = 1 << 5; // 骨骼
    }
}

#[derive(Debug)]
pub struct Geometry {
    pub uuid: Uuid,
    
    pub layout_version: u64,
    pub structure_version: u64,

    pub attributes: HashMap<String, Attribute>,
    pub index_attribute: Option<Attribute>,
    
    pub morph_attributes: HashMap<String, Vec<Attribute>>,
    pub morph_target_names: Vec<String>,

    pub topology: PrimitiveTopology,
    pub draw_range: Range<u32>,

    pub bounding_box: Option<BoundingBox>,
    pub bounding_sphere: Option<BoundingSphere>,
}

impl Geometry {
    pub fn new() -> Self {
        Self {
            uuid: Uuid::new_v4(),
            layout_version: 0,
            structure_version: 0,
            attributes: HashMap::new(),
            index_attribute: None,
            morph_attributes: HashMap::new(),
            morph_target_names: Vec::new(),
            topology: PrimitiveTopology::TriangleList,
            draw_range: 0..u32::MAX,
            bounding_box: None,
            bounding_sphere: None,
        }
    }

    pub fn layout_version(&self) -> u64 {
        self.layout_version
    }

    pub fn structure_version(&self) -> u64 {
        self.structure_version
    }

    pub fn set_attribute(&mut self, name: &str, attr: Attribute) {
        let layout_changed = if let Some(old) = self.attributes.get(name) {
            old.format != attr.format || old.step_mode != attr.step_mode
        } else {
            true
        };

        self.attributes.insert(name.to_string(), attr);
        
        if layout_changed {
            self.layout_version = self.layout_version.wrapping_add(1);
        }
        self.structure_version = self.structure_version.wrapping_add(1);
    }

    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.get(name)
    }

    pub fn set_indices(&mut self, indices: &[u16]) {
        let buffer_ref = BufferRef::new(
            indices, 
            BufferUsages::INDEX | BufferUsages::COPY_DST, 
            Some("IndexBuffer")
        );

        self.index_attribute = Some(Attribute {
            buffer: buffer_ref,
            format: VertexFormat::Uint16,
            offset: 0,
            count: indices.len() as u32,
            stride: 2,
            step_mode: VertexStepMode::Vertex,
        });
        self.structure_version = self.structure_version.wrapping_add(1);
    }

    pub fn set_indices_u32(&mut self, indices: &[u32]) {
        let buffer_ref = BufferRef::new(
            indices, 
            BufferUsages::INDEX | BufferUsages::COPY_DST, 
            Some("IndexBuffer")
        );

        self.index_attribute = Some(Attribute {
            buffer: buffer_ref,
            format: VertexFormat::Uint32,
            offset: 0,
            count: indices.len() as u32,
            stride: 4,
            step_mode: VertexStepMode::Vertex,
        });
        self.structure_version = self.structure_version.wrapping_add(1);
    }

    pub fn compute_bounding_volume(&mut self) {
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };

        let data = pos_attr.buffer.peek_data(|d| d.to_vec()).unwrap_or_default();
        let stride = pos_attr.stride as usize;
        let offset = pos_attr.offset as usize;
        let count = pos_attr.count as usize;

        if pos_attr.format != VertexFormat::Float32x3 {
            return;
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut sum_pos = Vec3::ZERO;
        let mut valid_points_count = 0;

        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;
            if end > data.len() { break; }

            let bytes: &[u8; 12] = data[start..end].try_into().unwrap();
            let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
            let vec = Vec3::from_array(*vals);

            min = min.min(vec);
            max = max.max(vec);
            
            sum_pos += vec;
            valid_points_count += 1;
        }

        if valid_points_count == 0 { return; }

        self.bounding_box = Some(BoundingBox { min, max });

        let centroid = sum_pos / (valid_points_count as f32);
        
        let mut max_dist_sq = 0.0;

        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;
            if end > data.len() { break; }
            let bytes: &[u8; 12] = data[start..end].try_into().unwrap();
            let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
            let vec = Vec3::from_array(*vals);

            let dist_sq = vec.distance_squared(centroid);
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
            }
        }

        self.bounding_sphere = Some(BoundingSphere {
            center: centroid,
            radius: max_dist_sq.sqrt(),
        });
    }

    pub fn get_features(&self) -> GeometryFeatures {
        let mut features = GeometryFeatures::empty();

        if self.attributes.contains_key("uv") {
            features |= GeometryFeatures::HAS_UV;
        }
        if self.attributes.contains_key("normal") {
            features |= GeometryFeatures::HAS_NORMAL;
        }
        if self.attributes.contains_key("color") {
            features |= GeometryFeatures::USE_VERTEX_COLOR;
        }
        if self.attributes.contains_key("tangent") {
            features |= GeometryFeatures::USE_TANGENT;
        }
        if !self.morph_attributes.is_empty() {
            features |= GeometryFeatures::USE_MORPHING;
        }

        features
    }

    pub fn new_box(width: f32, height: f32, depth: f32) -> Self {
        let w = width / 2.0;
        let h = height / 2.0;
        let d = depth / 2.0;

        // 24 个顶点 (每个面 4 个)
        // 格式: [x, y, z]
        let positions = [
            // Front face (+Z)
            [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
            // Back face (-Z)
            [-w, -h, -d], [-w,  h, -d], [ w,  h, -d], [ w, -h, -d],
            // Top face (+Y)
            [-w,  h, -d], [-w,  h,  d], [ w,  h,  d], [ w,  h, -d],
            // Bottom face (-Y)
            [-w, -h, -d], [ w, -h, -d], [ w, -h,  d], [-w, -h,  d],
            // Right face (+X)
            [ w, -h, -d], [ w,  h, -d], [ w,  h,  d], [ w, -h,  d],
            // Left face (-X)
            [-w, -h, -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d],
        ];

        // 法线 (每个面的 4 个顶点法线相同)
        let normals: [[f32; 3]; 24] = [
            // Front
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            // Back
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            // Top
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            // Bottom
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            // Right
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            // Left
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        ];

        // UV 坐标 (标准 0~1)
        let uvs: [[f32; 2]; 24] = [
            // Front
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            // Back
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0],
            // Top
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
            // Bottom
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            // Right
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            // Left
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
        ];

        // 索引 (每面 2 个三角形，逆时针绕序 CCW)
        // 0, 1, 2,  0, 2, 3
        let indices: Vec<u16> = (0..6).flat_map(|face| {
            let base = face * 4;
            [
                base, base + 1, base + 2,
                base, base + 2, base + 3,
            ]
        }).collect();

        let mut geo = Geometry::new();
        geo.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));
        geo.set_attribute("normal", Attribute::new_planar(&normals, VertexFormat::Float32x3));
        geo.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
        geo.set_indices(&indices);
        
        geo.compute_bounding_volume();

        geo
    }
}
