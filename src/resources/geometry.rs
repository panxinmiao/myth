use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;
use wgpu::{PrimitiveTopology, VertexFormat, VertexStepMode, BufferUsages};
use glam::Vec3;
use core::ops::Range;
use bitflags::bitflags;

use crate::resources::buffer::{BufferRef};
use crate::resources::primitives;

/// Attribute holds CPU-side data (Option<Arc<Vec<u8>>>) and metadata.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub buffer: BufferRef,
    
    /// CPU-side data shared via Arc (supports interleaved buffers)
    pub data: Option<Arc<Vec<u8>>>,

    /// Data version for change detection
    pub version: u64,
    
    pub format: VertexFormat,
    pub offset: u64,
    pub count: u32,
    pub stride: u64,
    pub step_mode: VertexStepMode,
}

static NEXT_ATTR_VERSION: AtomicU64 = AtomicU64::new(1);

impl Attribute {
    /// 创建 Planar (非交错) 属性
    pub fn new_planar<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        let size = raw_data.len();
        
        // 创建句柄
        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::VERTEX | BufferUsages::COPY_DST, 
            Some("GeometryVertexAttr")
        );

        Self {
            buffer: buffer_ref,
            data: Some(Arc::new(raw_data)),
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
            format,
            offset: 0,
            count: data.len() as u32,
            stride: std::mem::size_of::<T>() as u64,
            step_mode: VertexStepMode::Vertex,
        }
    }

    /// 创建 Instance 属性
    pub fn new_instanced<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        let size = raw_data.len();
        
        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::VERTEX | BufferUsages::COPY_DST, 
            Some("GeometryInstanceAttr")
        );

        Self {
            buffer: buffer_ref,
            data: Some(Arc::new(raw_data)),
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
            format,
            offset: 0,
            count: data.len() as u32,
            stride: std::mem::size_of::<T>() as u64,
            step_mode: VertexStepMode::Instance,
        }
    }

    /// 创建 Interleaved (交错) 属性
    /// 多个 Attribute 可以共享同一个 BufferRef 和 data (Arc)
    pub fn new_interleaved(
        buffer: BufferRef, 
        data: Option<Arc<Vec<u8>>>,
        format: VertexFormat, 
        offset: u64, 
        count: u32,
        stride: u64,
        step_mode: VertexStepMode
    ) -> Self {
        Self { 
            buffer, 
            data,
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
            format, 
            offset, 
            count, 
            stride, 
            step_mode 
        }
    }

    /// 原地更新数据 (保留 ID，复用显存)
    /// 使用 Arc::make_mut 实现 Copy-On-Write
    pub fn update_data<T: bytemuck::Pod>(&mut self, new_data: &[T]) {
        if let Some(arc_vec) = &mut self.data {
            // Arc::make_mut：如果只有一个引用，直接修改；否则克隆后修改
            let vec = Arc::make_mut(arc_vec);
            
            let bytes: &[u8] = bytemuck::cast_slice(new_data);
            
            // 如果长度变了，需要调整 Vec
            if vec.len() != bytes.len() {
                vec.resize(bytes.len(), 0);
            }
            vec.copy_from_slice(bytes);
            
            // 更新元数据
            self.count = new_data.len() as u32;
            self.version = NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 局部更新属性数据
    pub fn update_region<T: bytemuck::Pod>(
        &mut self, 
        offset_bytes: u64, 
        new_data: &[T]
    ) {
        if let Some(arc_vec) = &mut self.data {
            let vec = Arc::make_mut(arc_vec);
            let bytes = bytemuck::cast_slice(new_data);
            
            let start = offset_bytes as usize;
            let end = start + bytes.len();
            
            // 边界检查
            if end <= vec.len() {
                vec[start..end].copy_from_slice(bytes);
                self.version = NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed);
            }
        }
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

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
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
        let raw_data = bytemuck::cast_slice(indices).to_vec();
        let size = raw_data.len();
        
        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::INDEX | BufferUsages::COPY_DST, 
            Some("IndexBuffer")
        );

        self.index_attribute = Some(Attribute {
            buffer: buffer_ref,
            data: Some(Arc::new(raw_data)),
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
            format: VertexFormat::Uint16,
            offset: 0,
            count: indices.len() as u32,
            stride: 2,
            step_mode: VertexStepMode::Vertex,
        });
        self.structure_version = self.structure_version.wrapping_add(1);
    }

    pub fn set_indices_u32(&mut self, indices: &[u32]) {
        let raw_data = bytemuck::cast_slice(indices).to_vec();
        let size = raw_data.len();
        
        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::INDEX | BufferUsages::COPY_DST, 
            Some("IndexBuffer")
        );

        self.index_attribute = Some(Attribute {
            buffer: buffer_ref,
            data: Some(Arc::new(raw_data)),
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
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

        // 从 Attribute 的 data 中获取数据
        let data = match &pos_attr.data {
            Some(arc_data) => arc_data.as_ref().clone(),
            None => return,
        };
        
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

            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _>{
                    let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                    let vec = Vec3::from_array(*vals);

                    min = min.min(vec);
                    max = max.max(vec);
                    sum_pos += vec;
                    valid_points_count += 1;
                }
            } else {
                break; 
            }
        }

        if valid_points_count == 0 { return; }

        self.bounding_box = Some(BoundingBox { min, max });

        let centroid = sum_pos / (valid_points_count as f32);
        
        let mut max_dist_sq = 0.0;

        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;
            
            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _>{
                    let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                    let vec = Vec3::from_array(*vals);
                    let dist_sq = vec.distance_squared(centroid);
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                    }
                }
            }else {
                break; 
            }
            
        }

        self.bounding_sphere = Some(BoundingSphere {
            center: centroid,
            radius: max_dist_sq.sqrt(),
        });
    }

    /// 设置交错属性 (Interleaved Attributes)
    /// 从一个交错数组中创建多个共享同一个 Buffer 的 Attribute
    pub fn set_interleaved_attributes(
        &mut self,
        interleaved_data: Vec<u8>, // 原始交错数据
        stride: u64,
        attributes: Vec<(&str, VertexFormat, u64)> // (名字, 格式, 偏移量)
    ) {
        let shared_data = Arc::new(interleaved_data);
        let count = (shared_data.len() as u64 / stride) as u32;

        let shared_buffer_ref = BufferRef::new(
            shared_data.len(),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            Some("InterleavedBuffer")
        );

        // Create attributes sharing the same buffer and data
        for (name, format, offset) in attributes {
            let attr = Attribute {
                buffer: shared_buffer_ref.clone(), 
                data: Some(shared_data.clone()), 

                version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
                format,
                offset,
                stride,
                count,
                step_mode: VertexStepMode::Vertex,
            };

            self.set_attribute(name, attr);
        }
    }

    /// 局部更新属性数据
    pub fn update_attribute_region<T: bytemuck::Pod>(
        &mut self, 
        name: &str, 
        offset_bytes: u64, 
        data: &[T]
    ) {
        if let Some(attr) = self.attributes.get_mut(name) {
            attr.update_region(offset_bytes, data);
        }
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
        primitives::create_box(width, height, depth)
    }

    pub fn new_sphere(radius: f32) -> Self {
        primitives::create_sphere(primitives::SphereOptions {
            radius,
            ..Default::default()
        })
    }

    pub fn new_plane(width: f32, height: f32) -> Self {
        primitives::create_plane(primitives::PlaneOptions {
            width,
            height,
            ..Default::default()
        })
    }
}
