use std::cell::RefCell;
use std::sync::Arc;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;
use wgpu::{PrimitiveTopology, VertexFormat, VertexStepMode, BufferUsages};
use glam::{Affine3A, Vec3, Vec4};
use core::ops::Range;

use crate::resources::buffer::BufferRef;
use crate::resources::primitives;
use crate::resources::shader_defines::ShaderDefines;

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

    pub fn read_vec3(&self, i: u32) -> Option<Vec3> {
        if self.format != VertexFormat::Float32x3 {
            return None;
        }
        let stride = self.stride as usize;
        let offset = self.offset as usize + (i as usize) * stride;
        
        if let Some(data) = &self.data {
            let slice = data.as_ref();
            if offset + 12 <= slice.len() {
                let bytes: &[u8; 12] = slice[offset..offset + 12].try_into().ok()?;
                let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                return Some(Vec3::from_array(*vals));
            }
        }
        None
    }

    pub fn read_vec4(&self, i: u32) -> Option<Vec4> {
        if self.format != VertexFormat::Float32x4 {
            return None;
        }
        let stride = self.stride as usize;
        let offset = self.offset as usize + (i as usize) * stride;
        
        if let Some(data) = &self.data {
            let slice = data.as_ref();
            if offset + 16 <= slice.len() {
                let bytes: &[u8; 16] = slice[offset..offset + 16].try_into().ok()?;
                let vals: &[f32; 4] = bytemuck::cast_ref(bytes);
                return Some(Vec4::from_array(*vals));
            }
        }
        None
    }

    pub fn read<T>(&self, i: u32) -> Option<T> where T: bytemuck::Pod {
        let stride = self.stride as usize;
        let offset = self.offset as usize + (i as usize) * stride;
        let size = std::mem::size_of::<T>();
        
        if let Some(data) = &self.data {
            let slice = data.as_ref();
            if offset + size <= slice.len() {
                let bytes: &[u8] = &slice[offset..offset + size];
                let val: &T = bytemuck::from_bytes(bytes);
                return Some(*val);
            }
        }
        None
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
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn transform(&self, matrix: &Affine3A) -> Self {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];

        let mut new_min = Vec3::splat(f32::INFINITY);
        let mut new_max = Vec3::splat(f32::NEG_INFINITY);

        for point in corners {
            // 假设 Affine3A 可以直接 transform_point3
            let transformed = matrix.transform_point3(point);
            new_min = new_min.min(transformed);
            new_max = new_max.max(transformed);
        }

        Self { min: new_min, max: new_max }
    }
    
    // 简单的膨胀方法
    pub fn inflate(&self, amount: f32) -> Self {
        Self {
            min: self.min * Vec3::splat(1.0 - amount),
            max: self.max * Vec3::splat(1.0 + amount),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Debug)]
pub struct Geometry {
    pub uuid: Uuid,
    
    // vertex lay out versioning
    layout_version: u64,
    // vertex buffers versioning
    structure_version: u64,
    data_version: u64,

    attributes: FxHashMap<String, Attribute>,
    index_attribute: Option<Attribute>,

    pub morph_attributes: FxHashMap<String, Vec<Attribute>>,

    pub morph_target_names: Vec<String>,

    /// Morph Target Storage Buffers (紧凑 f32 存储)
    /// 布局: [ Target 0 所有顶点 | Target 1 所有顶点 | ... ]
    /// 每个顶点存储 3 个 f32 (Position/Normal/Tangent displacement)
    pub morph_position_buffer: Option<BufferRef>,
    pub morph_normal_buffer: Option<BufferRef>,
    pub morph_tangent_buffer: Option<BufferRef>,
    
    /// Morph Target 数据 (CPU 端保持以支持上传)
    morph_position_data: Option<Vec<f32>>,
    morph_normal_data: Option<Vec<f32>>,
    morph_tangent_data: Option<Vec<f32>>,
    
    /// 每个 Target 的顶点数
    pub morph_vertex_count: u32,
    /// Morph Target 数量
    pub morph_target_count: u32,

    pub topology: PrimitiveTopology,
    pub draw_range: Range<u32>,

    pub bounding_box: RefCell<Option<BoundingBox>>,
    pub bounding_sphere: RefCell<Option<BoundingSphere>>,

    /// ShaderDefines 缓存：(layout_version, cached_defines)
    cached_shader_defines: RefCell<Option<(u64, ShaderDefines)>>,
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
            data_version: 0,
            attributes: FxHashMap::default(),
            index_attribute: None,
            morph_attributes: FxHashMap::default(),
            morph_target_names: Vec::new(),
            morph_position_buffer: None,
            morph_normal_buffer: None,
            morph_tangent_buffer: None,
            morph_position_data: None,
            morph_normal_data: None,
            morph_tangent_data: None,
            morph_vertex_count: 0,
            morph_target_count: 0,
            topology: PrimitiveTopology::TriangleList,
            draw_range: 0..u32::MAX,
            bounding_box: RefCell::new(None),
            bounding_sphere: RefCell::new(None),
            cached_shader_defines: RefCell::new(None),
        }
    }

    // Version accessors
    pub fn layout_version(&self) -> u64 {
        self.layout_version
    }

    pub fn structure_version(&self) -> u64 {
        self.structure_version
    }

    pub fn data_version(&self) -> u64 {
        self.data_version
    }

    // Attributes accessors
    pub fn attributes(&self) -> &FxHashMap<String, Attribute> {
        &self.attributes
    }
    
    // Index attribute accessors
    pub fn index_attribute(&self) -> Option<&Attribute> {
        self.index_attribute.as_ref()
    }
    
    pub fn index_attribute_mut(&mut self) -> &mut Option<Attribute> {
        self.structure_version = self.structure_version.wrapping_add(1);
        self.data_version = self.data_version.wrapping_add(1);
        &mut self.index_attribute
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
        self.data_version = self.data_version.wrapping_add(1);
    }

    pub fn remove_attribute(&mut self, name: &str) -> Option<Attribute> {
        let removed = self.attributes.remove(name);
        if removed.is_some() {
            self.layout_version = self.layout_version.wrapping_add(1);
            self.structure_version = self.structure_version.wrapping_add(1);
        }
        removed
    }

    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.get(name)
    }

    pub fn get_attribute_mut(&mut self, name: &str) -> Option<&mut Attribute> {
        self.data_version += 1;
        self.attributes.get_mut(name)
    }

    pub fn add_morph_attribute(&mut self, morph_name: &str, attr: Attribute) {
        let entry = self.morph_attributes.entry(morph_name.to_string()).or_insert_with(Vec::new);
        entry.push(attr);
        self.data_version = self.data_version.wrapping_add(1);
    }

    /// 从 morph_attributes 构建紧凑的 Storage Buffers
    /// 布局: [ Target 0 所有顶点 | Target 1 所有顶点 | ... ]
    /// 每个顶点存储 3 个 f32 (compact Vec3)
    pub fn build_morph_storage_buffers(&mut self) {
        // 获取 position morph targets
        let position_attrs = self.morph_attributes.get("position");
        
        if position_attrs.is_none() || position_attrs.unwrap().is_empty() {
            return;
        }
        
        let position_attrs = position_attrs.unwrap();
        let target_count = position_attrs.len();
        
        // 获取每个 target 的顶点数 (假设所有 target 顶点数相同)
        let vertex_count = position_attrs.first()
            .map(|attr| attr.count)
            .unwrap_or(0);
        
        if vertex_count == 0 {
            return;
        }
        
        self.morph_target_count = target_count as u32;
        self.morph_vertex_count = vertex_count;
        
        // 构建 position storage buffer (Target-Major 布局)
        // 总大小 = target_count * vertex_count * 3 floats
        let total_floats = target_count * vertex_count as usize * 3;
        let mut position_data: Vec<f32> = Vec::with_capacity(total_floats);
        
        for attr in position_attrs {
            if let Some(data) = &attr.data {
                // 将 [u8] 转换为 [f32]
                let floats: &[f32] = bytemuck::cast_slice(data.as_slice());
                position_data.extend_from_slice(floats);
            }
        }
        
        if !position_data.is_empty() {
            let buffer_size = position_data.len() * std::mem::size_of::<f32>();
            self.morph_position_buffer = Some(BufferRef::new(
                buffer_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some("MorphPositionStorage")
            ));
            self.morph_position_data = Some(position_data);
        }
        
        // 构建 normal storage buffer (如果有)
        if let Some(normal_attrs) = self.morph_attributes.get("normal") {
            if !normal_attrs.is_empty() {
                let mut normal_data: Vec<f32> = Vec::with_capacity(total_floats);
                
                for attr in normal_attrs {
                    if let Some(data) = &attr.data {
                        let floats: &[f32] = bytemuck::cast_slice(data.as_slice());
                        normal_data.extend_from_slice(floats);
                    }
                }
                
                if !normal_data.is_empty() {
                    let buffer_size = normal_data.len() * std::mem::size_of::<f32>();
                    self.morph_normal_buffer = Some(BufferRef::new(
                        buffer_size,
                        BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        Some("MorphNormalStorage")
                    ));
                    self.morph_normal_data = Some(normal_data);
                }
            }
        }
        
        // 构建 tangent storage buffer (如果有)
        if let Some(tangent_attrs) = self.morph_attributes.get("tangent") {
            if !tangent_attrs.is_empty() {
                let mut tangent_data: Vec<f32> = Vec::with_capacity(total_floats);
                
                for attr in tangent_attrs {
                    if let Some(data) = &attr.data {
                        let floats: &[f32] = bytemuck::cast_slice(data.as_slice());
                        tangent_data.extend_from_slice(floats);
                    }
                }
                
                if !tangent_data.is_empty() {
                    let buffer_size = tangent_data.len() * std::mem::size_of::<f32>();
                    self.morph_tangent_buffer = Some(BufferRef::new(
                        buffer_size,
                        BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        Some("MorphTangentStorage")
                    ));
                    self.morph_tangent_data = Some(tangent_data);
                }
            }
        }
        
        self.data_version = self.data_version.wrapping_add(1);
    }
    
    /// 获取 morph position 数据的字节切片
    pub fn morph_position_bytes(&self) -> Option<&[u8]> {
        self.morph_position_data.as_ref().map(|d| bytemuck::cast_slice(d.as_slice()))
    }
    
    /// 获取 morph normal 数据的字节切片
    pub fn morph_normal_bytes(&self) -> Option<&[u8]> {
        self.morph_normal_data.as_ref().map(|d| bytemuck::cast_slice(d.as_slice()))
    }
    
    /// 获取 morph tangent 数据的字节切片
    pub fn morph_tangent_bytes(&self) -> Option<&[u8]> {
        self.morph_tangent_data.as_ref().map(|d| bytemuck::cast_slice(d.as_slice()))
    }
    
    /// 检查是否有 morph targets
    pub fn has_morph_targets(&self) -> bool {
        self.morph_target_count > 0 && self.morph_position_buffer.is_some()
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
        self.data_version = self.data_version.wrapping_add(1);
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
        self.data_version = self.data_version.wrapping_add(1);
    }

    pub fn compute_vertex_normals(&mut self) {
        // 1. 获取位置属性 (必须存在)
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };
        
        // 获取位置数据引用
        let pos_bytes = match &pos_attr.data {
            Some(data) => data.as_ref(),
            None => return,
        };

        if pos_attr.format != VertexFormat::Float32x3 {
            return;
        }

        let pos_count = pos_attr.count as usize;
        let mut normals = vec![Vec3::ZERO; pos_count];

        // 辅助函数：解析位置
        let pos_stride = pos_attr.stride as usize;
        let pos_offset = pos_attr.offset as usize;
        
        // 这一步只是为了方便读取，和之前一样
        let get_pos = |i: usize| -> Vec3 {
            let start = pos_offset + i * pos_stride;
            // 边界检查，防止恶意数据导致的 panic
            if start + 12 > pos_bytes.len() { return Vec3::ZERO; }
            
            let slice = &pos_bytes[start..start + 12];
            let vals: &[f32; 3] = bytemuck::cast_slice(slice).try_into().unwrap_or(&[0.0; 3]);
            Vec3::from_array(*vals)
        };

        let mut accumulate_triangle = |i0: usize, i1: usize, i2: usize| {
            // 简单的越界保护
            if i0 >= pos_count || i1 >= pos_count || i2 >= pos_count { return; }

            let v0 = get_pos(i0);
            let v1 = get_pos(i1);
            let v2 = get_pos(i2);

            // 面积加权法线 (Area Weighted)
            // 叉积的模长 = 2 * 三角形面积
            let face_normal = (v1 - v0).cross(v2 - v0);

            // 累加
            normals[i0] += face_normal;
            normals[i1] += face_normal;
            normals[i2] += face_normal;
        };

        // 2. 检查索引属性是否存在
        if let Some(index_attr) = &self.index_attribute {
            // === 情况 A: Indexed Geometry (有索引) ===
            if let Some(index_bytes) = &index_attr.data {
                let index_bytes = index_bytes.as_ref();
                
                match index_attr.format {
                    VertexFormat::Uint16 => {
                        let u16s: &[u16] = bytemuck::cast_slice(index_bytes);
                        for chunk in u16s.chunks_exact(3) {
                            accumulate_triangle(chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                        }
                    },
                    VertexFormat::Uint32 => {
                        let u32s: &[u32] = bytemuck::cast_slice(index_bytes);
                        for chunk in u32s.chunks_exact(3) {
                            accumulate_triangle(chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                        }
                    },
                    _ => {} // 不支持的索引格式
                }
            }
        } else {
            // === 情况 B: Non-Indexed Geometry (无索引) ===
            // 假定顶点是每 3 个组成一个三角形 (TRIANGLES 拓扑)
            // 直接遍历 0..pos_count
            for i in (0..pos_count).step_by(3) {
                // 确保最后不够 3 个顶点时不处理
                if i + 2 < pos_count {
                    accumulate_triangle(i, i + 1, i + 2);
                }
            }
        }

        // 3. 最后统一归一化
        for n in normals.iter_mut() {
            *n = n.normalize_or_zero();
        }

        // 4. 创建属性并存回
        let normal_attr = Attribute::new_planar(&normals, VertexFormat::Float32x3);
        self.set_attribute("normal", normal_attr);
    }

    pub fn compute_bounding_volume(&self) {
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };

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
        // 移除 sum_pos，不再需要计算平均值
        // let mut sum_pos = Vec3::ZERO; 
        let mut valid_points_count = 0;

        // Pass 1: 计算 AABB (Min/Max)
        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;

            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _>{
                    let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                    let vec = Vec3::from_array(*vals);

                    min = min.min(vec);
                    max = max.max(vec);
                    // sum_pos += vec; 
                    valid_points_count += 1;
                }
            } else {
                break; 
            }
        }

        if valid_points_count == 0 { return; }

        // 更新 BoundingBox
        *self.bounding_box.borrow_mut() = Some(BoundingBox { min, max });

        // 使用 AABB 的几何中心作为球心
        let aabb_center = (min + max) * 0.5;
        
        let mut max_dist_sq = 0.0;

        // Pass 2: 基于新的中心计算半径
        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;
            
            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _>{
                    let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                    let vec = Vec3::from_array(*vals);
                    
                    // 计算到 AABB 中心的距离
                    let dist_sq = vec.distance_squared(aabb_center);
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                    }
                }
            } else {
                break; 
            }
        }

        *self.bounding_sphere.borrow_mut() = Some(BoundingSphere {
            center: aabb_center,
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

                version: 0,
                format,
                offset,
                stride,
                count,
                step_mode: VertexStepMode::Vertex,
            };

            self.set_attribute(name, attr);
        }
        self.data_version = self.data_version.wrapping_add(1);
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
            self.data_version = self.data_version.wrapping_add(1);
        }
    }

    /// 计算几何体的 Shader 宏定义
    /// 
    /// 使用内部缓存机制，仅当 `layout_version` 发生变化时才重新计算。
    /// 这避免了热路径上的 Map 遍历开销。
    pub fn shader_defines(&self) -> ShaderDefines {
        // 快速路径：检查缓存
        {
            let cache = self.cached_shader_defines.borrow();
            if let Some((cached_version, cached_defines)) = cache.as_ref() {
                if *cached_version == self.layout_version {
                    return cached_defines.clone();
                }
            }
        }

        // 慢速路径：重新计算
        let mut defines = ShaderDefines::new();
        
        for name in self.attributes.keys() {
            let macro_name = format!("HAS_{}", name.to_uppercase());
            defines.set(&macro_name, "1");
        }
        
        // Morph Target 特性检测
        if self.has_morph_targets() {
            defines.set("HAS_MORPH_TARGETS", "1");
            
            if self.morph_normal_buffer.is_some() {
                defines.set("HAS_MORPH_NORMALS", "1");
            }
            if self.morph_tangent_buffer.is_some() {
                defines.set("HAS_MORPH_TANGENTS", "1");
            }
        }

        // Skinning 特性检测
        let has_joints = self.attributes.contains_key("joints");
        let has_weights = self.attributes.contains_key("weights");
        
        if has_joints && has_weights {
            defines.set("HAS_SKINNING", "1");
        }

        // 更新缓存
        *self.cached_shader_defines.borrow_mut() = Some((self.layout_version, defines.clone()));

        defines
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
