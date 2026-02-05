use core::ops::Range;
use glam::{Affine3A, Vec3, Vec4};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;
use wgpu::{BufferUsages, PrimitiveTopology, VertexFormat, VertexStepMode};

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
    /// Creates a Planar (non-interleaved) attribute
    pub fn new_planar<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        let size = raw_data.len();

        // Create handle
        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            Some("GeometryVertexAttr"),
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

    /// Creates an Instance attribute
    pub fn new_instanced<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        let size = raw_data.len();

        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            Some("GeometryInstanceAttr"),
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

    /// Creates an Interleaved attribute
    /// Multiple Attributes can share the same `BufferRef` and data (Arc)
    pub fn new_interleaved(
        buffer: BufferRef,
        data: Option<Arc<Vec<u8>>>,
        format: VertexFormat,
        offset: u64,
        count: u32,
        stride: u64,
        step_mode: VertexStepMode,
    ) -> Self {
        Self {
            buffer,
            data,
            version: NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed),
            format,
            offset,
            count,
            stride,
            step_mode,
        }
    }

    /// Updates data in-place (preserves ID, reuses GPU memory)
    /// Uses `Arc::make_mut` to implement Copy-On-Write
    pub fn update_data<T: bytemuck::Pod>(&mut self, new_data: &[T]) {
        if let Some(arc_vec) = &mut self.data {
            // Arc::make_mut: modifies directly if only one reference; otherwise clones then modifies
            let vec = Arc::make_mut(arc_vec);

            let bytes: &[u8] = bytemuck::cast_slice(new_data);

            // If length changed, need to resize the Vec
            if vec.len() != bytes.len() {
                vec.resize(bytes.len(), 0);
            }
            vec.copy_from_slice(bytes);

            // Update metadata
            self.count = new_data.len() as u32;
            self.version = NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Partially updates attribute data
    pub fn update_region<T: bytemuck::Pod>(&mut self, offset_bytes: u64, new_data: &[T]) {
        if let Some(arc_vec) = &mut self.data {
            let vec = Arc::make_mut(arc_vec);
            let bytes = bytemuck::cast_slice(new_data);

            let start = offset_bytes as usize;
            let end = start + bytes.len();

            // Bounds check
            if end <= vec.len() {
                vec[start..end].copy_from_slice(bytes);
                self.version = NEXT_ATTR_VERSION.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[must_use]
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

    #[must_use]
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

    #[must_use]
    pub fn read<T>(&self, i: u32) -> Option<T>
    where
        T: bytemuck::Pod,
    {
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
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    #[must_use]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
    #[must_use]
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[must_use]
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
            // Assuming Affine3A can directly transform_point3
            let transformed = matrix.transform_point3(point);
            new_min = new_min.min(transformed);
            new_max = new_max.max(transformed);
        }

        Self {
            min: new_min,
            max: new_max,
        }
    }

    // Simple inflation method
    #[must_use]
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

    /// Morph Target Storage Buffers (compact f32 storage)
    /// Layout: [ Target 0 all vertices | Target 1 all vertices | ... ]
    /// Each vertex stores 3 f32 values (Position/Normal/Tangent displacement)
    pub morph_position_buffer: Option<BufferRef>,
    pub morph_normal_buffer: Option<BufferRef>,
    pub morph_tangent_buffer: Option<BufferRef>,

    /// Morph Target data (kept on CPU side to support uploading)
    morph_position_data: Option<Vec<f32>>,
    morph_normal_data: Option<Vec<f32>>,
    morph_tangent_data: Option<Vec<f32>>,

    /// Vertex count per target
    pub morph_vertex_count: u32,
    /// Morph target count
    pub morph_target_count: u32,

    pub topology: PrimitiveTopology,
    pub draw_range: Range<u32>,

    pub bounding_box: RwLock<Option<BoundingBox>>,
    pub bounding_sphere: RwLock<Option<BoundingSphere>>,

    /// `ShaderDefines` cache: (`layout_version`, `cached_defines`)
    cached_shader_defines: RwLock<Option<(u64, ShaderDefines)>>,
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Geometry {
    #[must_use]
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
            bounding_box: RwLock::new(None),
            bounding_sphere: RwLock::new(None),
            cached_shader_defines: RwLock::new(None),
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
        let entry = self
            .morph_attributes
            .entry(morph_name.to_string())
            .or_default();
        entry.push(attr);
        self.data_version = self.data_version.wrapping_add(1);
    }

    /// Builds compact Storage Buffers from `morph_attributes`
    /// Layout: [ Target 0 all vertices | Target 1 all vertices | ... ]
    /// Each vertex stores 3 f32 values (compact Vec3)
    pub fn build_morph_storage_buffers(&mut self) {
        // Get position morph targets
        let position_attrs = self.morph_attributes.get("position");

        if position_attrs.is_none() || position_attrs.unwrap().is_empty() {
            return;
        }

        let position_attrs = position_attrs.unwrap();
        let target_count = position_attrs.len();

        // Get vertex count per target (assuming all targets have the same vertex count)
        let vertex_count = position_attrs.first().map_or(0, |attr| attr.count);

        if vertex_count == 0 {
            return;
        }

        self.morph_target_count = target_count as u32;
        self.morph_vertex_count = vertex_count;

        // Build position storage buffer (Target-Major layout)
        // Total size = target_count * vertex_count * 3 floats
        let total_floats = target_count * vertex_count as usize * 3;
        let mut position_data: Vec<f32> = Vec::with_capacity(total_floats);

        for attr in position_attrs {
            if let Some(data) = &attr.data {
                // Convert [u8] to [f32]
                let floats: &[f32] = bytemuck::cast_slice(data.as_slice());
                position_data.extend_from_slice(floats);
            }
        }

        if !position_data.is_empty() {
            let buffer_size = position_data.len() * std::mem::size_of::<f32>();
            self.morph_position_buffer = Some(BufferRef::new(
                buffer_size,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some("MorphPositionStorage"),
            ));
            self.morph_position_data = Some(position_data);
        }

        // Build normal storage buffer (if available)
        if let Some(normal_attrs) = self.morph_attributes.get("normal")
            && !normal_attrs.is_empty()
        {
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
                    Some("MorphNormalStorage"),
                ));
                self.morph_normal_data = Some(normal_data);
            }
        }

        // Build tangent storage buffer (if available)
        if let Some(tangent_attrs) = self.morph_attributes.get("tangent")
            && !tangent_attrs.is_empty()
        {
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
                    Some("MorphTangentStorage"),
                ));
                self.morph_tangent_data = Some(tangent_data);
            }
        }

        self.data_version = self.data_version.wrapping_add(1);
    }

    /// Gets the byte slice of morph position data
    pub fn morph_position_bytes(&self) -> Option<&[u8]> {
        self.morph_position_data
            .as_ref()
            .map(|d| bytemuck::cast_slice(d.as_slice()))
    }

    /// Gets the byte slice of morph normal data
    pub fn morph_normal_bytes(&self) -> Option<&[u8]> {
        self.morph_normal_data
            .as_ref()
            .map(|d| bytemuck::cast_slice(d.as_slice()))
    }

    /// Gets the byte slice of morph tangent data
    pub fn morph_tangent_bytes(&self) -> Option<&[u8]> {
        self.morph_tangent_data
            .as_ref()
            .map(|d| bytemuck::cast_slice(d.as_slice()))
    }

    /// Checks if morph targets exist
    pub fn has_morph_targets(&self) -> bool {
        self.morph_target_count > 0 && self.morph_position_buffer.is_some()
    }

    pub fn set_indices(&mut self, indices: &[u16]) {
        let raw_data = bytemuck::cast_slice(indices).to_vec();
        let size = raw_data.len();

        let buffer_ref = BufferRef::new(
            size,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            Some("IndexBuffer"),
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
            Some("IndexBuffer"),
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
        // 1. Get position attribute (must exist)
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };

        // Get position data reference
        let pos_bytes = match &pos_attr.data {
            Some(data) => data.as_ref(),
            None => return,
        };

        if pos_attr.format != VertexFormat::Float32x3 {
            return;
        }

        let pos_count = pos_attr.count as usize;
        let mut normals = vec![Vec3::ZERO; pos_count];

        // Helper function: parse position
        let pos_stride = pos_attr.stride as usize;
        let pos_offset = pos_attr.offset as usize;

        // This step is just for convenience reading, same as before
        let get_pos = |i: usize| -> Vec3 {
            let start = pos_offset + i * pos_stride;
            // Bounds check to prevent panic from malicious data
            if start + 12 > pos_bytes.len() {
                return Vec3::ZERO;
            }

            let slice = &pos_bytes[start..start + 12];
            let vals: &[f32; 3] = bytemuck::cast_slice(slice).try_into().unwrap_or(&[0.0; 3]);
            Vec3::from_array(*vals)
        };

        let mut accumulate_triangle = |i0: usize, i1: usize, i2: usize| {
            // Simple out of bounds protection
            if i0 >= pos_count || i1 >= pos_count || i2 >= pos_count {
                return;
            }

            let v0 = get_pos(i0);
            let v1 = get_pos(i1);
            let v2 = get_pos(i2);

            // Area weighted normal
            // Cross product magnitude = 2 * triangle area
            let face_normal = (v1 - v0).cross(v2 - v0);

            // Accumulate
            normals[i0] += face_normal;
            normals[i1] += face_normal;
            normals[i2] += face_normal;
        };

        // 2. Check if index attribute exists
        if let Some(index_attr) = &self.index_attribute {
            // === Case A: Indexed Geometry ===
            if let Some(index_bytes) = &index_attr.data {
                let index_bytes = index_bytes.as_ref();

                match index_attr.format {
                    VertexFormat::Uint16 => {
                        let u16s: &[u16] = bytemuck::cast_slice(index_bytes);
                        for chunk in u16s.chunks_exact(3) {
                            accumulate_triangle(
                                chunk[0] as usize,
                                chunk[1] as usize,
                                chunk[2] as usize,
                            );
                        }
                    }
                    VertexFormat::Uint32 => {
                        let u32s: &[u32] = bytemuck::cast_slice(index_bytes);
                        for chunk in u32s.chunks_exact(3) {
                            accumulate_triangle(
                                chunk[0] as usize,
                                chunk[1] as usize,
                                chunk[2] as usize,
                            );
                        }
                    }
                    _ => {} // Unsupported index format
                }
            }
        } else {
            // === Case B: Non-Indexed Geometry ===
            // Assumes vertices form triangles in groups of 3 (TRIANGLES topology)
            // Iterate directly over 0..pos_count
            for i in (0..pos_count).step_by(3) {
                // Ensure we don't process when fewer than 3 vertices remain
                if i + 2 < pos_count {
                    accumulate_triangle(i, i + 1, i + 2);
                }
            }
        }

        // 3. Normalize all at the end
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }

        // 4. Create attribute and store back
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
        // Removed sum_pos, no longer need to calculate average
        // let mut sum_pos = Vec3::ZERO;
        let mut valid_points_count = 0;

        // Pass 1: Compute AABB (Min/Max)
        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;

            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _> {
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

        if valid_points_count == 0 {
            return;
        }

        // Update BoundingBox
        *self.bounding_box.write() = Some(BoundingBox { min, max });

        // Use AABB geometric center as sphere center
        let aabb_center = (min + max) * 0.5;

        let mut max_dist_sq = 0.0;

        // Pass 2: Compute radius based on the new center
        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;

            if let Some(slice) = data.get(start..end) {
                if let Ok(bytes) = slice.try_into() as Result<&[u8; 12], _> {
                    let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
                    let vec = Vec3::from_array(*vals);

                    // Calculate distance to AABB center
                    let dist_sq = vec.distance_squared(aabb_center);
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                    }
                }
            } else {
                break;
            }
        }

        *self.bounding_sphere.write() = Some(BoundingSphere {
            center: aabb_center,
            radius: max_dist_sq.sqrt(),
        });
    }

    /// Sets interleaved attributes
    /// Creates multiple Attributes sharing the same Buffer from an interleaved array
    pub fn set_interleaved_attributes(
        &mut self,
        interleaved_data: Vec<u8>, // Raw interleaved data
        stride: u64,
        attributes: Vec<(&str, VertexFormat, u64)>, // (name, format, offset)
    ) {
        let shared_data = Arc::new(interleaved_data);
        let count = (shared_data.len() as u64 / stride) as u32;

        let shared_buffer_ref = BufferRef::new(
            shared_data.len(),
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            Some("InterleavedBuffer"),
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

    /// Partially updates attribute data
    pub fn update_attribute_region<T: bytemuck::Pod>(
        &mut self,
        name: &str,
        offset_bytes: u64,
        data: &[T],
    ) {
        if let Some(attr) = self.attributes.get_mut(name) {
            attr.update_region(offset_bytes, data);
            self.data_version = self.data_version.wrapping_add(1);
        }
    }

    /// Computes the geometry's shader macro definitions
    ///
    /// Uses internal caching mechanism, only recalculates when `layout_version` changes.
    /// This avoids Map traversal overhead on the hot path.
    pub fn shader_defines(&self) -> ShaderDefines {
        // Fast path: check cache
        {
            let cache = self.cached_shader_defines.read();
            if let Some((cached_version, cached_defines)) = cache.as_ref()
                && *cached_version == self.layout_version
            {
                return cached_defines.clone();
            }
        }

        // Slow path: recalculate
        let mut defines = ShaderDefines::new();

        for name in self.attributes.keys() {
            let macro_name = format!("HAS_{}", name.to_uppercase());
            defines.set(&macro_name, "1");
        }

        // Morph Target feature detection
        if self.has_morph_targets() {
            defines.set("HAS_MORPH_TARGETS", "1");

            if self.morph_normal_buffer.is_some() {
                defines.set("HAS_MORPH_NORMALS", "1");
            }
            if self.morph_tangent_buffer.is_some() {
                defines.set("HAS_MORPH_TANGENTS", "1");
            }
        }

        // Skinning feature detection
        let has_joints = self.attributes.contains_key("joints");
        let has_weights = self.attributes.contains_key("weights");

        if has_joints && has_weights {
            defines.set("SUPPORT_SKINNING", "1");
        }

        // Update cache
        *self.cached_shader_defines.write() = Some((self.layout_version, defines.clone()));

        defines
    }

    #[must_use]
    pub fn new_box(width: f32, height: f32, depth: f32) -> Self {
        primitives::create_box(width, height, depth)
    }

    #[must_use]
    pub fn new_sphere(radius: f32) -> Self {
        primitives::create_sphere(primitives::SphereOptions {
            radius,
            ..Default::default()
        })
    }

    #[must_use]
    pub fn new_plane(width: f32, height: f32) -> Self {
        primitives::create_plane(primitives::PlaneOptions {
            width,
            height,
            ..Default::default()
        })
    }
}
