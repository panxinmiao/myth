use std::collections::HashMap;
use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::{PrimitiveTopology, VertexFormat, VertexStepMode, BufferUsages};
use glam::Vec3;
use core::ops::Range;

use crate::core::buffer::{BufferRef};
// [新增] 引入 Shader 编译选项 (复用 Material 的，或者新建一个 GeometryCompilationOptions)
// 为了简单起见，且通常 Defines 是合并处理的，我们可以复用或新建。
// 这里建议新建一个 GeometryFeatures，为了演示，我先使用简单的 Bitflags。
use bitflags::bitflags;

// ============================================================================
// 2. 数据视图：Attribute
// ============================================================================

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
    /// 快捷创建：普通顶点属性 (Per-Vertex)
    /// 适用于 Position, Normal, UV 等
    pub fn new_planar<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let stride = std::mem::size_of::<T>() as u64;
        // 创建通用 DataBuffer
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

    /// 快捷创建：实例化属性 (Per-Instance)
    /// 适用于 InstanceMatrix, InstanceColor 等
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

    /// 快捷创建 Interleaved Attribute (需要传入已存在的 Shared Buffer)
    pub fn new_interleaved(
        buffer: BufferRef, 
        format: VertexFormat, 
        offset: u64, 
        count: u32,
        stride: u64, // 需要显式传入 stride
        step_mode: VertexStepMode
    ) -> Self {
        Self { buffer, format, offset, count, stride, step_mode }
    }
}

// ============================================================================
// 3. 辅助结构：包围盒与球
// ============================================================================

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

// ============================================================================
// 4. 核心容器：Geometry
// ============================================================================
#[derive(Debug)]
pub struct Geometry {
    pub uuid: Uuid, // Geometry 自身的 ID
    pub version: AtomicU64, // 结构版本号 (如果 set_attribute 了，需要重建 Pipeline)

    // 核心数据
    pub attributes: HashMap<String, Attribute>,
    pub index_attribute: Option<Attribute>,
    
    // 变形目标 (Morph Targets)
    // Key 是属性名 (如 "position"), Value 是每个 Target 的 Attribute 列表
    // 例如: morph_attributes["position"] = [Target1_Pos, Target2_Pos, ...]
    pub morph_attributes: HashMap<String, Vec<Attribute>>,
    pub morph_target_names: Vec<String>, // 表情名列表

    // 渲染配置
    pub topology: PrimitiveTopology,
    pub draw_range: Range<u32>,

    // 空间数据 (用于剔除)
    pub bounding_box: Option<BoundingBox>,
    pub bounding_sphere: Option<BoundingSphere>,
}

impl Geometry {
    pub fn new() -> Self {
        Self {
            uuid: Uuid::new_v4(),
            version: AtomicU64::new(0),
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

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    pub fn mark_dirty(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_attribute(&mut self, name: &str, attr: Attribute) {
        self.attributes.insert(name.to_string(), attr);
        self.mark_dirty(); // 标记结构变动
    }

    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.get(name)
    }

    pub fn set_indices(&mut self, indices: &[u16]) {
        // self.index_attribute = Some(Attribute::new_planar(indices, VertexFormat::Uint16));
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
            step_mode: VertexStepMode::Vertex, // Index buffer 忽略 step_mode
        });
        self.mark_dirty();
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
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// 计算包围体 (AABB 和 Sphere 分开计算，追求更优的包围球)
    pub fn compute_bounding_volume(&mut self) {
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };

        // 1. 准备数据读取
        let data = &pos_attr.buffer.read_data();
        let stride = pos_attr.stride as usize;
        let offset = pos_attr.offset as usize;
        let count = pos_attr.count as usize;

        if pos_attr.format != VertexFormat::Float32x3 {
            return; // 暂不支持非 float32x3
        }

        // ==========================================
        // 阶段 1: 计算 AABB 和 球心 (Centroid)
        // ==========================================
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut sum_pos = Vec3::ZERO;
        let mut valid_points_count = 0;

        // 第一次遍历：计算 AABB 和 所有点的重心
        for i in 0..count {
            let start = offset + i * stride;
            let end = start + 12;
            if end > data.len() { break; }

            let bytes: &[u8; 12] = data[start..end].try_into().unwrap();
            let vals: &[f32; 3] = bytemuck::cast_ref(bytes);
            let vec = Vec3::from_array(*vals);

            // AABB 更新
            min = min.min(vec);
            max = max.max(vec);
            
            // 重心累加
            sum_pos += vec;
            valid_points_count += 1;
        }

        if valid_points_count == 0 { return; }

        self.bounding_box = Some(BoundingBox { min, max });

        // ==========================================
        // 阶段 2: 计算更紧凑的包围球 (Bounding Sphere)
        // ==========================================
        
        // 策略：使用重心作为球心，而不是 AABB 中心
        // AABB 中心受离群点(outliers)影响很大，重心相对稳定
        let centroid = sum_pos / (valid_points_count as f32);
        
        let mut max_dist_sq = 0.0;

        // 第二次遍历：找到离重心最远的点，确定半径
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
        
        // 注意：这种方法比 AABB.center 得到的球更小，更适合做剔除。
        // 虽然不是数学上的最小覆盖球(Welzl算法)，但在引擎初始化速度和剔除效率之间是最好的平衡。
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
        // 根据 buffer 是否存在来判断，而不是仅仅看 attributes
        // if self.morph_buffer.is_some() {
        //     features |= GeometryFeatures::USE_MORPHING;
        // }
        if self.morph_attributes.len() > 0 {
            features |= GeometryFeatures::USE_MORPHING;
        }
        // Skinning 同理...

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
        
        // 别忘了计算包围球，这对于视锥剔除很重要！
        geo.compute_bounding_volume();

        geo
    }
}

