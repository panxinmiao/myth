use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use wgpu::{PrimitiveTopology, VertexFormat, VertexStepMode};
use glam::Vec3;
use core::ops::Range;
use crate::core::binding::{Bindable, BindingDescriptor, BindingResource, BindingType};

// ============================================================================
// 1. 底层数据存储：GeometryBuffer
// ============================================================================

#[derive(Debug)]
pub struct GeometryBuffer {
    pub id: Uuid,         // 资源的唯一标识，Renderer 用它做 Cache Key
    pub data: Vec<u8>,    // 原始字节数据
    pub stride: u64,      // 步长
    pub version: u64,     // 数据版本号 (data 变了 +1)
}

impl GeometryBuffer {
    /// 创建一个新的 Buffer
    pub fn new<T: bytemuck::Pod>(data: &[T], stride: u64) -> Self {

        if stride % 4 != 0 {
            log::warn!("GeometryBuffer created with stride {} (not a multiple of 4). This may cause WGPU validation errors.", stride);
        }

        Self {
            id: Uuid::new_v4(),
            data: bytemuck::cast_slice(data).to_vec(),
            stride,
            version: 0,
        }
    }

    /// 更新数据
    pub fn update<T: bytemuck::Pod>(&mut self, data: &[T]) {
        self.data = bytemuck::cast_slice(data).to_vec();
        self.version += 1;
    }
}

// ============================================================================
// 2. 数据视图：Attribute
// ============================================================================

#[derive(Debug, Clone)]
pub struct Attribute {
    pub buffer: Arc<RwLock<GeometryBuffer>>, // 共享所有权
    pub format: VertexFormat,
    pub offset: u64,
    pub count: u32,
    pub step_mode: VertexStepMode,
}

impl Attribute {
    /// 快捷创建：普通顶点属性 (Per-Vertex)
    /// 适用于 Position, Normal, UV 等
    pub fn new_planar<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let stride = std::mem::size_of::<T>() as u64;
        let buffer = GeometryBuffer::new(data, stride);
        Self {
            buffer: Arc::new(RwLock::new(buffer)),
            format,
            offset: 0,
            count: data.len() as u32,
            step_mode: VertexStepMode::Vertex,
        }
    }

    /// 快捷创建：实例化属性 (Per-Instance)
    /// 适用于 InstanceMatrix, InstanceColor 等
    pub fn new_instanced<T: bytemuck::Pod>(data: &[T], format: VertexFormat) -> Self {
        let stride = std::mem::size_of::<T>() as u64;
        let buffer = GeometryBuffer::new(data, stride);
        Self {
            buffer: Arc::new(RwLock::new(buffer)),
            format,
            offset: 0,
            count: data.len() as u32,
            step_mode: VertexStepMode::Instance,
        }
    }

    /// 快捷创建 Interleaved Attribute (需要传入已存在的 Shared Buffer)
    pub fn new_interleaved(
        buffer: Arc<RwLock<GeometryBuffer>>, 
        format: VertexFormat, 
        offset: u64, 
        count: u32,
        step_mode: VertexStepMode
    ) -> Self {
        Self { buffer, format, offset, count, step_mode }
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

// ============================================================================
// 4. 核心容器：Geometry
// ============================================================================
#[derive(Debug, Clone)]
pub struct Geometry {
    pub id: Uuid, // Geometry 自身的 ID
    pub version: u64, // 结构版本号 (如果 set_attribute 了，需要重建 Pipeline)

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
            id: Uuid::new_v4(),
            version: 0,
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

    pub fn set_attribute(&mut self, name: &str, attr: Attribute) {
        self.attributes.insert(name.to_string(), attr);
        self.version += 1; // 标记结构变动
    }

    pub fn set_indices(&mut self, indices: &[u16]) {
        self.index_attribute = Some(Attribute::new_planar(indices, VertexFormat::Uint16));
        self.version += 1;
    }

    pub fn set_indices_u32(&mut self, indices: &[u32]) {
        self.index_attribute = Some(Attribute::new_planar(indices, VertexFormat::Uint32));
        self.version += 1;
    }

    /// 计算包围体 (AABB 和 Sphere 分开计算，追求更优的包围球)
    pub fn compute_bounding_volume(&mut self) {
        let pos_attr = match self.attributes.get("position") {
            Some(attr) => attr,
            None => return,
        };

        // 1. 准备数据读取
        let buffer_guard = pos_attr.buffer.read().unwrap();
        let data = &buffer_guard.data;
        let stride = buffer_guard.stride as usize;
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
}


// 实现 Bindable
impl Bindable for Geometry {

    fn get_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'_>>) {
        let mut bindings = Vec::new();
        let mut resources = Vec::new();
        // let mut index = 0;

        // todo : 根据morph target, 生成morth texture的binding
        // for (name, attr) in &self.attributes {
        //     bindings.push(BindingDescriptor {
        //         name: name,
        //         index: bindings.len() as u32,
        //         bind_type: BindingType::StorageBuffer,
        //         visibility: wgpu::ShaderStages::VERTEX,
        //     });
        //     resources.push(BindingResource::Buffer(&attr.buffer.read().unwrap().data));
        // }
        (bindings, resources)

    }

}