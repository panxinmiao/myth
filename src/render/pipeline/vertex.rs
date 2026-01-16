use rustc_hash::FxHashMap;
use wgpu::VertexFormat;
use crate::resources::geometry::{Geometry, Attribute};
use crate::resources::buffer::BufferRef;

#[derive(Debug, Clone)]
pub struct OwnedVertexBufferDesc {
    pub array_stride: u64,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
    pub buffer: BufferRef, 
}

impl OwnedVertexBufferDesc {
    pub fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

/// 生成结果
#[derive(Debug, Clone)]
pub struct GeneratedVertexLayout {
    /// Pipeline 需要的 Layout 列表
    pub buffers: Vec<OwnedVertexBufferDesc>,
    
    /// 注入 Shader 的 WGSL 代码 (struct VertexInput)
    pub vertex_input_code: String,
    
    /// 属性名 -> Shader Location 映射 (供调试或绑定检查)
    pub _attribute_locations: FxHashMap<String, u32>,
}

pub fn generate_vertex_layout(geometry: &Geometry) -> GeneratedVertexLayout {

    let mut buffer_groups: FxHashMap<u64, Vec<(&String, &Attribute)>> = FxHashMap::default();

    for (name, attr) in geometry.attributes() {
        let buffer_id = attr.buffer.id();
        buffer_groups.entry(buffer_id).or_default().push((name, attr));
    }

    let mut sorted_groups: Vec<_> = buffer_groups.into_iter().collect();

    sorted_groups.sort_by(|a, b| {
        // 取该 Buffer 中第一个属性的名字来代表该 Buffer 的顺序
        // 例如 "position" vs "normal"
        let name_a = a.1[0].0; 
        let name_b = b.1[0].0;
        name_a.cmp(name_b)
    });

    let mut owned_layouts = Vec::new();
    let mut wgsl_struct_fields = Vec::new();
    let mut location_map = FxHashMap::default();
    
    let mut current_location = 0;

    // 2. 遍历每个 Buffer 组
    for (buffer_id, mut attrs) in sorted_groups {
        // 排序：保证 Interleaved Buffer 属性顺序正确 (Offset -> Name)
        attrs.sort_by(|a, b| {
            a.1.offset.cmp(&b.1.offset).then(a.0.cmp(b.0))
        });

        // 获取第一个属性的信息作为 Buffer 基准
        let first_attr = attrs[0].1;
        let stride = first_attr.stride;
        let step_mode = first_attr.step_mode;

        // 【检查】同一 Buffer 内所有属性 step_mode 必须一致
        if attrs.iter().any(|(_, a)| a.step_mode != step_mode) {
            log::warn!("Mixed step_mode detected in buffer {:?}. wgpu requires consistent step_mode per buffer. Using {:?}", buffer_id, step_mode);
        }

        let mut wgpu_attributes = Vec::new();

        // 3. 构建 Layout 和 WGSL
        for (name, attr) in attrs {
            let location = current_location;
            current_location += 1;

            // WGPU Attribute
            wgpu_attributes.push(wgpu::VertexAttribute {
                format: attr.format,
                offset: attr.offset,
                shader_location: location,
            });

            // WGSL Code
            let wgsl_type = format_to_wgsl_type(attr.format);
            wgsl_struct_fields.push(format!("    @location({}) {}: {},", location, name, wgsl_type));
            
            location_map.insert(name.clone(), location);
        }

        // 4. 保存为 OwnedLayout
        owned_layouts.push(OwnedVertexBufferDesc {
            array_stride: stride,
            step_mode,
            attributes: wgpu_attributes,
            buffer: first_attr.buffer.clone(),
        });
    }

    // 生成最终 WGSL 代码
    let vertex_input_code = format!(
        "struct VertexInput {{\n{}\n}};", 
        wgsl_struct_fields.join("\n")
    );

    GeneratedVertexLayout {
        buffers: owned_layouts,
        vertex_input_code,
        _attribute_locations: location_map,
    }
}

// 完整的类型映射表
fn format_to_wgsl_type(format: VertexFormat) -> &'static str {
    match format {
        // Float
        VertexFormat::Float32 => "f32",
        VertexFormat::Float32x2 => "vec2<f32>",
        VertexFormat::Float32x3 => "vec3<f32>",
        VertexFormat::Float32x4 => "vec4<f32>",
        VertexFormat::Float64 => "f32", // 注意：WGSL暂不支持 f64，通常作为 f32 处理
        VertexFormat::Float64x2 => "vec2<f32>",
        VertexFormat::Float64x3 => "vec3<f32>",
        VertexFormat::Float64x4 => "vec4<f32>",
        
        // Uint
        VertexFormat::Uint32 => "u32",
        VertexFormat::Uint32x2 => "vec2<u32>",
        VertexFormat::Uint32x3 => "vec3<u32>",
        VertexFormat::Uint32x4 => "vec4<u32>",
        
        // Sint
        VertexFormat::Sint32 => "i32",
        VertexFormat::Sint32x2 => "vec2<i32>",
        VertexFormat::Sint32x3 => "vec3<i32>",
        VertexFormat::Sint32x4 => "vec4<i32>",
        
        // Normalized (在 Shader 中自动转为 float)
        VertexFormat::Unorm8x2 => "vec2<f32>",
        VertexFormat::Unorm8x4 => "vec4<f32>",
        VertexFormat::Snorm8x2 => "vec2<f32>",
        VertexFormat::Snorm8x4 => "vec4<f32>",
        VertexFormat::Unorm16x2 => "vec2<f32>",
        VertexFormat::Unorm16x4 => "vec4<f32>",
        VertexFormat::Snorm16x2 => "vec2<f32>",
        VertexFormat::Snorm16x4 => "vec4<f32>",

        // 整数类型 (通常用于索引或特殊用途)
        VertexFormat::Uint8x2 => "vec2<u32>",
        VertexFormat::Uint8x4 => "vec4<u32>",
        VertexFormat::Sint8x2 => "vec2<i32>",
        VertexFormat::Sint8x4 => "vec4<i32>",
        VertexFormat::Uint16x2 => "vec2<u32>",
        VertexFormat::Uint16x4 => "vec4<u32>",
        VertexFormat::Sint16x2 => "vec2<i32>",
        VertexFormat::Sint16x4 => "vec4<i32>",

        // 其他格式
        _ => "f32", // 默认回退为 f32，实际使用中应该避免

    }
}