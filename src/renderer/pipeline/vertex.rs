//! Vertex Layout Generator
//!
//! Generates WGPU vertex layouts and WGSL code from Geometry attributes

use crate::resources::buffer::BufferRef;
use crate::resources::geometry::{Attribute, Geometry};
use rustc_hash::FxHashMap;
use wgpu::VertexFormat;

#[derive(Debug, Clone)]
pub struct OwnedVertexBufferDesc {
    pub array_stride: u64,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
    pub buffer: BufferRef,
}

impl OwnedVertexBufferDesc {
    #[must_use]
    pub fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedVertexLayout {
    pub buffers: Vec<OwnedVertexBufferDesc>,
    pub vertex_input_code: String,
    /// Attribute locations map (currently unused, reserved for future use)
    pub attribute_locations: FxHashMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexLayoutSignature {
    pub buffers: Vec<VertexBufferLayoutSignature>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBufferLayoutSignature {
    pub array_stride: u64,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl GeneratedVertexLayout {
    #[must_use]
    pub fn to_signature(&self) -> VertexLayoutSignature {
        let buffers = self
            .buffers
            .iter()
            .map(|b| VertexBufferLayoutSignature {
                array_stride: b.array_stride,
                step_mode: b.step_mode,
                attributes: b.attributes.clone(),
            })
            .collect();

        VertexLayoutSignature { buffers }
    }
}

#[must_use]
pub fn generate_vertex_layout(geometry: &Geometry) -> GeneratedVertexLayout {
    let mut buffer_groups: FxHashMap<u64, Vec<(&String, &Attribute)>> = FxHashMap::default();

    for (name, attr) in geometry.attributes() {
        let buffer_id = attr.buffer.id();
        buffer_groups
            .entry(buffer_id)
            .or_default()
            .push((name, attr));
    }

    let mut sorted_groups: Vec<_> = buffer_groups.into_iter().collect();

    // Sort attributes within each group first to ensure a.1[0] is deterministic (e.g., by offset or name)
    for (_, attrs) in &mut sorted_groups {
        attrs.sort_by(|a, b| a.1.offset.cmp(&b.1.offset).then(a.0.cmp(b.0)));
    }

    // Then sort the groups themselves
    sorted_groups.sort_by(|a, b| {
        let name_a = a.1[0].0;
        let name_b = b.1[0].0;
        name_a.cmp(name_b)
    });

    let mut owned_layouts = Vec::new();
    let mut wgsl_struct_fields = Vec::new();
    let mut location_map = FxHashMap::default();
    let mut current_location = 0;

    for (buffer_id, attrs) in sorted_groups {
        let first_attr = attrs[0].1;
        let stride = first_attr.stride;
        let step_mode = first_attr.step_mode;

        if attrs.iter().any(|(_, a)| a.step_mode != step_mode) {
            log::warn!("Mixed step_mode in buffer {buffer_id:?}. Using {step_mode:?}");
        }

        let mut wgpu_attributes = Vec::new();

        for (name, attr) in attrs {
            let location = current_location;
            current_location += 1;

            wgpu_attributes.push(wgpu::VertexAttribute {
                format: attr.format,
                offset: attr.offset,
                shader_location: location,
            });

            let wgsl_type = format_to_wgsl_type(attr.format);
            wgsl_struct_fields.push(format!("    @location({location}) {name}: {wgsl_type},"));
            location_map.insert(name.clone(), location);
        }

        owned_layouts.push(OwnedVertexBufferDesc {
            array_stride: stride,
            step_mode,
            attributes: wgpu_attributes,
            buffer: first_attr.buffer.clone(),
        });
    }

    let vertex_input_code = format!(
        "struct VertexInput {{\n{}\n}};",
        wgsl_struct_fields.join("\n")
    );

    GeneratedVertexLayout {
        buffers: owned_layouts,
        vertex_input_code,
        attribute_locations: location_map,
    }
}

#[allow(clippy::match_same_arms)]
fn format_to_wgsl_type(format: VertexFormat) -> &'static str {
    match format {
        VertexFormat::Float32 => "f32",
        VertexFormat::Float32x2 => "vec2<f32>",
        VertexFormat::Float32x3 => "vec3<f32>",
        VertexFormat::Float32x4 => "vec4<f32>",
        VertexFormat::Float64 => "f32",
        VertexFormat::Float64x2 => "vec2<f32>",
        VertexFormat::Float64x3 => "vec3<f32>",
        VertexFormat::Float64x4 => "vec4<f32>",
        VertexFormat::Uint32 => "u32",
        VertexFormat::Uint32x2 => "vec2<u32>",
        VertexFormat::Uint32x3 => "vec3<u32>",
        VertexFormat::Uint32x4 => "vec4<u32>",
        VertexFormat::Sint32 => "i32",
        VertexFormat::Sint32x2 => "vec2<i32>",
        VertexFormat::Sint32x3 => "vec3<i32>",
        VertexFormat::Sint32x4 => "vec4<i32>",
        VertexFormat::Unorm8x2 => "vec2<f32>",
        VertexFormat::Unorm8x4 => "vec4<f32>",
        VertexFormat::Snorm8x2 => "vec2<f32>",
        VertexFormat::Snorm8x4 => "vec4<f32>",
        VertexFormat::Unorm16x2 => "vec2<f32>",
        VertexFormat::Unorm16x4 => "vec4<f32>",
        VertexFormat::Snorm16x2 => "vec2<f32>",
        VertexFormat::Snorm16x4 => "vec4<f32>",
        VertexFormat::Uint8x2 => "vec2<u32>",
        VertexFormat::Uint8x4 => "vec4<u32>",
        VertexFormat::Sint8x2 => "vec2<i32>",
        VertexFormat::Sint8x4 => "vec4<i32>",
        VertexFormat::Uint16x2 => "vec2<u32>",
        VertexFormat::Uint16x4 => "vec4<u32>",
        VertexFormat::Sint16x2 => "vec2<i32>",
        VertexFormat::Sint16x4 => "vec4<i32>",
        _ => "f32",
    }
}
