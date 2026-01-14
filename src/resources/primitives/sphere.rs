use crate::resources::geometry::{Geometry, Attribute};
use wgpu::VertexFormat;
use std::f32::consts::PI;

pub struct SphereOptions {
    pub radius: f32,
    pub width_segments: u32,
    pub height_segments: u32,
}

impl Default for SphereOptions {
    fn default() -> Self {
        Self {
            radius: 1.0,
            width_segments: 32,
            height_segments: 16,
        }
    }
}

pub fn create_sphere(options: SphereOptions) -> Geometry {
    let radius = options.radius;
    let width_segments = options.width_segments.max(3);
    let height_segments = options.height_segments.max(2);

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // 生成顶点数据
    for y in 0..=height_segments {
        let v_ratio = y as f32 / height_segments as f32;
        // 纬度角度：从 0 到 PI (南极到北极)
        let theta = v_ratio * PI; 
        
        // y 坐标 (Y-up)
        let py = -radius * (theta.cos()); 
        // 当前纬度圈的半径
        let ring_radius = radius * (theta.sin());

        for x in 0..=width_segments {
            let u_ratio = x as f32 / width_segments as f32;
            // 经度角度：从 0 到 2PI
            let phi = u_ratio * 2.0 * PI;

            let px = -ring_radius * phi.cos();
            let pz = ring_radius * phi.sin();

            positions.push([px, py, pz]);

            // 法线就是归一化的位置向量
            let nx = px / radius;
            let ny = py / radius;
            let nz = pz / radius;
            normals.push([nx, ny, nz]);

            // UV
            uvs.push([u_ratio, 1.0 - v_ratio]);
        }
    }

    // 生成索引
    // 每一格由两个三角形组成
    let stride = width_segments + 1;
    for y in 0..height_segments {
        for x in 0..width_segments {
            let v0 = y * stride + x;
            let v1 = v0 + 1;
            let v2 = (y + 1) * stride + x;
            let v3 = v2 + 1;

            // 如果不是最后一圈（南极点不需要第一个三角形，但为了算法简单通常不做特殊剔除，退化三角形会被GPU忽略）
            if y != 0 || true { 
                indices.push(v0 as u16);
                indices.push(v1 as u16);
                indices.push(v2 as u16);
            }
            
            // 如果不是第一圈（北极点）
            if y != height_segments - 1 || true {
                indices.push(v1 as u16);
                indices.push(v3 as u16);
                indices.push(v2 as u16);
            }
        }
    }

    let mut geo = Geometry::new();
    geo.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));
    geo.set_attribute("normal", Attribute::new_planar(&normals, VertexFormat::Float32x3));
    geo.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
    geo.set_indices(&indices);

    geo.compute_bounding_volume();
    geo
}