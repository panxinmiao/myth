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

    // Generate vertex data
    for y in 0..=height_segments {
        let v_ratio = y as f32 / height_segments as f32;
        // Latitude angle: from 0 to PI (south pole to north pole)
        let theta = v_ratio * PI; 
        
        // y coordinate (Y-up)
        let py = -radius * (theta.cos()); 
        // Radius of current latitude ring
        let ring_radius = radius * (theta.sin());

        for x in 0..=width_segments {
            let u_ratio = x as f32 / width_segments as f32;
            // Longitude angle: from 0 to 2*PI
            let phi = u_ratio * 2.0 * PI;

            let px = -ring_radius * phi.cos();
            let pz = ring_radius * phi.sin();

            positions.push([px, py, pz]);

            // Normal is the normalized position vector
            let nx = px / radius;
            let ny = py / radius;
            let nz = pz / radius;
            normals.push([nx, ny, nz]);

            // UV
            uvs.push([u_ratio, 1.0 - v_ratio]);
        }
    }

    // Generate indices
    // Each grid cell consists of two triangles
    let stride = width_segments + 1;
    for y in 0..height_segments {
        for x in 0..width_segments {
            let v0 = y * stride + x;
            let v1 = v0 + 1;
            let v2 = (y + 1) * stride + x;
            let v3 = v2 + 1;

            // If not the last row (south pole doesn't need the first triangle, but for simplicity we usually don't do special culling, degenerate triangles will be ignored by the GPU)
            if y != 0 || true { 
                indices.push(v0 as u16);
                indices.push(v1 as u16);
                indices.push(v2 as u16);
            }
            
            // If not the first row (north pole)
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