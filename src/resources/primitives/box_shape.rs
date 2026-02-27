use crate::resources::geometry::{Attribute, Geometry};
use wgpu::VertexFormat;

#[must_use]
pub fn create_box(width: f32, height: f32, depth: f32) -> Geometry {
    let w = width / 2.0;
    let h = height / 2.0;
    let d = depth / 2.0;

    // 24 vertices (4 per face)
    // Format: [x, y, z]
    let positions = [
        // Front face (+Z)
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d],
        // Back face (-Z)
        [-w, -h, -d],
        [-w, h, -d],
        [w, h, -d],
        [w, -h, -d],
        // Top face (+Y)
        [-w, h, -d],
        [-w, h, d],
        [w, h, d],
        [w, h, -d],
        // Bottom face (-Y)
        [-w, -h, -d],
        [w, -h, -d],
        [w, -h, d],
        [-w, -h, d],
        // Right face (+X)
        [w, -h, -d],
        [w, h, -d],
        [w, h, d],
        [w, -h, d],
        // Left face (-X)
        [-w, -h, -d],
        [-w, -h, d],
        [-w, h, d],
        [-w, h, -d],
    ];

    // Normals (all 4 vertices of each face share the same normal)
    let normals: [[f32; 3]; 24] = [
        // Front
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        // Back
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        // Top
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        // Bottom
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        // Right
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        // Left
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];

    // UV coordinates (standard 0–1 range)
    let uvs: [[f32; 2]; 24] = [
        // Front
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        // Back
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        // Top
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        // Bottom
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        // Right
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        // Left
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];

    // Indices (2 triangles per face, counter-clockwise winding order CCW)
    // 0, 1, 2,  0, 2, 3
    let indices: Vec<u16> = (0..6)
        .flat_map(|face| {
            let base = face * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    let mut geo = Geometry::new();
    geo.set_attribute(
        "position",
        Attribute::new_planar(&positions, VertexFormat::Float32x3),
    );
    geo.set_attribute(
        "normal",
        Attribute::new_planar(&normals, VertexFormat::Float32x3),
    );
    geo.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
    geo.set_indices(&indices);

    geo.compute_bounding_volume();

    geo
}
