use crate::resources::geometry::{Geometry, Attribute};
use wgpu::VertexFormat;

pub fn create_box(width: f32, height: f32, depth: f32) -> Geometry {
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
    
    geo.compute_bounding_volume();

    geo
}