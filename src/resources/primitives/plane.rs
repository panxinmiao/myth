use crate::resources::geometry::{Attribute, Geometry};
use wgpu::VertexFormat;

pub struct PlaneOptions {
    pub width: f32,
    pub height: f32,
    pub width_segments: u32,
    pub height_segments: u32,
}

impl Default for PlaneOptions {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            width_segments: 1,
            height_segments: 1,
        }
    }
}

#[must_use]
pub fn create_plane(options: PlaneOptions) -> Geometry {
    let width_half = options.width / 2.0;
    let height_half = options.height / 2.0;

    let grid_x = options.width_segments;
    let grid_y = options.height_segments;

    let grid_x1 = grid_x + 1;
    let grid_y1 = grid_y + 1;

    let segment_width = options.width / grid_x as f32;
    let segment_height = options.height / grid_y as f32;

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for iy in 0..grid_y1 {
        let y = iy as f32 * segment_height - height_half;
        for ix in 0..grid_x1 {
            let x = ix as f32 * segment_width - width_half;

            positions.push([x, -y, 0.0]); // 注意 -y 是为了对应 UV 方向
            normals.push([0.0, 0.0, 1.0]);
            uvs.push([ix as f32 / grid_x as f32, 1.0 - (iy as f32 / grid_y as f32)]);
        }
    }

    // 索引
    for iy in 0..grid_y {
        for ix in 0..grid_x {
            let a = ix + grid_x1 * iy;
            let b = ix + grid_x1 * (iy + 1);
            let c = (ix + 1) + grid_x1 * (iy + 1);
            let d = (ix + 1) + grid_x1 * iy;

            indices.push(a as u16);
            indices.push(b as u16);
            indices.push(d as u16);

            indices.push(b as u16);
            indices.push(c as u16);
            indices.push(d as u16);
        }
    }

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
