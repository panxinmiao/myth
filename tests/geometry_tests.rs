//! Geometry and BoundingBox Tests
//!
//! Tests for:
//! - BoundingBox center, size, union, transform, inflate
//! - Geometry bounding volume computation
//! - Vertex normal computation (area-weighted)
//! - Primitive geometry creation (box, sphere, plane)
//! - Geometry attribute management and versioning
//! - ShaderDefines auto-generation

use glam::{Affine3A, Vec3};
use wgpu::VertexFormat;

use myth::resources::geometry::{Attribute, BoundingBox, Geometry};

const EPSILON: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

fn vec3_approx(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

// ============================================================================
// BoundingBox Tests
// ============================================================================

#[test]
fn bbox_center() {
    let bb = BoundingBox {
        min: Vec3::new(-1.0, -2.0, -3.0),
        max: Vec3::new(1.0, 2.0, 3.0),
    };
    assert!(vec3_approx(bb.center(), Vec3::ZERO));
}

#[test]
fn bbox_size() {
    let bb = BoundingBox {
        min: Vec3::new(0.0, 0.0, 0.0),
        max: Vec3::new(2.0, 4.0, 6.0),
    };
    assert!(vec3_approx(bb.size(), Vec3::new(2.0, 4.0, 6.0)));
}

#[test]
fn bbox_union() {
    let a = BoundingBox {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    };
    let b = BoundingBox {
        min: Vec3::new(0.0, 0.0, 0.0),
        max: Vec3::new(3.0, 3.0, 3.0),
    };
    let u = a.union(&b);
    assert!(vec3_approx(u.min, Vec3::new(-1.0, -1.0, -1.0)));
    assert!(vec3_approx(u.max, Vec3::new(3.0, 3.0, 3.0)));
}

#[test]
fn bbox_transform_identity() {
    let bb = BoundingBox {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    };
    let transformed = bb.transform(&Affine3A::IDENTITY);
    assert!(vec3_approx(transformed.min, bb.min));
    assert!(vec3_approx(transformed.max, bb.max));
}

#[test]
fn bbox_transform_translation() {
    let bb = BoundingBox {
        min: Vec3::new(0.0, 0.0, 0.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    };
    let mat = Affine3A::from_translation(Vec3::new(10.0, 20.0, 30.0));
    let transformed = bb.transform(&mat);
    assert!(vec3_approx(transformed.min, Vec3::new(10.0, 20.0, 30.0)));
    assert!(vec3_approx(transformed.max, Vec3::new(11.0, 21.0, 31.0)));
}

#[test]
fn bbox_transform_scale() {
    let bb = BoundingBox {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    };
    let mat = Affine3A::from_scale(Vec3::splat(2.0));
    let transformed = bb.transform(&mat);
    assert!(vec3_approx(transformed.min, Vec3::splat(-2.0)));
    assert!(vec3_approx(transformed.max, Vec3::splat(2.0)));
}

#[test]
fn bbox_transform_rotation() {
    let bb = BoundingBox {
        min: Vec3::new(0.0, 0.0, 0.0),
        max: Vec3::new(1.0, 0.0, 0.0),
    };
    // 90° rotation around Y: (1,0,0) → (0,0,-1)
    let mat = Affine3A::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let transformed = bb.transform(&mat);
    // After rotation, the AABB should expand to contain rotated corners
    assert!(transformed.min.z <= -0.9);
    assert!(transformed.max.x >= -0.01);
}

#[test]
fn bbox_infinite() {
    let bb = BoundingBox::infinite();
    assert!(!bb.is_finite());
}

#[test]
fn bbox_finite() {
    let bb = BoundingBox {
        min: Vec3::ZERO,
        max: Vec3::ONE,
    };
    assert!(bb.is_finite());
}

// ============================================================================
// Geometry Primitive Tests
// ============================================================================

#[test]
fn geometry_box_has_correct_attributes() {
    let geom = Geometry::new_box(2.0, 3.0, 4.0);
    assert!(geom.get_attribute("position").is_some());
    assert!(geom.get_attribute("normal").is_some());
    assert!(geom.get_attribute("uv").is_some());
    assert!(geom.index_attribute().is_some());
}

#[test]
fn geometry_box_bounding_volume() {
    let geom = Geometry::new_box(2.0, 4.0, 6.0);
    // Half-extents should be 1, 2, 3
    assert!(approx(geom.bounding_box.min.x, -1.0));
    assert!(approx(geom.bounding_box.max.x, 1.0));
    assert!(approx(geom.bounding_box.min.y, -2.0));
    assert!(approx(geom.bounding_box.max.y, 2.0));
    assert!(approx(geom.bounding_box.min.z, -3.0));
    assert!(approx(geom.bounding_box.max.z, 3.0));
}

#[test]
fn geometry_sphere_has_position_normal_uv() {
    let geom = Geometry::new_sphere(1.0);
    assert!(geom.get_attribute("position").is_some());
    assert!(geom.get_attribute("normal").is_some());
    assert!(geom.get_attribute("uv").is_some());
}

#[test]
fn geometry_sphere_bounding_sphere_radius() {
    let geom = Geometry::new_sphere(5.0);
    // Bounding sphere radius should be approximately the sphere radius
    assert!(
        (geom.bounding_sphere.radius - 5.0).abs() < 0.1,
        "Expected radius ≈ 5.0, got {}",
        geom.bounding_sphere.radius
    );
}

#[test]
fn geometry_plane_has_attributes() {
    let geom = Geometry::new_plane(10.0, 10.0);
    assert!(geom.get_attribute("position").is_some());
    assert!(geom.get_attribute("normal").is_some());
    assert!(geom.get_attribute("uv").is_some());
}

// ============================================================================
// Geometry Attribute Management Tests
// ============================================================================

#[test]
fn geometry_set_attribute_increments_version() {
    let mut geom = Geometry::new();
    let initial_layout = geom.layout_version();
    let initial_data = geom.data_version();

    let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    let attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);
    geom.set_attribute("position", attr);

    assert!(
        geom.layout_version() > initial_layout,
        "Layout version should increment on new attribute"
    );
    assert!(
        geom.data_version() > initial_data,
        "Data version should increment on set_attribute"
    );
}

#[test]
fn geometry_remove_attribute() {
    let mut geom = Geometry::new();
    let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    let attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);
    geom.set_attribute("position", attr);

    assert!(geom.get_attribute("position").is_some());
    let removed = geom.remove_attribute("position");
    assert!(removed.is_some());
    assert!(geom.get_attribute("position").is_none());
}

#[test]
fn geometry_shader_defines_generated_from_attributes() {
    let mut geom = Geometry::new();

    let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    geom.set_attribute(
        "position",
        Attribute::new_planar(&positions, VertexFormat::Float32x3),
    );
    geom.set_attribute(
        "normal",
        Attribute::new_planar(&positions, VertexFormat::Float32x3),
    );

    let defines = geom.shader_defines();
    assert!(
        defines.contains("HAS_POSITION"),
        "Should have HAS_POSITION define"
    );
    assert!(
        defines.contains("HAS_NORMAL"),
        "Should have HAS_NORMAL define"
    );
}

// ============================================================================
// Vertex Normal Computation Tests
// ============================================================================

#[test]
fn compute_normals_single_triangle_facing_z() {
    let mut geom = Geometry::new();

    // Triangle in XY plane → normal should be +Z or -Z
    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    geom.set_attribute(
        "position",
        Attribute::new_planar(&positions, VertexFormat::Float32x3),
    );

    geom.compute_vertex_normals();

    let normal_attr = geom.get_attribute("normal").expect("Should have normals");
    for i in 0..3 {
        let n = normal_attr.read_vec3(i).unwrap();
        // Normal should point along Z axis (CCW winding → +Z)
        assert!(
            n.z.abs() > 0.9,
            "Normal {i} should be approximately ±Z, got {:?}",
            n
        );
    }
}

#[test]
fn compute_normals_indexed_geometry() {
    let mut geom = Geometry::new();

    // A quad as 2 indexed triangles
    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    geom.set_attribute(
        "position",
        Attribute::new_planar(&positions, VertexFormat::Float32x3),
    );
    geom.set_indices(&[0u16, 1, 2, 0, 2, 3]);

    geom.compute_vertex_normals();

    let normal_attr = geom.get_attribute("normal").expect("Should have normals");
    for i in 0..4 {
        let n = normal_attr.read_vec3(i).unwrap();
        // All normals should point along +Z (quad in XY plane)
        assert!(n.z > 0.9, "Normal {i} should point +Z, got {:?}", n);
    }
}

// ============================================================================
// Attribute Data Read/Write Tests
// ============================================================================

#[test]
fn attribute_read_vec3() {
    let positions = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
    let attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);

    let v0 = attr.read_vec3(0).unwrap();
    assert!(vec3_approx(v0, Vec3::new(1.0, 2.0, 3.0)));

    let v1 = attr.read_vec3(1).unwrap();
    assert!(vec3_approx(v1, Vec3::new(4.0, 5.0, 6.0)));
}

#[test]
fn attribute_update_data() {
    let positions = vec![Vec3::ZERO, Vec3::X];
    let mut attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);

    let new_positions = vec![Vec3::new(10.0, 20.0, 30.0), Vec3::new(40.0, 50.0, 60.0)];
    attr.update_data(&new_positions);

    let v0 = attr.read_vec3(0).unwrap();
    assert!(vec3_approx(v0, Vec3::new(10.0, 20.0, 30.0)));
}
