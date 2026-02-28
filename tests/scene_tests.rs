//! Scene Integration Tests
//!
//! Tests for:
//! - Scene: create/remove nodes, attach/detach hierarchy
//! - Component management: set/get mesh, camera, light, morph weights
//! - Node query: names, root_nodes, subtree collection
//! - SceneNode wrapper convenience API

use glam::Vec3;
use myth::assets::AssetServer;
use myth::scene::camera::Camera;
use myth::scene::light::{Light, LightKind};
use myth::scene::node::Node;
use myth::scene::scene::Scene;

fn new_scene() -> Scene {
    Scene::new(AssetServer::new())
}

// ============================================================================
// Node Creation & Removal
// ============================================================================

#[test]
fn scene_create_node() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    assert!(scene.get_node(handle).is_some());
}

#[test]
fn scene_create_node_with_name() {
    let mut scene = new_scene();
    let handle = scene.create_node_with_name("TestNode");
    assert_eq!(scene.get_name(handle), Some("TestNode"));
}

#[test]
fn scene_set_name() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_name(handle, "Renamed");
    assert_eq!(scene.get_name(handle), Some("Renamed"));
}

#[test]
fn scene_add_node_to_root() {
    let mut scene = new_scene();
    let handle = scene.add_node(Node::new());
    assert!(scene.root_nodes.contains(&handle));
}

#[test]
fn scene_remove_node_removes_from_root() {
    let mut scene = new_scene();
    let handle = scene.add_node(Node::new());
    assert!(scene.root_nodes.contains(&handle));

    scene.remove_node(handle);
    assert!(!scene.root_nodes.contains(&handle));
    assert!(scene.get_node(handle).is_none());
}

#[test]
fn scene_remove_node_removes_subtree() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let child = scene.create_node();
    let grandchild = scene.create_node();

    scene.attach(child, parent);
    scene.attach(grandchild, child);

    scene.remove_node(parent);

    assert!(scene.get_node(parent).is_none());
    assert!(scene.get_node(child).is_none());
    assert!(scene.get_node(grandchild).is_none());
}

// ============================================================================
// Hierarchy: Attach / Detach
// ============================================================================

#[test]
fn scene_attach_sets_parent_child() {
    let mut scene = new_scene();
    let parent = scene.create_node();
    let child = scene.create_node();

    scene.attach(child, parent);

    assert_eq!(scene.get_node(child).unwrap().parent, Some(parent));
    assert!(scene.get_node(parent).unwrap().children.contains(&child));
}

#[test]
fn scene_attach_removes_from_old_parent() {
    let mut scene = new_scene();
    let parent1 = scene.create_node();
    let parent2 = scene.create_node();
    let child = scene.create_node();

    scene.attach(child, parent1);
    assert!(scene.get_node(parent1).unwrap().children.contains(&child));

    // Re-attach to parent2
    scene.attach(child, parent2);
    assert!(
        !scene.get_node(parent1).unwrap().children.contains(&child),
        "Child should be removed from old parent"
    );
    assert!(
        scene.get_node(parent2).unwrap().children.contains(&child),
        "Child should be in new parent"
    );
}

#[test]
fn scene_attach_to_self_is_noop() {
    let mut scene = new_scene();
    let node = scene.create_node();

    // attach to self should not crash
    scene.attach(node, node);

    assert_eq!(scene.get_node(node).unwrap().parent, None);
}

#[test]
fn scene_add_to_parent() {
    let mut scene = new_scene();
    let parent = scene.add_node(Node::new());
    let child = scene.add_to_parent(Node::new(), parent);

    assert_eq!(scene.get_node(child).unwrap().parent, Some(parent));
    assert!(scene.get_node(parent).unwrap().children.contains(&child));
}

// ============================================================================
// Component Management: Camera, Light
// ============================================================================

#[test]
fn scene_set_get_camera() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    let camera = Camera::new_perspective(60.0, 16.0 / 9.0, 0.1);
    scene.set_camera(handle, camera);

    assert!(scene.get_camera(handle).is_some());
}

#[test]
fn scene_active_camera() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_camera(handle, Camera::new_perspective(60.0, 1.0, 0.1));
    scene.active_camera = Some(handle);

    assert_eq!(scene.active_camera, Some(handle));
}

#[test]
fn scene_set_get_light() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    let light = Light::new_directional(Vec3::ONE, 1.0);
    scene.set_light(handle, light);

    let l = scene.get_light(handle).unwrap();
    assert!(matches!(l.kind, LightKind::Directional(_)));
}

#[test]
fn scene_set_morph_weights() {
    let mut scene = new_scene();
    let handle = scene.create_node();
    scene.set_morph_weights(handle, vec![0.5, 0.3, 0.1]);

    let weights = scene.get_morph_weights(handle).unwrap();
    assert_eq!(weights.len(), 3);
    assert!((weights[0] - 0.5).abs() < 1e-5);
}

// ============================================================================
// Hierarchy + Transform Dirty Propagation
// ============================================================================

#[test]
fn scene_attach_marks_child_dirty() {
    let mut scene = new_scene();
    let parent = scene.create_node();
    let child = scene.create_node();

    // Consume dirty flag by calling update_local_matrix
    scene
        .get_node_mut(child)
        .unwrap()
        .transform
        .update_local_matrix();

    scene.attach(child, parent);

    // attach should mark child transform as dirty, so update_local_matrix returns true
    let child_node = scene.get_node_mut(child).unwrap();
    assert!(
        child_node.transform.update_local_matrix(),
        "Attach should mark child transform dirty"
    );
}

// ============================================================================
// Multiple Nodes & Iteration
// ============================================================================

#[test]
fn scene_iterate_active_lights() {
    let mut scene = new_scene();
    let h1 = scene.create_node();
    scene.set_light(h1, Light::new_directional(Vec3::ONE, 1.0));

    let h2 = scene.create_node();
    scene.set_light(h2, Light::new_point(Vec3::ONE, 1.0, 10.0));

    // Make h2 invisible
    scene.get_node_mut(h2).unwrap().visible = false;

    let active_lights: Vec<_> = scene.iter_active_lights().collect();
    assert_eq!(
        active_lights.len(),
        1,
        "Only visible lights should be iterated"
    );
}

#[test]
fn scene_unique_ids() {
    let s1 = new_scene();
    let s2 = new_scene();
    assert_ne!(s1.id, s2.id, "Each scene should have a unique ID");
}
