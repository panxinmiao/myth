//! Extension methods for [`Scene`] that require [`AssetServer`] access.
//!
//! These methods live in `myth_assets` (rather than `myth_scene`) because they
//! bridge the scene graph with the asset system.

use std::sync::Arc;

use myth_animation::mixer::AnimationMixer;
use myth_animation::{AnimationAction, Binder};
use myth_core::{NodeHandle, SkeletonKey};
use myth_resources::geometry::Geometry;
use myth_resources::mesh::Mesh;
use myth_scene::skeleton::{BindMode, Skeleton};
use myth_scene::Scene;

use crate::prefab::Prefab;
use crate::resolve::{ResolveGeometry, ResolveMaterial};
use crate::AssetServer;

/// Extension trait that adds asset-aware helper methods to [`Scene`].
pub trait SceneExt {
    /// Instantiates a [`Prefab`] into the scene, returning the root node handle.
    fn instantiate(&mut self, prefab: &Prefab) -> NodeHandle;

    /// Spawns a mesh node from any geometry/material combination.
    ///
    /// Accepts either pre-registered handles or raw resource structs.
    fn spawn(
        &mut self,
        geometry: impl ResolveGeometry,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle;

    /// Spawns a box mesh node.
    fn spawn_box(
        &mut self,
        w: f32,
        h: f32,
        d: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle;

    /// Spawns a sphere mesh node.
    fn spawn_sphere(
        &mut self,
        radius: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle;

    /// Spawns a plane mesh node.
    fn spawn_plane(
        &mut self,
        width: f32,
        height: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle;
}

impl SceneExt for Scene {
    fn instantiate(&mut self, prefab: &Prefab) -> NodeHandle {
        let node_count = prefab.nodes.len();
        let mut node_map: Vec<NodeHandle> = Vec::with_capacity(node_count);

        // Pass 1: Create all nodes and map indices
        for p_node in &prefab.nodes {
            let handle = self.create_node();

            if let Some(name) = &p_node.name {
                self.set_name(handle, name);
            }

            if let Some(node) = self.get_node_mut(handle) {
                node.transform = p_node.transform;
            }

            if let Some(mesh) = &p_node.mesh {
                self.set_mesh(handle, mesh.clone());
            }

            if let Some(weights) = &p_node.morph_weights {
                self.set_morph_weights(handle, weights.clone());
            }

            if p_node.is_split_primitive {
                self.mark_as_split_primitive(handle);
            }

            node_map.push(handle);
        }

        // Pass 2: Establish hierarchy relationships
        for (i, p_node) in prefab.nodes.iter().enumerate() {
            let parent_handle = node_map[i];
            for &child_idx in &p_node.children_indices {
                if child_idx < node_map.len() {
                    let child_handle = node_map[child_idx];
                    self.attach(child_handle, parent_handle);
                }
            }
        }

        // Pass 3: Rebuild skeletons
        let mut skeleton_keys: Vec<SkeletonKey> = Vec::with_capacity(prefab.skeletons.len());
        for p_skel in &prefab.skeletons {
            let bones: Vec<NodeHandle> = p_skel
                .bone_indices
                .iter()
                .filter_map(|&idx| node_map.get(idx).copied())
                .collect();

            let skeleton = Skeleton::new(
                &p_skel.name,
                bones,
                p_skel.inverse_bind_matrices.clone(),
                p_skel.root_bone_index,
            );
            let skel_key = self.add_skeleton(skeleton);
            skeleton_keys.push(skel_key);
        }

        // Pass 4: Bind skeletons to nodes
        for (i, p_node) in prefab.nodes.iter().enumerate() {
            if let Some(skin_idx) = p_node.skin_index
                && let Some(&skel_key) = skeleton_keys.get(skin_idx)
            {
                let node_handle = node_map[i];
                self.bind_skeleton(node_handle, skel_key, BindMode::Attached);
            }
        }

        // Pass 5: Create virtual root node and mount all top-level nodes
        let root_handle = self.create_node_with_name("gltf_root");
        self.push_root_node(root_handle);

        for &root_idx in &prefab.root_indices {
            if let Some(&node_handle) = node_map.get(root_idx) {
                self.attach(node_handle, root_handle);
            }
        }

        // Pass 6: Create animation mixer and bind animations
        if !prefab.animations.is_empty() {
            let mut mixer = AnimationMixer::new();

            let rig = Binder::build_rig(self, root_handle);

            for clip in &prefab.animations {
                let clip_binding = Binder::build_clip_binding(self, &rig, clip);

                let mut action = AnimationAction::new(Arc::new(clip.clone()));
                action.clip_binding = clip_binding;
                action.enabled = false;
                action.weight = 0.0;

                mixer.add_action(action);
            }

            mixer.set_rig(rig);
            self.animation_mixers.insert(root_handle, mixer);
        }

        // Pass 7: Garbage collection of orphan nodes
        {
            let mut visited = vec![false; node_count];
            let mut stack = prefab.root_indices.clone();

            while let Some(idx) = stack.pop() {
                if visited[idx] {
                    continue;
                }
                visited[idx] = true;
                for &child_idx in &prefab.nodes[idx].children_indices {
                    stack.push(child_idx);
                }
            }

            for (i, &handle) in node_map.iter().enumerate() {
                if !visited[i] {
                    self.remove_node(handle);
                }
            }
        }

        root_handle
    }

    fn spawn(
        &mut self,
        geometry: impl ResolveGeometry,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle {
        let geo_handle = geometry.resolve(assets);
        let mat_handle = material.resolve(assets);
        let mesh = Mesh::new(geo_handle, mat_handle);
        self.add_mesh(mesh)
    }

    fn spawn_box(
        &mut self,
        w: f32,
        h: f32,
        d: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle {
        self.spawn(Geometry::new_box(w, h, d), material, assets)
    }

    fn spawn_sphere(
        &mut self,
        radius: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle {
        self.spawn(Geometry::new_sphere(radius), material, assets)
    }

    fn spawn_plane(
        &mut self,
        width: f32,
        height: f32,
        material: impl ResolveMaterial,
        assets: &AssetServer,
    ) -> NodeHandle {
        self.spawn(Geometry::new_plane(width, height), material, assets)
    }
}
