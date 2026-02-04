//! Render Extract Phase
//!
//! Before rendering begins, extract minimal data needed for the current frame from the Scene.
//! After extraction is complete, the Scene can be released and subsequent render preparation doesn't depend on Scene's borrow.
//!
//! # Design Principles
//! - Only copy "minimal data" needed for rendering, don't copy actual Mesh/Material resources
//! - Front-load frustum culling, only extract visible objects
//! - Use Copy types to minimize overhead
//! - Carry cache IDs to avoid repeated lookups each frame

use std::collections::HashSet;

use glam::Mat4;

use crate::renderer::core::{BindGroupContext, ResourceManager};
use crate::resources::shader_defines::ShaderDefines;
use crate::scene::environment::Environment;
use crate::scene::{NodeHandle, Scene, SkeletonKey};
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::scene::camera::RenderCamera;

/// Minimal render item, containing only data needed by GPU
/// 
/// Uses Clone instead of Copy because SkinBinding contains non-Copy types
#[derive(Clone)]
pub struct ExtractedRenderItem {
    /// Node handle (for debugging and cache write-back)
    pub node_handle: NodeHandle,
    /// World transform matrix (64 bytes)
    pub world_matrix: Mat4,

    pub object_bind_group: BindGroupContext,
    /// Geometry handle (8 bytes)
    pub geometry: GeometryHandle,
    /// Material handle (8 bytes)
    pub material: MaterialHandle,

    pub item_variant_flags: u32,

    pub item_shader_defines: ShaderDefines,

    /// Squared distance to camera (for sorting)
    pub distance_sq: f32,

}

/// Extracted skeleton data
#[derive(Clone)]
pub struct ExtractedSkeleton {
    pub skeleton_key: SkeletonKey,
}

/// Extracted scene data
/// 
/// This is a lightweight structure containing only the minimal dataset needed for current frame rendering.
/// Populated during Extract phase, after which the Scene borrow can be safely released.
pub struct ExtractedScene {
    /// List of visible render items (already frustum culled)
    pub render_items: Vec<ExtractedRenderItem>,
    /// Scene's shader macro definitions
    pub scene_defines: ShaderDefines,
    pub scene_id: u32,
    pub background: Option<glam::Vec4>,
    pub envvironment: Environment,
    pub has_transmission: bool,

    collected_meshes: Vec<CollectedMesh>,
    collected_skeleton_keys: HashSet<SkeletonKey>,

}

struct CollectedMesh {
    pub node_handle: NodeHandle,
    pub skeleton: Option<SkeletonKey>,
}

impl ExtractedScene {
    /// Creates an empty extracted scene
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            has_transmission: false,
            collected_meshes: Vec::new(),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// Pre-allocates capacity
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            has_transmission: false,
            collected_meshes: Vec::with_capacity(item_capacity),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// Clear data for reuse
    pub fn clear(&mut self) {
        self.render_items.clear();
        // self.skeletons.clear();
        self.scene_defines.clear();
        self.scene_id = 0;


        self.collected_meshes.clear();
        self.collected_skeleton_keys.clear();
    }

    /// Reuse current instance memory, extract data from Scene
    pub fn extract_into(&mut self, scene: &mut Scene, camera: &RenderCamera, assets: &AssetServer, resource_manager: &mut ResourceManager) {
        self.clear();
        self.extract_render_items(scene, camera, assets , resource_manager);
        self.extract_environment(scene);
    }

    /// Extract visible render items
    fn extract_render_items(&mut self, scene: &mut Scene, camera: &RenderCamera, assets: &AssetServer, resource_manager: &mut ResourceManager) {
        let frustum = camera.frustum;
        let camera_pos = camera.position;

        // =========================================================
        // Phase 1: Frustum culling & collection (holding read lock)
        // =========================================================
        {

            let geo_guard = assets.geometries.read_lock();

            for (node_handle, mesh) in scene.meshes.iter() {
                if !mesh.visible {
                    continue;
                }

                let Some(node) = scene.nodes.get(node_handle) else {
                    continue;
                };

                if !node.visible {
                    continue;
                }

                let geo_handle = mesh.geometry;
                // let mat_handle = mesh.material;

                let Some(geometry) = geo_guard.map.get(geo_handle) else {
                    log::warn!("Node {:?} refers to missing Geometry {:?}", node_handle, geo_handle);
                    continue;
                };

                let node_world = node.transform.world_matrix;
                let skin_binding = scene.skins.get(node_handle);

                // Frustum culling: select different bounding box based on skeleton binding
                let is_visible = if let Some(binding) = skin_binding {
                    // Has skeleton binding: use Skeleton's bounding box
                    if let Some(skeleton) = scene.skeleton_pool.get(binding.skeleton) {
                        if let Some(local_bounds) = skeleton.local_bounds() {
                            let world_bounds = local_bounds.transform(&node_world);
                            frustum.intersects_box(world_bounds.min, world_bounds.max)
                        } else {
                            // Bounding box not yet computed, default to visible
                            true
                        }
                    } else {
                        true
                    }
                } else {
                    // No skeleton binding: use Geometry's bounding box
                    if let Some(bbox) = geometry.bounding_box.read().as_ref() {
                        let world_bounds = bbox.transform(&node_world);
                        frustum.intersects_box(world_bounds.min, world_bounds.max)
                    } else if let Some(bs) = geometry.bounding_sphere.read().as_ref() {
                        // Fallback to bounding sphere
                        let scale = node.transform.scale.max_element();
                        let center = node_world.transform_point3(bs.center);
                        frustum.intersects_sphere(center, bs.radius * scale)
                    } else {
                        true
                    }
                };

                if !is_visible {
                    continue;
                }

                self.collected_meshes.push(CollectedMesh {
                    node_handle,
                    skeleton: skin_binding.map(|skin| skin.skeleton),
                });

                if let Some(binding) = skin_binding {
                    self.collected_skeleton_keys.insert(binding.skeleton);
                }

            }

        }

        // =========================================================
        // Phase 2: Prepare resources (no lock held)
        // =========================================================

        // Prepare skeleton data
        for skeleton_key in &self.collected_skeleton_keys {
            if let Some(skeleton) = scene.skeleton_pool.get(*skeleton_key){
                resource_manager.prepare_skeleton(skeleton);
            }
        }

        // Ensure model buffer capacity
        resource_manager.ensure_model_buffer_capacity(self.collected_meshes.len());

        // Update and populate render items
        for collected_mesh in &self.collected_meshes {
            let node_handle = collected_mesh.node_handle;

            let node = if let Some(n) = scene.nodes.get(node_handle) {
                n
            } else {
                continue;
            };

            let mesh = if let Some(m) = scene.meshes.get_mut(node_handle) {
                m
            } else {
                continue;
            };
            
            let node_world = node.transform.world_matrix;
            let world_matrix = Mat4::from(node_world);


            let skeleton = collected_mesh.skeleton.and_then(|key| scene.skeleton_pool.get(key).map(|s| s));
            mesh.update_morph_uniforms();
            
            let object_bind_group = if let Some(binding) = resource_manager.prepare_mesh(assets, mesh, skeleton) {
                binding
            } else {
                continue;
            };


            let distance_sq = camera_pos.distance_squared(node_world.translation);
            let mut item_shader_defines = ShaderDefines::with_capacity(1);

            if skeleton.is_some() {
                item_shader_defines.set("HAS_SKINNING", "1");
            }

            let has_negative_scale = world_matrix.determinant() < 0.0;
            let has_negative_scale_flag = (has_negative_scale as u32) << 0;

            let has_skeleton = skeleton.is_some();
            let has_skeleton_flag = (has_skeleton as u32) << 1;

            // compose item variant flags (has_negative_scale, has_skeleton)
            let item_variant_flags = has_negative_scale_flag | has_skeleton_flag;

            self.render_items.push(ExtractedRenderItem {
                node_handle,
                world_matrix,
                object_bind_group,
                geometry: mesh.geometry,
                material: mesh.material,
                item_variant_flags: item_variant_flags,
                item_shader_defines,
                distance_sq,
            });
        }
    }



    /// Extract environment data
    fn extract_environment(&mut self, scene: &Scene) {
        // self.background = scene.background;
        self.scene_defines = scene.shader_defines();
        self.scene_id = scene.id;
        self.envvironment = scene.environment.clone();
    }

    /// Get render item count
    #[inline]
    pub fn render_item_count(&self) -> usize {
        self.render_items.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.render_items.is_empty()
    }
}

impl Default for ExtractedScene {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_render_item_size() {
        // Ensure struct size is reasonable
        let size = std::mem::size_of::<ExtractedRenderItem>();
        println!("ExtractedRenderItem size: {} bytes", size);
        // Should be within reasonable range (not exceeding 256 bytes)
        assert!(size < 256, "ExtractedRenderItem is too large: {} bytes", size);
    }
}
