//! Render Extract Phase
//!
//! Before rendering begins, extract minimal data needed for the current frame from the Scene.
//! After extraction is complete, the Scene can be released and subsequent render preparation doesn't depend on Scene's borrow.
//!
//! # Design Principles
//! - Only copy "minimal data" needed for rendering, don't copy actual Mesh/Material resources
//! - Extract ALL active meshes without frustum culling (culling is deferred to the Cull phase)
//! - Use Copy types to minimize overhead
//! - Carry cache IDs to avoid repeated lookups each frame
//! - Single source of truth: one `render_items` list consumed by all `RenderView`s

use std::collections::HashSet;

use glam::{Mat4, Vec3};

use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::renderer::core::{BindGroupContext, ResourceManager};
use crate::resources::shader_defines::ShaderDefines;
use crate::scene::camera::RenderCamera;
use crate::scene::environment::Environment;
use crate::scene::light::{LightKind, ShadowConfig};
use crate::scene::{NodeHandle, Scene, SkeletonKey};

/// Minimal render item, containing only data needed by GPU
///
/// Uses Clone instead of Copy because `SkinBinding` contains non-Copy types.
/// Contains all per-object attributes needed for view-independent filtering and culling.
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

    pub cast_shadows: bool,
    pub receive_shadows: bool,

    /// World-space bounding sphere (center, radius).
    /// Pre-computed during Extract for efficient frustum culling in the Cull phase.
    /// If no bounding volume is available, `radius` is `f32::INFINITY` (always passes culling).
    pub world_bounding_sphere: (Vec3, f32),
}

#[derive(Clone)]
pub struct ExtractedLight {
    pub id: u64,
    pub cast_shadows: bool,
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    pub shadow: Option<ShadowConfig>,
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
///
/// # "Single Source of Truth" Design
///
/// `render_items` is the **single, unified list** of all active renderables.
/// No frustum culling is performed during extraction — that is deferred to the
/// Cull phase where each `RenderView` (main camera, shadow cascades, etc.)
/// performs its own culling against this shared list.
pub struct ExtractedScene {
    /// All active render items (NOT frustum-culled).
    /// Each `RenderView` in the Cull phase filters and culls from this list.
    pub render_items: Vec<ExtractedRenderItem>,
    /// Scene's shader macro definitions
    pub scene_defines: ShaderDefines,
    pub scene_id: u32,
    pub background: Option<glam::Vec4>,
    pub envvironment: Environment,
    pub has_transmission: bool,
    pub lights: Vec<ExtractedLight>,

    collected_meshes: Vec<CollectedMesh>,
    collected_skeleton_keys: HashSet<SkeletonKey>,
}

struct CollectedMesh {
    pub node_handle: NodeHandle,
    pub skeleton: Option<SkeletonKey>,
}

impl ExtractedScene {
    /// Creates an empty extracted scene
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::new(),
            collected_meshes: Vec::new(),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// Pre-allocates capacity
    #[must_use]
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::with_capacity(16),
            collected_meshes: Vec::with_capacity(item_capacity),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// Clear data for reuse
    pub fn clear(&mut self) {
        self.render_items.clear();
        self.scene_defines.clear();
        self.scene_id = 0;
        self.lights.clear();

        self.collected_meshes.clear();
        self.collected_skeleton_keys.clear();
    }

    /// Reuse current instance memory, extract data from Scene.
    ///
    /// Extracts ALL active meshes into `render_items` without frustum culling.
    /// Frustum culling is deferred to the Cull phase where each `RenderView`
    /// independently culls from this unified list.
    pub fn extract_into(
        &mut self,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
    ) {
        self.clear();
        self.extract_lights(scene);
        self.extract_render_items(scene, camera, assets, resource_manager);
        self.extract_environment(scene);

        if self.lights.iter().any(|light| light.cast_shadows) {
            self.scene_defines.set("USE_SHADOWS", "1");
        }
    }

    fn extract_lights(&mut self, scene: &Scene) {
        self.lights.reserve(scene.lights.len());

        for (light, world_matrix) in scene.iter_active_lights() {
            let position = world_matrix.translation.to_vec3();
            let direction = world_matrix
                .transform_vector3(-glam::Vec3::Z)
                .normalize_or_zero();

            self.lights.push(ExtractedLight {
                id: light.id,
                cast_shadows: light.cast_shadows,
                kind: light.kind.clone(),
                position,
                direction,
                shadow: light.shadow.clone(),
            });
        }
    }

    /// Extract all active render items (no frustum culling).
    ///
    /// Only performs lightweight validity checks:
    /// - `mesh.visible` and `node.visible` flags
    /// - Geometry asset exists
    ///
    /// World-space bounding spheres are pre-computed here so the Cull phase
    /// can test each item against multiple `RenderView` frustums without
    /// re-acquiring the geometry read lock.
    #[allow(clippy::too_many_lines)]
    fn extract_render_items(
        &mut self,
        scene: &mut Scene,
        _camera: &RenderCamera,
        assets: &AssetServer,
        resource_manager: &mut ResourceManager,
    ) {
        // =========================================================
        // Phase 1: Collect active meshes (holding read lock)
        // =========================================================
        {
            let geo_guard = assets.geometries.read_lock();

            for (node_handle, mesh) in &scene.meshes {
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

                if !geo_guard.map.contains_key(geo_handle) {
                    log::warn!("Node {node_handle:?} refers to missing Geometry {geo_handle:?}");
                    continue;
                }

                let skin_binding = scene.skins.get(node_handle);

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
        // Phase 2: Prepare resources & build render items (no lock)
        // =========================================================

        // Prepare skeleton data
        for skeleton_key in &self.collected_skeleton_keys {
            if let Some(skeleton) = scene.skeleton_pool.get(*skeleton_key) {
                resource_manager.prepare_skeleton(skeleton);
            }
        }

        // Ensure model buffer capacity
        resource_manager.ensure_model_buffer_capacity(self.collected_meshes.len());

        // Pre-compute world bounding spheres (requires geo lock)
        let geo_guard = assets.geometries.read_lock();

        for collected_mesh in &self.collected_meshes {
            let node_handle = collected_mesh.node_handle;

            let Some(node) = scene.nodes.get(node_handle) else {
                continue;
            };

            let Some(mesh) = scene.meshes.get_mut(node_handle) else {
                continue;
            };

            let node_world = node.transform.world_matrix;
            let world_matrix = Mat4::from(node_world);

            let skeleton = collected_mesh
                .skeleton
                .and_then(|key| scene.skeleton_pool.get(key));
            mesh.update_morph_uniforms();

            let Some(object_bind_group) = resource_manager.prepare_mesh(assets, mesh, skeleton)
            else {
                continue;
            };

            let mut item_shader_defines = ShaderDefines::with_capacity(1);

            if skeleton.is_some() {
                item_shader_defines.set("HAS_SKINNING", "1");
            }
            if mesh.receive_shadows {
                item_shader_defines.set("RECEIVE_SHADOWS", "1");
            }

            let has_negative_scale = world_matrix.determinant() < 0.0;
            let has_negative_scale_flag = u32::from(has_negative_scale);

            let has_skeleton = skeleton.is_some();
            let has_skeleton_flag = u32::from(has_skeleton) << 1;

            // compose item variant flags (has_negative_scale, has_skeleton)
            let item_variant_flags = has_negative_scale_flag | has_skeleton_flag;

            // Pre-compute world-space bounding sphere for frustum culling in Cull phase.
            // Priority: skeleton bounds > geometry AABB > geometry bounding sphere > infinite
            let world_bounding_sphere = if let Some(binding) = scene.skins.get(node_handle) {
                if let Some(skel) = scene.skeleton_pool.get(binding.skeleton) {
                    if let Some(local_bounds) = skel.local_bounds() {
                        let world_bounds = local_bounds.transform(&node_world);
                        let center = (world_bounds.min + world_bounds.max) * 0.5;
                        let radius = (world_bounds.max - world_bounds.min).length() * 0.5;
                        (center, radius)
                    } else {
                        // Skeleton bounds not yet computed, treat as always visible
                        (node_world.translation.into(), f32::INFINITY)
                    }
                } else {
                    (node_world.translation.into(), f32::INFINITY)
                }
            } else if let Some(geometry) = geo_guard.map.get(mesh.geometry) {
                let bbox = geometry.bounding_box;
                let world_bounds = bbox.transform(&node_world);
                let center = (world_bounds.min + world_bounds.max) * 0.5;
                let radius = (world_bounds.max - world_bounds.min).length() * 0.5;
                (center, radius)
            } else {
                #[cfg(debug_assertions)]
                log::warn!(
                    "Geometry {:?} has zero bounds! Did you forget to set position?",
                    mesh.geometry
                );

                (node_world.translation.into(), f32::INFINITY)
            };

            self.render_items.push(ExtractedRenderItem {
                node_handle,
                world_matrix,
                object_bind_group,
                geometry: mesh.geometry,
                material: mesh.material,
                item_variant_flags,
                item_shader_defines,
                cast_shadows: mesh.cast_shadows,
                receive_shadows: mesh.receive_shadows,
                world_bounding_sphere,
            });
        }
    }

    /// Extract environment data
    fn extract_environment(&mut self, scene: &Scene) {
        // self.background = scene.background;
        self.scene_defines = scene.shader_defines.clone();
        self.scene_id = scene.id;
        self.envvironment = scene.environment.clone();
    }

    /// Get render item count
    #[inline]
    #[must_use]
    pub fn render_item_count(&self) -> usize {
        self.render_items.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
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
        println!("ExtractedRenderItem size: {size} bytes");
        // Should be within reasonable range (not exceeding 256 bytes)
        assert!(size < 256, "ExtractedRenderItem is too large: {size} bytes");
    }
}
