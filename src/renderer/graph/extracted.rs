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
use rustc_hash::FxHashMap;

use crate::assets::{AssetServer, GeometryHandle, MaterialHandle, ScreenSpaceProfileHandle};
use crate::renderer::core::{BindGroupContext, ResourceManager};
use crate::resources::BoundingBox;
use crate::resources::screen_space::{ScreenSpaceMaterialData, FEATURE_NONE};
use crate::resources::shader_defines::ShaderDefines;
use crate::scene::background::BackgroundMode;
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

    /// World-space axis-aligned bounding box.
    pub world_aabb: BoundingBox,
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
    pub scene_id: u32,
    pub scene_variants: u32,
    pub scene_defines: ShaderDefines,
    pub background: BackgroundMode,
    pub envvironment: Environment,
    pub has_transmission: bool,
    pub lights: Vec<ExtractedLight>,

    // --- Screen-Space Feature Extraction (Thin G-Buffer Hybrid Pipeline) ---

    /// Whether any screen-space features are active this frame.
    /// Cached here so the Prepass and SssssPass can branch without re-reading the Scene.
    pub has_screen_space_features: bool,

    /// Per-frame flat array of `ScreenSpaceMaterialData` for upload to the GPU Storage Buffer.
    ///
    /// - Index 0: default sentinel (no effects) — occupies the ID-0 slot.
    /// - Indices 1–254: profile data for SSS (and future) materials.
    ///
    /// Populated by `extract_screen_space_data()`; uploaded by `SssssPass::prepare()`.
    pub current_screen_space_profiles: Vec<ScreenSpaceMaterialData>,

    /// Dirty flag: set when `current_screen_space_profiles` changed vs. last frame.
    /// `SssssPass::prepare()` only calls `queue.write_buffer` when this is `true`.
    pub screen_space_profiles_changed: bool,

    collected_meshes: Vec<CollectedMesh>,
    collected_skeleton_keys: HashSet<SkeletonKey>,
}

struct CollectedMesh {
    pub node_handle: NodeHandle,
    pub skeleton: Option<SkeletonKey>,

    pub world_matrix: Mat4,
    pub world_aabb: BoundingBox,
    pub item_variant_flags: u32,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
}

impl ExtractedScene {
    /// Creates an empty extracted scene
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            scene_id: 0,
            scene_variants: 0,
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::new(),
            has_screen_space_features: false,
            current_screen_space_profiles: vec![ScreenSpaceMaterialData::default()],
            screen_space_profiles_changed: true,
            collected_meshes: Vec::new(),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// Pre-allocates capacity
    #[must_use]
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            scene_id: 0,
            scene_variants: 0,
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::with_capacity(16),
            has_screen_space_features: false,
            current_screen_space_profiles: vec![ScreenSpaceMaterialData::default()],
            screen_space_profiles_changed: true,
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

        // Reset screen-space extraction state.
        self.has_screen_space_features = false;
        let prev_profile_count = self.current_screen_space_profiles.len();
        self.current_screen_space_profiles.clear();
        // Always keep the sentinel at index 0.
        self.current_screen_space_profiles.push(ScreenSpaceMaterialData::default());
        // Mark unchanged if only the sentinel remains (same as after previous clear).
        // A real change will be detected by comparing count and data below.
        self.screen_space_profiles_changed = prev_profile_count != 1;

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

        let mut variants = 0;
        if self.lights.iter().any(|light| light.cast_shadows) {
            self.scene_defines.set("HAS_SHADOWS", "1");
            variants |= 1 << 0;
        }

        if scene.ssao.enabled {
            self.scene_defines.set("USE_SSAO", "1");
            variants |= 1 << 1;
        }

        if scene.screen_space.enable_sss {
            self.extract_screen_space_data(scene, assets);
        }

        if self.has_screen_space_features {
            self.scene_defines.set("USE_SCREEN_SPACE_FEATURES", "1");
            variants |= 1 << 2;
        }

        self.scene_variants = variants;
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

                let Some(geometry) = geo_guard.map.get(mesh.geometry) else {
                    log::warn!("Node {node_handle:?} refers to missing Geometry");
                    continue;
                };

                // 2. prepare basic data
                let node_world = node.transform.world_matrix;
                let world_matrix = Mat4::from(node_world);
                let skin_binding = scene.skins.get(node_handle);
                let skeleton_key = skin_binding.map(|s| s.skeleton);

                // 3. calculate Flags (pure math calculation)
                let has_negative_scale = world_matrix.determinant() < 0.0;
                let has_negative_scale_flag = u32::from(has_negative_scale);
                let has_skeleton_flag = u32::from(skeleton_key.is_some()) << 1;
                let item_variant_flags = has_negative_scale_flag | has_skeleton_flag;

                // Pre-compute world-space axis-aligned bounding box for frustum culling in Cull phase.
                // Priority: skeleton bounds > geometry AABB > geometry bounding sphere > infinite
                let world_aabb = if let Some(key) = skeleton_key
                    && let Some(skel) = scene.skeleton_pool.get(key)
                    && let Some(local_bounds) = skel.local_bounds()
                {
                    // Skeleton bounds (if available) provide a better fit than static geometry bounds, because they account for animation deformation.
                    local_bounds.transform(&node_world)
                } else {
                    // Static mesh bounding box
                    geometry.bounding_box.transform(&node_world)
                };

                self.collected_meshes.push(CollectedMesh {
                    node_handle,
                    skeleton: skeleton_key,
                    world_matrix,
                    world_aabb,
                    item_variant_flags,
                    cast_shadows: mesh.cast_shadows,
                    receive_shadows: mesh.receive_shadows,
                });

                if let Some(key) = skeleton_key {
                    self.collected_skeleton_keys.insert(key);
                }
            }
        } // release geometry read lock here

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

        for item in &self.collected_meshes {
            // let node_handle = collected_mesh.node_handle;

            let Some(mesh) = scene.meshes.get_mut(item.node_handle) else {
                continue;
            };
            let skeleton = item.skeleton.and_then(|k| scene.skeleton_pool.get(k));

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

            self.render_items.push(ExtractedRenderItem {
                node_handle: item.node_handle,
                world_matrix: item.world_matrix,
                object_bind_group,
                geometry: mesh.geometry,
                material: mesh.material,
                item_variant_flags: item.item_variant_flags,
                item_shader_defines,
                cast_shadows: item.cast_shadows,
                receive_shadows: item.receive_shadows,
                world_aabb: item.world_aabb,
            });
        }
    }

    /// Extract screen-space profile data and write per-frame GPU IDs back to material uniforms.
    ///
    /// Builds the `current_screen_space_profiles` array (uploaded to the StorageBuffer by
    /// `SssssPass::prepare()`) and writes a compact **per-frame u8 GPU ID** into each active
    /// material's `screen_space_id` uniform field.
    ///
    /// # Diff-sync optimisation
    /// Both the profile list and the per-material ID are written **only when they change**,
    /// so steady-state frames with unchanged SSS materials incur no `CpuBuffer` version bump
    /// and no GPU upload cost beyond the StorageBuffer copy.
    ///
    /// # ID encoding
    /// - `0`  → no screen-space effects (sentinel / non-SSS geometry)
    /// - `1–254` → index into `current_screen_space_profiles`
    /// - `255` → reserved (non-SSS geometry writes alpha=1.0 in the prepass)
    fn extract_screen_space_data(&mut self, _scene: &Scene, assets: &AssetServer) {
        let mat_guard = assets.materials.read_lock();
        let profile_guard = assets.screen_space_profiles.read_lock();

        // Per-frame handle → compact u8 ID mapping.  Rebuilt from scratch every frame.
        let mut handle_to_id: FxHashMap<ScreenSpaceProfileHandle, u8> = FxHashMap::default();

        for item in &self.render_items {
            let Some(mat_arc) = mat_guard.map.get(item.material) else {
                continue;
            };
            let Some(phys) = mat_arc.as_physical() else {
                // Non-physical material: no SSS possible, skip silently.
                continue;
            };

            if let Some(profile_handle) = phys.screen_space_profile {
                // --- Assign / look up per-frame GPU ID for this profile ---
                let gpu_id: u8 = match handle_to_id.get(&profile_handle).copied() {
                    Some(existing) => existing,
                    None => {
                        let next_idx = self.current_screen_space_profiles.len();
                        if next_idx >= 255 {
                            log::warn!(
                                "[SSS] Per-frame ScreenSpaceProfile limit (254) exceeded. \
                                 Extra materials fall back to ID 0 (no SSS)."
                            );
                            0
                        } else {
                            let id = next_idx as u8; // 1 ≤ id ≤ 254
                            let gpu_data = profile_guard
                                .map
                                .get(profile_handle)
                                .map_or_else(ScreenSpaceMaterialData::default, |arc| {
                                    arc.to_gpu_data()
                                });
                            self.current_screen_space_profiles.push(gpu_data);
                            handle_to_id.insert(profile_handle, id);
                            id
                        }
                    }
                };

                let feature_flags = if gpu_id > 0 {
                    profile_guard
                        .map
                        .get(profile_handle)
                        .map_or(FEATURE_NONE, |arc| arc.feature_flags)
                } else {
                    FEATURE_NONE
                };

                // Diff-sync: only write if value changed to avoid spurious GPU upload.
                let new_id = u32::from(gpu_id);
                let new_flags = feature_flags;
                let needs_update = {
                    let u = phys.uniforms.read();
                    u.screen_space_id != new_id || u.screen_space_flags != new_flags
                };
                if needs_update {
                    let mut w = phys.uniforms.write();
                    w.screen_space_id = new_id;
                    w.screen_space_flags = new_flags;
                }
            } else {
                // No profile assigned: reset any stale GPU ID from a prior frame.
                let needs_reset = {
                    let u = phys.uniforms.read();
                    u.screen_space_id != 0 || u.screen_space_flags != 0
                };
                if needs_reset {
                    let mut w = phys.uniforms.write();
                    w.screen_space_id = 0;
                    w.screen_space_flags = 0;
                }
            }
        }

        // `len() > 1` means at least one real profile was assigned (not just the sentinel).
        self.has_screen_space_features = self.current_screen_space_profiles.len() > 1;
        // Downstream SssssPass::prepare() uploads the StorageBuffer only when this is true.
        self.screen_space_profiles_changed = self.has_screen_space_features;
    }

    /// Extract environment data
    fn extract_environment(&mut self, scene: &Scene) {
        self.background = scene.background.mode.clone();
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
