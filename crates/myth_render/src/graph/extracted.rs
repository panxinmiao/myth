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

use bitflags::{Flags, bitflags};

use crate::core::{BindGroupContext, ResourceManager};
use myth_assets::{AssetServer, GeometryHandle, MaterialHandle};
use myth_resources::BoundingBox;
use myth_resources::buffer::CpuBuffer;
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage, LightBufferMetadata};
use myth_scene::background::BackgroundMode;
use myth_scene::camera::RenderCamera;
use myth_scene::environment::Environment;
use myth_scene::light::{LightKind, ShadowConfig, SpotLight};
use myth_scene::{NodeHandle, Scene, SkeletonKey};

const SPOT_TIGHT_SPHERE_COS_THRESHOLD: f32 = std::f32::consts::FRAC_1_SQRT_2; // ~ 0.70710678, corresponds to a 90-degree outer cone angle
const LIGHT_SCORE_DISTANCE_FLOOR: f32 = 1.0;

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

    /// Previous frame's world transform matrix (64 bytes)
    pub prev_world_matrix: Mat4,

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
    pub color: Vec3,
    pub intensity: f32,
    pub flags: u32,
    pub cast_shadows: bool,
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    pub shadow: Option<ShadowConfig>,
}

impl ExtractedLight {
    #[inline]
    #[must_use]
    pub fn is_directional(&self) -> bool {
        matches!(self.kind, LightKind::Directional(_))
    }

    #[inline]
    #[must_use]
    pub fn local_range(&self) -> f32 {
        match &self.kind {
            LightKind::Point(point) => point.range,
            LightKind::Spot(spot) => spot.range,
            LightKind::Directional(_) => 0.0,
        }
    }
}

/// Extracted skeleton data
#[derive(Clone)]
pub struct ExtractedSkeleton {
    pub skeleton_key: SkeletonKey,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SceneFeatures: u32 {
        const HAS_SHADOWS = 1 << 0;
        const USE_SSAO = 1 << 1;
        const USE_SSS = 1 << 2;
        const USE_SSR = 1 << 3;
        const USE_CLUSTERED_SHADING = 1 << 4;
        const USE_SSGI = 1 << 5;


        const USE_SCREEN_SPACE_FEATURES = Self::USE_SSS.bits() | Self::USE_SSR.bits();
    }
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
    pub scene_variants: SceneFeatures,
    pub scene_defines: ShaderDefines,
    pub background: BackgroundMode,
    pub envvironment: Environment,
    pub has_transmission: bool,
    pub lights: Vec<ExtractedLight>,
    /// Number of local point / spot lights occupying the head segment of `lights`.
    local_light_count: usize,
    /// GPU-visible directional-light payload for the current frame.
    pub directional_light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    /// GPU-visible local-light payload for the current frame.
    pub local_light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    /// GPU-visible local-light metadata used by RDG scene-light bindings.
    pub local_light_metadata_buffer: CpuBuffer<LightBufferMetadata>,
    /// GPU-visible environment uniforms for the current frame.
    pub environment_uniforms_buffer: CpuBuffer<EnvironmentUniforms>,

    collected_meshes: Vec<CollectedMesh>,
    collected_skeleton_keys: HashSet<SkeletonKey>,
    local_light_upload_cache: Vec<GpuLightStorage>,
    directional_light_upload_cache: Vec<GpuLightStorage>,
}

struct CollectedMesh {
    pub node_handle: NodeHandle,
    pub skeleton: Option<SkeletonKey>,

    pub world_matrix: Mat4,
    pub prev_world_matrix: Mat4,
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
            scene_variants: SceneFeatures::empty(),
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::new(),
            local_light_count: 0,
            directional_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneDirectionalLightStorageBuffer"),
            ),
            local_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default()],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightStorageBuffer"),
            ),
            local_light_metadata_buffer: CpuBuffer::new(
                LightBufferMetadata::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightMetadata"),
            ),
            environment_uniforms_buffer: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneEnvironmentUniforms"),
            ),

            collected_meshes: Vec::new(),
            collected_skeleton_keys: HashSet::default(),
            local_light_upload_cache: Vec::with_capacity(16),
            directional_light_upload_cache: Vec::with_capacity(4),
        }
    }

    /// Pre-allocates capacity
    #[must_use]
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            scene_id: 0,
            scene_variants: SceneFeatures::empty(),
            scene_defines: ShaderDefines::new(),
            background: BackgroundMode::default(),
            envvironment: Environment::default(),
            has_transmission: false,
            lights: Vec::with_capacity(16),
            local_light_count: 0,
            directional_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default(); 4],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneDirectionalLightStorageBuffer"),
            ),
            local_light_storage_buffer: CpuBuffer::new(
                vec![GpuLightStorage::default(); 16],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightStorageBuffer"),
            ),
            local_light_metadata_buffer: CpuBuffer::new(
                LightBufferMetadata::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneLocalLightMetadata"),
            ),
            environment_uniforms_buffer: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ExtractedSceneEnvironmentUniforms"),
            ),

            collected_meshes: Vec::with_capacity(item_capacity),
            collected_skeleton_keys: HashSet::default(),
            local_light_upload_cache: Vec::with_capacity(16),
            directional_light_upload_cache: Vec::with_capacity(4),
        }
    }

    /// Clear data for reuse
    pub fn clear(&mut self) {
        self.render_items.clear();
        self.scene_defines.clear();
        self.scene_id = 0;
        self.lights.clear();
        self.local_light_count = 0;

        self.collected_meshes.clear();
        self.collected_skeleton_keys.clear();
    }

    #[must_use]
    pub fn local_light_count(&self) -> usize {
        self.local_light_count
    }

    #[must_use]
    pub fn directional_light_count(&self) -> usize {
        self.lights.len().saturating_sub(self.local_light_count)
    }

    #[must_use]
    pub fn local_lights(&self) -> &[ExtractedLight] {
        &self.lights[..self.local_light_count]
    }

    #[must_use]
    pub fn directional_lights(&self) -> &[ExtractedLight] {
        &self.lights[self.local_light_count..]
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
        max_light_count: usize,
    ) {
        self.clear();
        self.extract_lights(scene, camera, max_light_count);
        self.extract_render_items(scene, camera, assets, resource_manager);
        self.extract_environment(scene);

        self.scene_variants.clear();

        if self.lights.iter().any(|light| light.cast_shadows) {
            self.scene_defines.set("HAS_SHADOWS", "1");
            self.scene_variants.insert(SceneFeatures::HAS_SHADOWS);
        }

        if scene.ssao.enabled {
            self.scene_defines.set("USE_SSAO", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSAO);
        }

        if scene.ssgi.enabled {
            self.scene_defines.set("USE_SSGI", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSGI);
        }

        if scene.ssss.enabled {
            self.scene_defines.set("USE_SCREEN_SPACE_FEATURES", "1");
            self.scene_defines.set("USE_SSS", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSS);
        }

        if scene.ssr.enabled {
            self.scene_defines.set("USE_SCREEN_SPACE_FEATURES", "1");
            self.scene_defines.set("USE_SSR", "1");
            self.scene_variants.insert(SceneFeatures::USE_SSR);
        }

        if matches!(scene.background.mode, BackgroundMode::Procedural(_)) {
            self.scene_defines.set("USE_ATMOSPHERE_TRANSMITTANCE", "1");
        }

        // Material-override debug view — inject shader defines so the PBR
        // fragment shader short-circuits lighting and outputs raw attributes.
        #[cfg(feature = "debug_view")]
        {
            use myth_scene::camera::DebugViewMode;
            match camera.debug_view.mode {
                DebugViewMode::Albedo => self.scene_defines.set("DEBUG_VIEW_ALBEDO", "1"),
                DebugViewMode::Roughness => self.scene_defines.set("DEBUG_VIEW_ROUGHNESS", "1"),
                DebugViewMode::Metalness => self.scene_defines.set("DEBUG_VIEW_METALNESS", "1"),
                _ => {}
            }
        }
    }

    /// Refresh the extracted scene's GPU-facing light and environment buffers.
    pub fn prepare_gpu_scene_inputs(&mut self, env_map_max_mip_level: f32) {
        self.sync_directional_light_storage_buffer();
        self.sync_local_light_storage_buffer();
        self.sync_local_light_metadata_buffer();
        self.sync_environment_uniforms(env_map_max_mip_level);
    }

    fn extract_lights(&mut self, scene: &Scene, camera: &RenderCamera, max_light_count: usize) {
        self.lights.reserve(scene.lights.len());

        for (light, world_matrix) in scene.iter_active_lights() {
            let position = world_matrix.translation.to_vec3();
            let direction = world_matrix
                .transform_vector3(-glam::Vec3::Z)
                .normalize_or_zero();

            let extracted_light = ExtractedLight {
                id: light.id(),
                color: light.color,
                intensity: light.intensity,
                flags: light.flags,
                cast_shadows: light.cast_shadows,
                kind: light.kind.clone(),
                position,
                direction,
                shadow: light.shadow.clone(),
            };

            if !light_is_visible_for_camera(&extracted_light, camera) {
                continue;
            }

            self.lights.push(extracted_light);
        }

        let visible_light_count = self.lights.len();
        let camera_position = camera.position.to_array().into();
        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut self.lights, camera_position, max_light_count);
        if truncated_count > 0 {
            log::warn!(
                "Extracted light list truncated from {} to {} entries to satisfy GPU storage buffer limits",
                visible_light_count,
                self.lights.len(),
            );
        }

        self.local_light_count = local_light_count;
    }

    fn sync_directional_light_storage_buffer(&mut self) {
        let mut upload_cache = std::mem::take(&mut self.directional_light_upload_cache);
        upload_cache.clear();
        upload_cache.reserve(self.directional_light_count().max(1));

        for light in self.directional_lights() {
            upload_cache.push(extracted_light_to_gpu(light));
        }

        if upload_cache.is_empty() {
            upload_cache.push(GpuLightStorage::default());
        }

        let needs_update =
            self.directional_light_storage_buffer.read().as_slice() != upload_cache.as_slice();
        if needs_update {
            self.directional_light_storage_buffer
                .write()
                .clone_from(&upload_cache);
        }

        self.directional_light_upload_cache = upload_cache;
    }

    fn sync_local_light_storage_buffer(&mut self) {
        let mut upload_cache = std::mem::take(&mut self.local_light_upload_cache);
        upload_cache.clear();
        upload_cache.reserve(self.local_light_count.max(1));

        for light in self.local_lights() {
            upload_cache.push(extracted_light_to_gpu(light));
        }

        if upload_cache.is_empty() {
            upload_cache.push(GpuLightStorage::default());
        }

        let needs_update =
            self.local_light_storage_buffer.read().as_slice() != upload_cache.as_slice();
        if needs_update {
            self.local_light_storage_buffer
                .write()
                .clone_from(&upload_cache);
        }

        self.local_light_upload_cache = upload_cache;
    }

    fn sync_local_light_metadata_buffer(&mut self) {
        let new_metadata = LightBufferMetadata {
            total_light_count: self.local_light_count as u32,
            active_local_light_count: self.local_light_count as u32,
            ..Default::default()
        };

        let needs_update = *self.local_light_metadata_buffer.read() != new_metadata;
        if needs_update {
            *self.local_light_metadata_buffer.write() = new_metadata;
        }
    }

    fn sync_environment_uniforms(&mut self, env_map_max_mip_level: f32) {
        let env = &self.envvironment;
        let new_uniforms = EnvironmentUniforms {
            ambient_light: env.ambient,
            directional_light_count: self.directional_light_count() as u32,
            env_map_intensity: env.intensity,
            env_map_rotation: env.rotation,
            env_map_max_mip_level,
            ..Default::default()
        };

        let needs_update = *self.environment_uniforms_buffer.read() != new_uniforms;
        if needs_update {
            *self.environment_uniforms_buffer.write() = new_uniforms;
        }
    }

    #[must_use]
    pub fn has_shadow_casters(&self) -> bool {
        self.scene_variants.contains(SceneFeatures::HAS_SHADOWS)
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

                let Some(geometry) = geo_guard.get_loaded(mesh.geometry) else {
                    continue;
                };

                // 2. prepare basic data
                let node_world = node.transform.world_matrix;
                let world_matrix = Mat4::from(node_world);
                let prev_world_matrix = Mat4::from(node.transform.previous_world_matrix);
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
                    prev_world_matrix,
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
        // resource_manager.ensure_model_buffer_capacity(self.collected_meshes.len());

        for item in &self.collected_meshes {
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
                prev_world_matrix: item.prev_world_matrix,
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

    /// Extract environment data
    fn extract_environment(&mut self, scene: &Scene) {
        self.background = scene.background.mode.clone();
        self.scene_defines = scene.shader_defines().clone();
        self.scene_id = scene.id();
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

fn light_is_visible_for_camera(light: &ExtractedLight, camera: &RenderCamera) -> bool {
    match &light.kind {
        LightKind::Directional(_) => true,
        LightKind::Point(point) => {
            point.range > 0.0
                && camera
                    .frustum
                    .intersects_sphere(light.position, point.range)
        }
        LightKind::Spot(spot) => {
            if spot.range <= 0.0 {
                return false;
            }

            let (sphere_center, sphere_radius) =
                spot_bounding_sphere(light.position, light.direction, spot);
            camera
                .frustum
                .intersects_sphere(sphere_center, sphere_radius)
        }
    }
}

fn extracted_light_to_gpu(light: &ExtractedLight) -> GpuLightStorage {
    let mut gpu_light = GpuLightStorage {
        color: light.color,
        intensity: light.intensity,
        position: light.position,
        direction: light.direction,
        flags: light.flags,
        shadow_layer_index: -1,
        ..Default::default()
    };

    match &light.kind {
        LightKind::Directional(_) => {
            gpu_light.light_type = 0;
        }
        LightKind::Point(point) => {
            gpu_light.light_type = 1;
            gpu_light.range = point.range;
        }
        LightKind::Spot(spot) => {
            gpu_light.light_type = 2;
            gpu_light.range = spot.range;
            gpu_light.inner_cone_cos = spot.inner_cone.cos();
            gpu_light.outer_cone_cos = spot.outer_cone.cos();
        }
    }

    gpu_light
}

fn extracted_light_score(light: &ExtractedLight, camera_position: Vec3) -> f32 {
    let radius = extracted_light_radius(light);
    if radius <= 0.0 || light.intensity <= 0.0 {
        return 0.0;
    }

    let distance = (light.position - camera_position)
        .length()
        .max(LIGHT_SCORE_DISTANCE_FLOOR);
    let score = light.intensity * (radius / distance);
    if score.is_finite() { score } else { 0.0 }
}

fn sort_and_truncate_lights(
    lights: &mut Vec<ExtractedLight>,
    camera_position: Vec3,
    max_light_count: usize,
) -> (usize, usize) {
    lights.sort_unstable_by(
        |lhs, rhs| match (lhs.is_directional(), rhs.is_directional()) {
            (false, false) => {
                let lhs_score = extracted_light_score(lhs, camera_position);
                let rhs_score = extracted_light_score(rhs, camera_position);
                rhs_score
                    .total_cmp(&lhs_score)
                    .then_with(|| lhs.id.cmp(&rhs.id))
            }
            (true, true) => lhs.id.cmp(&rhs.id),
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
        },
    );

    let visible_light_count = lights.len();
    let mut local_light_count = lights.partition_point(|light| !light.is_directional());
    let directional_light_count = visible_light_count - local_light_count;
    if visible_light_count > max_light_count {
        if directional_light_count >= max_light_count {
            lights.drain(..local_light_count);
            lights.truncate(max_light_count);
            local_light_count = 0;
        } else {
            let retained_local_count = max_light_count - directional_light_count;
            lights.drain(retained_local_count..local_light_count);
            local_light_count = retained_local_count;
        }
    }

    let truncated_count = visible_light_count.saturating_sub(lights.len());
    (local_light_count, truncated_count)
}

fn extracted_light_radius(light: &ExtractedLight) -> f32 {
    match &light.kind {
        LightKind::Directional(_) => 0.0,
        LightKind::Point(point) => point.range,
        LightKind::Spot(spot) => spot_bounding_sphere(light.position, light.direction, spot).1,
    }
}

fn spot_bounding_sphere(position: Vec3, direction: Vec3, spot: &SpotLight) -> (Vec3, f32) {
    let outer_cone_cos = spot.outer_cone.cos();
    if outer_cone_cos < SPOT_TIGHT_SPHERE_COS_THRESHOLD {
        return (position, spot.range);
    }

    let cos_sq = (outer_cone_cos * outer_cone_cos).max(1e-4);
    let radius = 0.5 * spot.range / cos_sq;
    (position + direction * radius, radius)
}

#[cfg(test)]
mod tests {
    use super::{sort_and_truncate_lights, spot_bounding_sphere};
    use glam::Vec3;
    use myth_scene::light::{DirectionalLight, LightKind, PointLight, SpotLight};

    use crate::graph::extracted::ExtractedLight;

    fn directional_light(id: u64) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity: 1.0,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Directional(DirectionalLight {}),
            position: Vec3::ZERO,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    fn point_light(id: u64, position: Vec3, intensity: f32, range: f32) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Point(PointLight { range }),
            position,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    fn spot_light(id: u64, position: Vec3, intensity: f32, range: f32) -> ExtractedLight {
        ExtractedLight {
            id,
            color: Vec3::ONE,
            intensity,
            flags: 0,
            cast_shadows: false,
            kind: LightKind::Spot(SpotLight {
                range,
                inner_cone: 0.25,
                outer_cone: 0.5,
            }),
            position,
            direction: -Vec3::Z,
            shadow: None,
        }
    }

    #[test]
    fn partition_sort_and_truncate_keeps_directional_lights_in_tail_segment() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -4.0), 4.0, 6.0),
            directional_light(2),
            spot_light(3, Vec3::new(0.0, 0.0, -2.0), 2.0, 5.0),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, usize::MAX);

        assert_eq!(local_light_count, 2);
        assert_eq!(truncated_count, 0);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }

    #[test]
    fn partition_sort_and_truncate_preserves_directional_lights_when_capping_locals() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -12.0), 2.0, 3.0),
            point_light(2, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            spot_light(3, Vec3::new(0.0, 0.0, -5.0), 3.0, 9.0),
            directional_light(4),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 2);

        assert_eq!(local_light_count, 1);
        assert_eq!(truncated_count, 2);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![2, 4]
        );
    }

    #[test]
    fn sort_and_truncate_reuses_the_input_allocation() {
        let mut lights = Vec::with_capacity(16);
        lights.extend([
            directional_light(9),
            point_light(2, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            directional_light(4),
            point_light(1, Vec3::new(0.0, 0.0, -12.0), 2.0, 3.0),
        ]);
        let allocation = lights.as_ptr();
        let capacity = lights.capacity();

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 3);

        assert_eq!(lights.as_ptr(), allocation);
        assert_eq!(lights.capacity(), capacity);
        assert_eq!(local_light_count, 1);
        assert_eq!(truncated_count, 1);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![2, 4, 9]
        );
    }

    #[test]
    fn sort_and_truncate_prefers_lowest_directional_ids_at_tight_capacity() {
        let mut lights = vec![
            point_light(1, Vec3::new(0.0, 0.0, -3.0), 4.0, 10.0),
            directional_light(9),
            directional_light(4),
            directional_light(7),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, 2);

        assert_eq!(local_light_count, 0);
        assert_eq!(truncated_count, 2);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![4, 7]
        );
    }

    #[test]
    fn sort_and_truncate_preserves_local_id_tie_breaking() {
        let mut lights = vec![
            point_light(9, Vec3::new(0.0, 0.0, -5.0), 2.0, 10.0),
            point_light(4, Vec3::new(0.0, 0.0, -5.0), 2.0, 10.0),
            directional_light(7),
        ];

        let (local_light_count, truncated_count) =
            sort_and_truncate_lights(&mut lights, Vec3::ZERO, usize::MAX);

        assert_eq!(local_light_count, 2);
        assert_eq!(truncated_count, 0);
        assert_eq!(
            lights.iter().map(|light| light.id).collect::<Vec<_>>(),
            vec![4, 9, 7]
        );
    }

    #[test]
    fn narrow_spot_uses_tighter_bounding_sphere() {
        let spot = SpotLight {
            range: 12.0,
            inner_cone: 0.1,
            outer_cone: 0.3,
        };

        let (center, radius) = spot_bounding_sphere(Vec3::ZERO, -Vec3::Z, &spot);

        assert!(radius > spot.range * 0.5);
        assert!(center.z < 0.0);
    }
}
