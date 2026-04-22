use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};

use myth_animation::{AnimationMixer, AnimationTarget};
use myth_core::{NodeHandle, SkeletonKey, Transform};
#[cfg(feature = "3dgs")]
use myth_resources::GaussianCloudHandle;
use myth_resources::Input;
use myth_resources::bloom::BloomSettings;
use myth_resources::buffer::CpuBuffer;
use myth_resources::mesh::Mesh;
use myth_resources::screen_space::ScreenSpaceSettings;
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::ssao::SsaoSettings;
use myth_resources::tone_mapping::ToneMappingSettings;
use myth_resources::uniforms::{EnvironmentUniforms, GpuLightStorage};

use crate::background::{BackgroundMode, BackgroundSettings};
use crate::camera::Camera;
use crate::environment::Environment;
use crate::light::Light;
use crate::light::LightKind;
use crate::node::Node;
use crate::skeleton::{BindMode, Skeleton, SkinBinding};
use crate::transform_system;
use crate::wrapper::SceneNode;
use glam::{Affine3A, Quat, Vec3};
use slotmap::{SecondaryMap, SlotMap, SparseSecondaryMap};

static NEXT_SCENE_ID: AtomicU32 = AtomicU32::new(1);

/// Trait for scene update logic.
///
/// Allows users to define custom behavior scripts that update
/// along with the scene lifecycle each frame.
///
/// # Example
///
/// ```rust,ignore
/// struct RotateScript {
///     target: NodeHandle,
///     speed: f32,
/// }
///
/// impl SceneLogic for RotateScript {
///     fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32) {
///         if let Some(node) = scene.get_node_mut(self.target) {
///             node.transform.rotation *= Quat::from_rotation_y(self.speed * dt);
///         }
///     }
/// }
/// ```
pub trait SceneLogic: Send + Sync + 'static {
    /// Called each frame to update scene state.
    fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32);
}

/// Syntactic sugar: allows using closures directly as scene logic.
pub struct CallbackLogic<F>(pub F);
impl<F> SceneLogic for CallbackLogic<F>
where
    F: FnMut(&mut Scene, &Input, f32) + Send + Sync + 'static,
{
    fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32) {
        (self.0)(scene, input, dt);
    }
}

/// Tag component indicating a split primitive node.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitPrimitiveTag;

/// The scene graph container.
///
/// Scene is the pure data layer that stores scene graph hierarchy and component data.
/// Uses `SlotMap` + `SecondaryMap` for high-performance component-based storage.
///
/// # Storage Layout
///
/// - `nodes`: Core node data (hierarchy and transforms) using `SlotMap`
/// - Dense components (names, meshes): Use `SecondaryMap`
/// - Sparse components (cameras, lights, skins): Use `SparseSecondaryMap`
///
/// # Example
///
/// ```rust,ignore
/// let mut scene = Scene::new();
///
/// // Create nodes
/// let root = scene.create_node_with_name("Root");
/// let child = scene.create_node_with_name("Child");
/// scene.attach(child, root);
///
/// // Add mesh component
/// scene.set_mesh(child, Mesh::new(geometry, material));
/// ```
pub struct Scene {
    /// Unique scene identifier (assigned automatically, read-only)
    id: u32,

    // === Core Node Storage ===
    /// All nodes in the scene (`SlotMap` for O(1) access)
    #[doc(hidden)]
    pub nodes: SlotMap<NodeHandle, Node>,
    /// Root-level nodes (no parent)
    root_nodes: Vec<NodeHandle>,

    // === Dense Components (most nodes have these) ===
    /// Node names - almost all nodes have a name
    pub names: SecondaryMap<NodeHandle, Cow<'static, str>>,

    // === Sparse Components (only some nodes have these) ===
    /// Mesh components stored directly on nodes
    pub meshes: SparseSecondaryMap<NodeHandle, Mesh>,
    /// Camera components stored directly on nodes
    pub cameras: SparseSecondaryMap<NodeHandle, Camera>,
    /// Light components stored directly on nodes
    pub lights: SparseSecondaryMap<NodeHandle, Light>,
    /// Skeletal skin bindings
    pub skins: SparseSecondaryMap<NodeHandle, SkinBinding>,
    /// Morph target weights
    pub morph_weights: SparseSecondaryMap<NodeHandle, Vec<f32>>,
    /// Animation mixer components (sparse, only character roots have animations)
    pub animation_mixers: SparseSecondaryMap<NodeHandle, AnimationMixer>,
    /// Rest pose transforms recorded before animation takes over.
    /// Used to restore nodes when animations stop or blend with weight < 1.0.
    pub rest_transforms: SparseSecondaryMap<NodeHandle, Transform>,
    /// Split primitive tags
    pub split_primitive_tags: SparseSecondaryMap<NodeHandle, SplitPrimitiveTag>,
    #[cfg(feature = "3dgs")]
    /// Gaussian splatting point cloud handles attached to nodes.
    pub gaussian_clouds: SparseSecondaryMap<NodeHandle, GaussianCloudHandle>,

    // === Resource Pools (only truly shared resources) ===
    /// Skeleton is a shared resource - multiple characters may reference the same skeleton definition
    pub skeleton_pool: SlotMap<SkeletonKey, Skeleton>,

    // === Environment and Global Settings ===
    /// Scene environment settings (skybox, IBL)
    pub environment: Environment,
    /// Tone mapping settings (exposure, mode)
    pub tone_mapping: ToneMappingSettings,
    /// Bloom post-processing settings
    pub bloom: BloomSettings,
    /// SSAO (Screen Space Ambient Occlusion) settings
    pub ssao: SsaoSettings,
    /// Screen space effects settings (SSS, SSR)
    pub screen_space: ScreenSpaceSettings,
    /// Background rendering settings (mode + skybox uniform buffer)
    pub background: BackgroundSettings,
    /// Currently active camera for rendering
    pub active_camera: Option<NodeHandle>,

    // === GPU Resource Descriptors ===
    #[doc(hidden)]
    pub light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    #[doc(hidden)]
    pub uniforms_buffer: CpuBuffer<EnvironmentUniforms>,
    light_data_cache: Vec<GpuLightStorage>,

    shader_defines: ShaderDefines,

    last_env_version: u64,

    // === Scene Logic System ===
    pub(crate) logics: Vec<Box<dyn SceneLogic>>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),

            nodes: SlotMap::with_key(),
            root_nodes: Vec::new(),

            // Dense components
            names: SecondaryMap::new(),

            // Sparse components (direct storage)
            meshes: SparseSecondaryMap::new(),
            cameras: SparseSecondaryMap::new(),
            lights: SparseSecondaryMap::new(),
            skins: SparseSecondaryMap::new(),
            morph_weights: SparseSecondaryMap::new(),
            animation_mixers: SparseSecondaryMap::new(),
            rest_transforms: SparseSecondaryMap::new(),

            split_primitive_tags: SparseSecondaryMap::new(),
            #[cfg(feature = "3dgs")]
            gaussian_clouds: SparseSecondaryMap::new(),

            // Resource pools (only truly shared resources)
            skeleton_pool: SlotMap::with_key(),

            environment: Environment::new(),
            tone_mapping: ToneMappingSettings::default(),
            bloom: BloomSettings::default(),
            ssao: SsaoSettings::default(),
            screen_space: ScreenSpaceSettings::default(),
            background: BackgroundSettings::default(),

            active_camera: None,

            light_storage_buffer: CpuBuffer::new(
                [GpuLightStorage::default(); 16].to_vec(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("SceneLightStorageBuffer"),
            ),
            uniforms_buffer: CpuBuffer::new(
                EnvironmentUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("SceneEnvironmentUniforms"),
            ),

            light_data_cache: Vec::with_capacity(16),

            shader_defines: ShaderDefines::default(),
            last_env_version: 0,

            logics: Vec::new(),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the unique scene identifier.
    #[inline]
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns a read-only slice of root-level node handles.
    #[inline]
    #[must_use]
    pub fn root_nodes(&self) -> &[NodeHandle] {
        &self.root_nodes
    }

    /// Registers a node as a root-level node (no parent).
    pub fn push_root_node(&mut self, handle: NodeHandle) {
        self.root_nodes.push(handle);
    }

    /// Returns a read-only reference to the node storage.
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &SlotMap<NodeHandle, Node> {
        &self.nodes
    }

    // ========================================================================
    // Node Management API
    // ========================================================================

    /// Creates a new node and returns its handle.
    pub fn create_node(&mut self) -> NodeHandle {
        self.nodes.insert(Node::new())
    }

    /// Creates a new node with a name.
    pub fn create_node_with_name(&mut self, name: &str) -> NodeHandle {
        let handle = self.nodes.insert(Node::new());
        self.names.insert(handle, Cow::Owned(name.to_string()));
        handle
    }

    /// Adds a node to the scene (defaults to root level).
    pub fn add_node(&mut self, node: Node) -> NodeHandle {
        let handle = self.nodes.insert(node);
        self.root_nodes.push(handle);
        handle
    }

    /// Adds a node as a child of the specified parent.
    pub fn add_to_parent(&mut self, child: Node, parent_handle: NodeHandle) -> NodeHandle {
        let handle = self.nodes.insert(child);

        // Establish parent-child relationship
        if let Some(parent) = self.nodes.get_mut(parent_handle) {
            parent.children.push(handle);
        }
        if let Some(child_node) = self.nodes.get_mut(handle) {
            child_node.parent = Some(parent_handle);
        }

        handle
    }

    /// Removes a node and all its descendants recursively.
    pub fn remove_node(&mut self, handle: NodeHandle) {
        // 1. Collect all nodes to remove (depth-first)
        let mut to_remove = Vec::new();
        self.collect_subtree(handle, &mut to_remove);

        // 2. Handle parent relationship
        if let Some(node) = self.nodes.get(handle) {
            if let Some(parent_handle) = node.parent {
                if let Some(parent) = self.nodes.get_mut(parent_handle) {
                    parent.children.retain(|&h| h != handle);
                }
            } else {
                self.root_nodes.retain(|&h| h != handle);
            }
        }

        // 3. Remove all nodes and their components
        for node_handle in to_remove {
            self.meshes.remove(node_handle);
            self.cameras.remove(node_handle);
            self.lights.remove(node_handle);
            self.skins.remove(node_handle);
            self.morph_weights.remove(node_handle);
            self.names.remove(node_handle);
            self.animation_mixers.remove(node_handle);
            self.rest_transforms.remove(node_handle);

            self.nodes.remove(node_handle);
        }
    }

    /// Collects all nodes in a subtree (depth-first).
    fn collect_subtree(&self, handle: NodeHandle, result: &mut Vec<NodeHandle>) {
        result.push(handle);
        if let Some(node) = self.nodes.get(handle) {
            for &child in &node.children {
                self.collect_subtree(child, result);
            }
        }
    }

    /// Attaches a node as a child of another (establishes parent-child relationship).
    pub fn attach(&mut self, child_handle: NodeHandle, parent_handle: NodeHandle) {
        if child_handle == parent_handle {
            log::warn!("Cannot attach node to itself!");
            return;
        }

        // 1. Detach from old parent
        if let Some(child_node) = self.nodes.get(child_handle) {
            if let Some(old_parent) = child_node.parent {
                if let Some(parent) = self.nodes.get_mut(old_parent) {
                    parent.children.retain(|&h| h != child_handle);
                }
            } else {
                self.root_nodes.retain(|&h| h != child_handle);
            }
        }

        // 2. Attach to new parent
        if let Some(parent) = self.nodes.get_mut(parent_handle) {
            parent.children.push(child_handle);
        } else {
            log::error!("Parent node not found during attach!");
            self.root_nodes.push(child_handle);
            return;
        }

        // 3. Update child
        if let Some(child) = self.nodes.get_mut(child_handle) {
            child.parent = Some(parent_handle);
            child.transform.mark_dirty();
        }
    }

    /// Returns a read-only reference to a node.
    #[inline]
    pub fn get_node(&self, handle: NodeHandle) -> Option<&Node> {
        self.nodes.get(handle)
    }

    /// Returns a mutable reference to a node.
    #[inline]
    pub fn get_node_mut(&mut self, handle: NodeHandle) -> Option<&mut Node> {
        self.nodes.get_mut(handle)
    }

    // ========================================================================
    // Component Management API (ECS-style)
    // ========================================================================

    /// Sets the name for a node.
    pub fn set_name(&mut self, handle: NodeHandle, name: &str) {
        self.names.insert(handle, Cow::Owned(name.to_string()));
    }

    /// Returns the name of a node.
    pub fn get_name(&self, handle: NodeHandle) -> Option<&str> {
        self.names.get(handle).map(std::convert::AsRef::as_ref)
    }

    /// Sets the mesh component for a node.
    pub fn set_mesh(&mut self, handle: NodeHandle, mesh: Mesh) {
        self.meshes.insert(handle, mesh);
    }

    /// Gets a reference to the node's Mesh component
    pub fn get_mesh(&self, handle: NodeHandle) -> Option<&Mesh> {
        self.meshes.get(handle)
    }

    /// Gets a mutable reference to the node's Mesh component
    pub fn get_mesh_mut(&mut self, handle: NodeHandle) -> Option<&mut Mesh> {
        self.meshes.get_mut(handle)
    }

    #[cfg(feature = "3dgs")]
    /// Attaches a Gaussian splatting point cloud handle to a node.
    pub fn set_gaussian_cloud(&mut self, handle: NodeHandle, cloud: GaussianCloudHandle) {
        self.gaussian_clouds.insert(handle, cloud);
    }

    #[cfg(feature = "3dgs")]
    /// Gets the Gaussian cloud handle attached to a node.
    pub fn get_gaussian_cloud(&self, handle: NodeHandle) -> Option<GaussianCloudHandle> {
        self.gaussian_clouds.get(handle).copied()
    }

    #[cfg(feature = "3dgs")]
    /// Creates a named node with a Gaussian splatting point cloud and adds it as a root.
    pub fn add_gaussian_cloud(
        &mut self,
        name: &str,
        cloud_handle: GaussianCloudHandle,
    ) -> NodeHandle {
        let handle = self.create_node_with_name(name);
        self.gaussian_clouds.insert(handle, cloud_handle);
        self.root_nodes.push(handle);
        handle
    }

    #[cfg(feature = "3dgs")]
    /// Returns `true` if the scene contains any Gaussian splatting clouds.
    #[inline]
    pub fn has_gaussian_clouds(&self) -> bool {
        !self.gaussian_clouds.is_empty()
    }

    /// Sets the Camera component for a node
    pub fn set_camera(&mut self, handle: NodeHandle, camera: Camera) {
        self.cameras.insert(handle, camera);
    }

    /// Gets a reference to the node's Camera component
    pub fn get_camera(&self, handle: NodeHandle) -> Option<&Camera> {
        self.cameras.get(handle)
    }

    /// Gets a mutable reference to the node's Camera component
    pub fn get_camera_mut(&mut self, handle: NodeHandle) -> Option<&mut Camera> {
        self.cameras.get_mut(handle)
    }

    /// Sets the Light component for a node
    pub fn set_light(&mut self, handle: NodeHandle, light: Light) {
        self.lights.insert(handle, light);
    }

    /// Gets a reference to the node's Light component
    pub fn get_light(&self, handle: NodeHandle) -> Option<&Light> {
        self.lights.get(handle)
    }

    /// Gets a mutable reference to the node's Light component
    pub fn get_light_mut(&mut self, handle: NodeHandle) -> Option<&mut Light> {
        self.lights.get_mut(handle)
    }

    /// Gets both the Light component and Transform for a node (for light processing)
    pub fn get_light_bundle(&mut self, handle: NodeHandle) -> Option<(&mut Light, &mut Node)> {
        let light = self.lights.get_mut(handle)?;
        let node = self.nodes.get_mut(handle)?;
        Some((light, node))
    }

    /// Binds a skeleton to a node
    pub fn bind_skeleton(
        &mut self,
        handle: NodeHandle,
        skeleton_key: SkeletonKey,
        bind_mode: BindMode,
    ) {
        if let Some(node) = self.nodes.get(handle) {
            let bind_matrix_inv = node.transform.world_matrix.inverse();
            self.skins.insert(
                handle,
                SkinBinding {
                    skeleton: skeleton_key,
                    bind_mode,
                    bind_matrix_inv,
                },
            );
        }
    }

    /// Gets the node's skin binding
    pub fn get_skin(&self, handle: NodeHandle) -> Option<&SkinBinding> {
        self.skins.get(handle)
    }

    /// Sets morph weights
    pub fn set_morph_weights(&mut self, handle: NodeHandle, weights: Vec<f32>) {
        self.morph_weights.insert(handle, weights);
    }

    /// Gets morph weights
    pub fn get_morph_weights(&self, handle: NodeHandle) -> Option<&Vec<f32>> {
        self.morph_weights.get(handle)
    }

    /// Gets a mutable reference to morph weights
    pub fn get_morph_weights_mut(&mut self, handle: NodeHandle) -> Option<&mut Vec<f32>> {
        self.morph_weights.get_mut(handle)
    }

    /// Sets morph weights for a node (from POD data)
    pub fn set_morph_weights_from_pod(
        &mut self,
        handle: NodeHandle,
        data: &myth_animation::values::MorphWeightData,
    ) {
        let weights = self.morph_weights.entry(handle).unwrap().or_default();

        if weights.len() != data.weights.len() {
            weights.resize(data.weights.len(), 0.0);
        }
        weights.copy_from_slice(&data.weights);
    }

    // ========================================================================
    // Iterate over all active lights in the scene
    // ========================================================================

    pub fn iter_active_lights(&self) -> impl Iterator<Item = (&Light, &Affine3A)> {
        self.lights.iter().filter_map(move |(node_handle, light)| {
            let node = self.nodes.get(node_handle)?;
            if node.visible {
                Some((light, &node.transform.world_matrix))
            } else {
                None
            }
        })
    }

    // ========================================================================
    // Component Query API
    // ========================================================================

    /// Gets the (Transform, Camera) bundle for the main camera
    pub fn query_main_camera_bundle(&mut self) -> Option<(&mut Transform, &mut Camera)> {
        let node_handle = self.active_camera?;
        self.query_camera_bundle(node_handle)
    }

    pub fn query_camera_bundle(
        &mut self,
        node_handle: NodeHandle,
    ) -> Option<(&mut Transform, &mut Camera)> {
        // Check if camera component exists
        if !self.cameras.contains_key(node_handle) {
            return None;
        }

        // Use pointers to avoid simultaneous borrow conflict between nodes and cameras
        let transform_ptr = self
            .nodes
            .get_mut(node_handle)
            .map(|n| &raw mut n.transform)?;
        let camera = self.cameras.get_mut(node_handle)?;

        // SAFETY: transform and camera are disjoint memory regions
        unsafe { Some((&mut *transform_ptr, camera)) }
    }

    /// Queries the Transform and Light for a specified node
    pub fn query_light_bundle(
        &mut self,
        node_handle: NodeHandle,
    ) -> Option<(&mut Transform, &Light)> {
        let light = self.lights.get(node_handle)?;
        let transform = &mut self.nodes.get_mut(node_handle)?.transform;
        Some((transform, light))
    }

    /// Queries the Transform and Mesh for a specified node
    pub fn query_mesh_bundle(
        &mut self,
        node_handle: NodeHandle,
    ) -> Option<(&mut Transform, &Mesh)> {
        let mesh = self.meshes.get(node_handle)?;
        let transform = &mut self.nodes.get_mut(node_handle)?.transform;
        Some((transform, mesh))
    }

    // ========================================================================
    // Matrix Update Pipeline
    // ========================================================================

    /// Updates world matrices for the entire scene
    pub fn update_matrix_world(&mut self) {
        transform_system::update_hierarchy_iterative(
            &mut self.nodes,
            &mut self.cameras,
            &self.root_nodes,
        );
    }

    /// Updates world matrices for a specified subtree
    pub fn update_subtree(&mut self, root_handle: NodeHandle) {
        transform_system::update_subtree(&mut self.nodes, &mut self.cameras, root_handle);
    }

    // ========================================================================
    // Resource Management API
    // ========================================================================

    pub fn add_mesh(&mut self, mesh: Mesh) -> NodeHandle {
        let node_handle = self.create_node_with_name(&mesh.name);
        self.meshes.insert(node_handle, mesh);
        self.root_nodes.push(node_handle);
        node_handle
    }

    pub fn add_mesh_to_parent(&mut self, mesh: Mesh, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name(&mesh.name);
        self.meshes.insert(node_handle, mesh);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> SkeletonKey {
        self.skeleton_pool.insert(skeleton)
    }

    pub fn add_camera(&mut self, camera: Camera) -> NodeHandle {
        let node_handle = self.create_node_with_name("Camera");
        self.cameras.insert(node_handle, camera);
        self.root_nodes.push(node_handle);
        node_handle
    }

    pub fn add_camera_to_parent(&mut self, camera: Camera, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name("Camera");
        self.cameras.insert(node_handle, camera);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn add_light(&mut self, light: Light) -> NodeHandle {
        let node_handle = self.create_node_with_name("Light");
        self.lights.insert(node_handle, light);
        self.root_nodes.push(node_handle);
        node_handle
    }

    pub fn add_light_to_parent(&mut self, light: Light, parent: NodeHandle) -> NodeHandle {
        let node_handle = self.create_node_with_name("Light");
        self.lights.insert(node_handle, light);
        self.attach(node_handle, parent);
        node_handle
    }

    pub fn mark_as_split_primitive(&mut self, handle: NodeHandle) {
        self.split_primitive_tags.insert(handle, SplitPrimitiveTag);
    }

    /// Synchronizes shader macro definitions based on the current scene state.
    fn sync_shader_defines(&mut self) {
        let current_env_version = self.environment.version();

        // Only recompute if the environment version has changed since the last computation
        if self.last_env_version != current_env_version {
            let mut defines = ShaderDefines::new();

            // Recompute logic
            if self.environment.has_env_map() {
                defines.set("HAS_ENV_MAP", "1");
            }
            // ... additional defines based on scene state can be added here ...

            self.shader_defines = defines;
            self.last_env_version = current_env_version;
        }
    }

    /// Computes the scene's shader macro definitions
    ///
    /// Uses internal caching mechanism, only recalculates when Environment version changes.
    pub fn shader_defines(&self) -> &ShaderDefines {
        &self.shader_defines
    }

    // ========================================================================
    // Scene Update and Logic System
    // ========================================================================

    pub fn add_logic<L: SceneLogic>(&mut self, logic: L) {
        self.logics.push(Box::new(logic));
    }

    /// Shortcut method: Add closure logic (for quick prototyping)
    pub fn on_update<F>(&mut self, f: F)
    where
        F: FnMut(&mut Scene, &Input, f32) + Send + Sync + 'static,
    {
        self.add_logic(CallbackLogic(f));
    }

    /// Updates scene state (called every frame)
    pub fn update(&mut self, input: &Input, dt: f32) {
        // 1. Execute user scripts (Gameplay)
        let mut logics = std::mem::take(&mut self.logics);
        for logic in &mut logics {
            logic.update(self, input, dt);
        }
        self.logics.append(&mut logics);

        // 2. Animation system update (modifies node Transform)
        {
            let mut mixers = std::mem::take(&mut self.animation_mixers);
            for (_handle, mixer) in &mut mixers {
                mixer.update(dt, self);
            }
            self.animation_mixers = mixers;
        }

        // 3. Execute internal engine systems (Transform, Skeleton, Morph)
        self.update_matrix_world();
        self.update_skeletons();
        self.sync_morph_weights();
        self.sync_shader_defines();
        self.sync_gpu_buffers();
    }

    /// Syncs GPU Buffer data
    pub fn sync_gpu_buffers(&mut self) {
        self.sync_light_buffer();
        self.sync_environment_buffer();
    }

    /// Syncs light data to GPU Buffer
    fn sync_light_buffer(&mut self) {
        let mut cache = std::mem::take(&mut self.light_data_cache);

        cache.clear();

        for (light, world_matrix) in self.iter_active_lights() {
            let pos = world_matrix.translation.to_vec3();
            let dir = world_matrix.transform_vector3(-Vec3::Z).normalize();

            let mut gpu_light = GpuLightStorage {
                color: light.color,
                intensity: light.intensity,
                position: pos,
                direction: dir,
                shadow_layer_index: -1,
                ..Default::default()
            };

            match &light.kind {
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
                LightKind::Directional(_) => {
                    gpu_light.light_type = 0;
                }
            }

            cache.push(gpu_light);
        }

        if cache.is_empty() {
            cache.push(GpuLightStorage::default());
        }

        self.light_data_cache = cache;

        let needs_update =
            self.light_storage_buffer.read().as_slice() != self.light_data_cache.as_slice();

        if needs_update {
            self.light_storage_buffer
                .write()
                .clone_from(&self.light_data_cache);
        }
    }

    /// Syncs environment data to GPU Buffer
    fn sync_environment_buffer(&mut self) {
        let env = &self.environment;
        let light_count = self.lights.len();

        let new_uniforms = EnvironmentUniforms {
            ambient_light: env.ambient,
            num_lights: light_count as u32,
            env_map_intensity: env.intensity,
            env_map_rotation: env.rotation,
            // env_map_max_mip_level is set by ResourceManager::resolve_gpu_environment
            // during the prepare phase, so we preserve the existing value here.
            env_map_max_mip_level: self.uniforms_buffer.read().env_map_max_mip_level,
            ..Default::default()
        };

        let needs_update = *self.uniforms_buffer.read() != new_uniforms;

        if needs_update {
            *self.uniforms_buffer.write() = new_uniforms;
        }
    }

    // ========================================================================
    // GPU Resource Access Interface
    // ========================================================================

    pub fn light_storage(&self) -> &CpuBuffer<Vec<GpuLightStorage>> {
        &self.light_storage_buffer
    }

    pub fn environment_uniforms(&self) -> &CpuBuffer<EnvironmentUniforms> {
        &self.uniforms_buffer
    }

    pub fn update_skeletons(&mut self) {
        let mut tasks = Vec::new();

        for (node_handle, binding) in &self.skins {
            if let Some(node) = self.nodes.get(node_handle) {
                let root_inv = match binding.bind_mode {
                    BindMode::Attached => node.transform.world_matrix.inverse(),
                    BindMode::Detached => binding.bind_matrix_inv,
                };
                tasks.push((binding.skeleton, root_inv));
            }
        }

        for (skeleton_id, root_inv) in tasks {
            if let Some(skeleton) = self.skeleton_pool.get_mut(skeleton_id) {
                skeleton.compute_joint_matrices(&self.nodes, root_inv);
                // Lazy compute bounding box (only computed when first needed)
                if skeleton.local_bounds.is_none() {
                    skeleton.compute_local_bounds(&self.nodes);
                }
            }
        }
    }

    pub fn sync_morph_weights(&mut self) {
        for (handle, weights) in &self.morph_weights {
            if weights.is_empty() {
                continue;
            }

            let weights_slice = weights.as_slice();

            if let Some(mesh) = self.meshes.get_mut(handle) {
                mesh.set_morph_target_influences(weights_slice);
                mesh.update_morph_uniforms();
            } else if let Some(node) = self.nodes.get(handle) {
                for &child_handle in &node.children {
                    // Broadcast to child nodes that have SplitPrimitiveTag
                    if self.split_primitive_tags.contains_key(child_handle)
                        && let Some(child_mesh) = self.meshes.get_mut(child_handle)
                    {
                        child_mesh.set_morph_target_influences(weights_slice);
                        child_mesh.update_morph_uniforms();
                    }
                }
            }
        }
    }

    pub fn main_camera_node_mut(&mut self) -> Option<&mut Node> {
        let handle = self.active_camera?;
        self.get_node_mut(handle)
    }

    pub fn main_camera_node(&self) -> Option<&Node> {
        let handle = self.active_camera?;
        self.get_node(handle)
    }

    // ========================================================================
    // Background API
    // ========================================================================

    /// Sets the background to a solid color.
    pub fn set_background_color(&mut self, r: f32, g: f32, b: f32) {
        self.background.set_mode(BackgroundMode::color(r, g, b));
    }

    // ========================================================================
    // High-Level Helpers (node wrapper, builder)
    // ========================================================================

    /// Returns a chainable wrapper for the given node.
    ///
    /// Silently no-ops if the handle is stale, avoiding `unwrap()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// scene.node(&handle)
    ///     .set_position(0.0, 3.0, 0.0)
    ///     .set_scale(2.0)
    ///     .look_at(Vec3::ZERO);
    /// ```
    pub fn node(&mut self, handle: &NodeHandle) -> SceneNode<'_> {
        SceneNode::new(self, *handle)
    }

    /// Starts building a node
    pub fn build_node(&mut self, name: &str) -> NodeBuilder<'_> {
        NodeBuilder::new(self, name)
    }

    /// Finds a node by name
    pub fn find_node_by_name(&self, name: &str) -> Option<NodeHandle> {
        for (handle, node_name) in &self.names {
            if node_name.as_ref() == name {
                return Some(handle);
            }
        }
        None
    }

    /// Gets the global transform matrix of a node
    pub fn get_global_transform(&self, handle: NodeHandle) -> Affine3A {
        self.nodes
            .get(handle)
            .map_or(Affine3A::IDENTITY, |n| n.transform.world_matrix)
    }

    /// Plays a specific animation clip on the node (if an AnimationMixer is present)
    pub fn play_animation(&mut self, node_handle: NodeHandle, clip_name: &str) {
        if let Some(mixer) = self.animation_mixers.get_mut(node_handle) {
            mixer.play(clip_name);
        } else {
            log::warn!("No animation mixer found for node {node_handle:?}");
        }
    }

    /// Plays any animation on the node (used for simple cases where clip name is not important)
    pub fn play_if_any_animation(&mut self, node_handle: NodeHandle) {
        if let Some(mixer) = self.animation_mixers.get_mut(node_handle) {
            mixer
                .any_action()
                .map(myth_animation::mixer::ActionControl::play);
        } else {
            log::info!("No animation mixer found for node {node_handle:?}");
        }
    }

    // ========================================================================
    // Bounding Box Queries
    // ========================================================================

    /// Computes the world-space bounding box of a single node (not including children).
    ///
    /// `query` implements [`GeometryQuery`] to map geometry handles to local-space bounding boxes.
    fn get_bbox_of_one_node(
        &self,
        node_handle: NodeHandle,
        query: &impl crate::GeometryQuery,
    ) -> Option<myth_resources::BoundingBox> {
        let node = self.get_node(node_handle)?;
        if !node.visible {
            return None;
        }
        let mesh = self.meshes.get(node_handle)?;
        if !mesh.visible {
            return None;
        }

        // When there's a skeleton binding, use the skeleton's bounding box
        if let Some(skeleton_binding) = self.skins.get(node_handle)
            && let Some(skeleton) = self.skeleton_pool.get(skeleton_binding.skeleton)
        {
            return skeleton.compute_tight_world_bounds(&self.nodes);
        }

        // Otherwise compute from the geometry's static bounding box
        let local_bbox = query.get_geometry_bbox(mesh.geometry)?;
        Some(local_bbox.transform(&node.transform.world_matrix))
    }

    /// Recursively computes the world-space bounding box enclosing a node and all its descendants.
    ///
    /// `query` implements [`crate::GeometryQuery`] to map geometry handles to local-space bounding boxes.
    pub fn get_bbox_of_node(
        &self,
        node_handle: NodeHandle,
        query: &impl crate::GeometryQuery,
    ) -> Option<myth_resources::BoundingBox> {
        let mut combined_bbox = self.get_bbox_of_one_node(node_handle, query);
        let node = self.get_node(node_handle)?;
        for &child_handle in &node.children {
            if let Some(child_bbox) = self.get_bbox_of_node(child_handle, query) {
                combined_bbox = match combined_bbox {
                    Some(existing) => Some(existing.union(&child_bbox)),
                    None => Some(child_bbox),
                };
            }
        }

        combined_bbox
    }
}

// ============================================================================
// NodeBuilder
// ============================================================================

pub struct NodeBuilder<'a> {
    scene: &'a mut Scene,
    handle: NodeHandle,
    parent: Option<NodeHandle>,
    mesh: Option<Mesh>,
}

impl<'a> NodeBuilder<'a> {
    pub fn new(scene: &'a mut Scene, name: &str) -> Self {
        let handle = scene.nodes.insert(Node::new());
        scene.names.insert(handle, Cow::Owned(name.to_string()));
        Self {
            scene,
            handle,
            parent: None,
            mesh: None,
        }
    }

    #[must_use]
    pub fn with_position(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.nodes.get_mut(self.handle) {
            node.transform.position = glam::Vec3::new(x, y, z);
        }
        self
    }

    #[must_use]
    pub fn with_scale(self, s: f32) -> Self {
        if let Some(node) = self.scene.nodes.get_mut(self.handle) {
            node.transform.scale = glam::Vec3::splat(s);
        }
        self
    }

    #[must_use]
    pub fn with_parent(mut self, parent: NodeHandle) -> Self {
        self.parent = Some(parent);
        self
    }

    #[must_use]
    pub fn with_mesh(mut self, mesh: Mesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    pub fn build(self) -> NodeHandle {
        let handle = self.handle;

        // Set Mesh component
        if let Some(mesh) = self.mesh {
            self.scene.meshes.insert(handle, mesh);
        }

        // Handle parent-child relationship
        if let Some(parent) = self.parent {
            self.scene.attach(handle, parent);
        } else {
            self.scene.root_nodes.push(handle);
        }

        handle
    }
}

// ============================================================================
// AnimationTarget Implementation
// ============================================================================

impl AnimationTarget for Scene {
    fn node_children(&self, handle: NodeHandle) -> Option<Vec<NodeHandle>> {
        self.nodes.get(handle).map(|n| n.children().to_vec())
    }

    fn node_name(&self, handle: NodeHandle) -> Option<String> {
        self.names.get(handle).map(|n| n.as_ref().to_string())
    }

    fn node_transform(&self, handle: NodeHandle) -> Option<Transform> {
        self.nodes.get(handle).map(|n| n.transform)
    }

    fn set_node_position(&mut self, handle: NodeHandle, position: Vec3) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform.position = position;
        }
    }

    fn set_node_rotation(&mut self, handle: NodeHandle, rotation: Quat) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform.rotation = rotation;
        }
    }

    fn set_node_scale(&mut self, handle: NodeHandle, scale: Vec3) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform.scale = scale;
        }
    }

    fn mark_node_dirty(&mut self, handle: NodeHandle) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform.mark_dirty();
        }
    }

    fn has_rest_transform(&self, handle: NodeHandle) -> bool {
        self.rest_transforms.contains_key(handle)
    }

    fn rest_transform(&self, handle: NodeHandle) -> Option<Transform> {
        self.rest_transforms.get(handle).copied()
    }

    fn store_rest_transform(&mut self, handle: NodeHandle, transform: Transform) {
        self.rest_transforms.insert(handle, transform);
    }

    fn morph_weights_mut(&mut self, handle: NodeHandle) -> &mut Vec<f32> {
        self.morph_weights.entry(handle).unwrap().or_default()
    }
}
