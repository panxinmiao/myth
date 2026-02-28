use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::AssetServer;
use crate::animation::{AnimationAction, AnimationMixer, AnimationSystem, Binder};
use crate::assets::prefab::Prefab;
use crate::resources::bloom::BloomSettings;
use crate::resources::buffer::CpuBuffer;
use crate::resources::fxaa::FxaaSettings;
use crate::resources::geometry::Geometry;
use crate::resources::mesh::Mesh;
use crate::resources::screen_space::ScreenSpaceSettings;
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::ssao::SsaoSettings;
use crate::resources::tone_mapping::ToneMappingSettings;
use crate::resources::uniforms::{EnvironmentUniforms, GpuLightStorage};
use crate::resources::{BoundingBox, Input};
use crate::scene::background::{BackgroundMode, BackgroundSettings};
use crate::scene::camera::Camera;
use crate::scene::environment::Environment;
use crate::scene::light::Light;
use crate::scene::light::LightKind;
use crate::scene::node::Node;
use crate::scene::resolve::{ResolveGeometry, ResolveMaterial};
use crate::scene::skeleton::{BindMode, Skeleton, SkinBinding};
use crate::scene::transform::Transform;
use crate::scene::transform_system;
use crate::scene::wrapper::SceneNode;
use glam::{Affine3A, Vec3};
use slotmap::{SecondaryMap, SlotMap, SparseSecondaryMap};

use crate::scene::{NodeHandle, SkeletonKey};

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
    /// Unique scene identifier
    pub id: u32,

    /// Built-in asset server reference (cheap Arc clone).
    /// Enables `spawn()` and other helpers to auto-register resources.
    pub assets: AssetServer,

    // === Core Node Storage ===
    /// All nodes in the scene (`SlotMap` for O(1) access)
    pub nodes: SlotMap<NodeHandle, Node>,
    /// Root-level nodes (no parent)
    pub root_nodes: Vec<NodeHandle>,

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
    /// Split primitive tags
    pub split_primitive_tags: SparseSecondaryMap<NodeHandle, SplitPrimitiveTag>,

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
    /// FXAA (Fast Approximate Anti-Aliasing) settings
    pub fxaa: FxaaSettings,
    /// SSAO (Screen Space Ambient Occlusion) settings
    pub ssao: SsaoSettings,
    /// Screen space effects settings (SSS, SSR)
    pub screen_space: ScreenSpaceSettings,
    /// Background rendering settings (mode + skybox uniform buffer)
    pub background: BackgroundSettings,
    /// Currently active camera for rendering
    pub active_camera: Option<NodeHandle>,

    // === GPU Resource Descriptors ===
    pub(crate) light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    pub(crate) uniforms_buffer: CpuBuffer<EnvironmentUniforms>,
    light_data_cache: Vec<GpuLightStorage>,

    pub shader_defines: ShaderDefines,

    last_env_version: u64,

    // === Scene Logic System ===
    pub(crate) logics: Vec<Box<dyn SceneLogic>>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new(AssetServer::default())
    }
}

impl Scene {
    pub fn new(assets: AssetServer) -> Self {
        Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            assets,

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

            split_primitive_tags: SparseSecondaryMap::new(),

            // Resource pools (only truly shared resources)
            skeleton_pool: SlotMap::with_key(),

            environment: Environment::new(),
            tone_mapping: ToneMappingSettings::default(),
            bloom: BloomSettings::default(),
            fxaa: FxaaSettings::default(),
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
        data: &crate::animation::values::MorphWeightData,
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

    /// Instantiates a Prefab into the scene
    ///
    /// Instantiates the node tree, skeletons, and animations from the Prefab as scene objects.
    /// Returns the root node handle and a mapping of all created node handles.
    pub fn instantiate(&mut self, prefab: &Prefab) -> NodeHandle {
        let node_count = prefab.nodes.len();
        let mut node_map: Vec<NodeHandle> = Vec::with_capacity(node_count);

        // Pass 1: Create all nodes and map indices
        for p_node in &prefab.nodes {
            let handle = self.create_node();

            if let Some(name) = &p_node.name {
                self.set_name(handle, name);
            }

            if let Some(node) = self.get_node_mut(handle) {
                node.transform = p_node.transform.clone();
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
        self.root_nodes.push(root_handle);

        for &root_idx in &prefab.root_indices {
            if let Some(&node_handle) = node_map.get(root_idx) {
                self.attach(node_handle, root_handle);
            }
        }

        // Pass 6: Create animation mixer and bind animations
        if !prefab.animations.is_empty() {
            let mut mixer = AnimationMixer::new();

            for clip in &prefab.animations {
                let bindings = Binder::bind(self, root_handle, clip);

                let mut action = AnimationAction::new(Arc::new(clip.clone()));
                action.bindings = bindings;
                action.enabled = false;
                action.weight = 0.0;

                mixer.add_action(action);
            }

            self.animation_mixers.insert(root_handle, mixer);
        }

        // Pass 7: Garbage collection of orphan nodes
        // remove all nodes that are not part of the current scene hierarchy (root_indices)
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
        AnimationSystem::update(self, dt);

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

    fn get_bbox_of_one_node(&self, node_handle: NodeHandle) -> Option<BoundingBox> {
        let node = self.get_node(node_handle)?;
        if !node.visible {
            return None;
        }
        let mesh = self.meshes.get(node_handle)?;
        if !mesh.visible {
            return None;
        }
        let geometry = self.assets.geometries.get(mesh.geometry)?;

        // When there's a skeleton binding, use Skeleton's bounding box
        if let Some(skeleton_binding) = self.skins.get(node_handle)
            && let Some(skeleton) = self.skeleton_pool.get(skeleton_binding.skeleton)
        {
            return skeleton.compute_tight_world_bounds(&self.nodes);
        }

        // When there's no skeleton binding, use Geometry's bounding box
        let local_bbox = geometry.bounding_box;
        Some(local_bbox.transform(&node.transform.world_matrix))
    }

    pub fn get_bbox_of_node(&self, node_handle: NodeHandle) -> Option<BoundingBox> {
        let mut combined_bbox = self.get_bbox_of_one_node(node_handle);

        let node = self.get_node(node_handle)?;
        for &child_handle in &node.children {
            if let Some(child_bbox) = self.get_bbox_of_node(child_handle) {
                combined_bbox = match combined_bbox {
                    Some(existing_bbox) => Some(existing_bbox.union(&child_bbox)),
                    None => Some(child_bbox),
                };
            }
        }

        combined_bbox
    }

    // ========================================================================
    // Background API
    // ========================================================================

    /// Sets the background to a solid color.
    pub fn set_background_color(&mut self, r: f32, g: f32, b: f32) {
        self.background.set_mode(BackgroundMode::color(r, g, b));
    }

    // ========================================================================
    // High-Level Helpers (spawn, node wrapper)
    // ========================================================================

    /// Creates a mesh node from any geometry/material combination.
    ///
    /// Accepts either pre-registered handles or raw resource structs.
    /// When structs are passed they are auto-registered via the scene's
    /// built-in [`AssetServer`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Pass structs (auto-registered):
    /// let cube = scene.spawn(
    ///     Geometry::new_box(1.0, 1.0, 1.0),
    ///     PhongMaterial::new(Vec4::ONE),
    /// );
    ///
    /// // Or pass existing handles:
    /// let cube = scene.spawn(geo_handle, mat_handle);
    /// ```
    pub fn spawn(
        &mut self,
        geometry: impl ResolveGeometry,
        material: impl ResolveMaterial,
    ) -> NodeHandle {
        let geo_handle = geometry.resolve(&self.assets);
        let mat_handle = material.resolve(&self.assets);
        let mesh = Mesh::new(geo_handle, mat_handle);
        self.add_mesh(mesh)
    }

    /// Shortcut: spawn a box mesh.
    pub fn spawn_box(
        &mut self,
        w: f32,
        h: f32,
        d: f32,
        material: impl ResolveMaterial,
    ) -> NodeHandle {
        self.spawn(Geometry::new_box(w, h, d), material)
    }

    /// Shortcut: spawn a sphere mesh.
    pub fn spawn_sphere(&mut self, radius: f32, material: impl ResolveMaterial) -> NodeHandle {
        self.spawn(Geometry::new_sphere(radius), material)
    }

    /// Shortcut: spawn a plane mesh.
    pub fn spawn_plane(
        &mut self,
        width: f32,
        height: f32,
        material: impl ResolveMaterial,
    ) -> NodeHandle {
        self.spawn(Geometry::new_plane(width, height), material)
    }

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
                .map(super::super::animation::mixer::ActionControl::play);
        } else {
            log::info!("No animation mixer found for node {node_handle:?}");
        }
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
