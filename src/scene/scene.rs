use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use slotmap::{SlotMap, SecondaryMap, SparseSecondaryMap};
use glam::{Affine3A, Vec3, Vec4}; 
use bitflags::bitflags;
use crate::AssetServer;
use crate::resources::BoundingBox;
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{EnvironmentUniforms, GpuLightStorage};
use crate::resources::mesh::MAX_MORPH_TARGETS;
use crate::scene::node::Node;
use crate::scene::skeleton::{BindMode, Skeleton, SkinBinding};
use crate::scene::transform::Transform;
use crate::scene::transform_system;
use crate::scene::light::LightKind;
use crate::resources::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::scene::light::Light;
use crate::scene::environment::Environment;

use crate::scene::{NodeHandle, SkeletonKey};

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct SceneFeatures: u32 {
        const USE_ENV_MAP  = 1 << 0;
    }
}

static NEXT_SCENE_ID: AtomicU32 = AtomicU32::new(1);

/// 场景图结构 (ECS 风格)
/// 
/// Scene 是纯数据层，存储场景图逻辑和组件数据。
/// 使用 SlotMap + SecondaryMap 实现高性能组件化存储。
/// 
/// 设计原则：
/// - `nodes`: 核心节点数据（层级关系和变换），使用 SlotMap 存储
/// - 密集组件（names, meshes）使用 SecondaryMap
/// - 稀疏组件（cameras, lights, skins）使用 SparseSecondaryMap
pub struct Scene {
    pub id: u32,

    // === 核心节点存储 ===
    pub nodes: SlotMap<NodeHandle, Node>,
    pub root_nodes: Vec<NodeHandle>,

    // === 密集组件 (大多数节点都有) ===
    /// 节点名称 - 几乎所有节点都有名字
    pub names: SecondaryMap<NodeHandle, Cow<'static, str>>,

    // === 稀疏组件 (只有少数节点有) ===
    /// 网格组件 - 直接存储 Mesh 到节点
    pub meshes: SparseSecondaryMap<NodeHandle, Mesh>,
    /// 相机组件 - 直接存储 Camera 到节点
    pub cameras: SparseSecondaryMap<NodeHandle, Camera>,
    /// 灯光组件 - 直接存储 Light 到节点
    pub lights: SparseSecondaryMap<NodeHandle, Light>,
    /// 骨骼蒙皮绑定
    pub skins: SparseSecondaryMap<NodeHandle, SkinBinding>,
    /// 形变权重
    pub morph_weights: SparseSecondaryMap<NodeHandle, Vec<f32>>,

    morph_target_nodes: Vec<NodeHandle>,

    // === 资源池 (只保留真正需要共享的资源) ===
    /// Skeleton 是真正的共享资源，多个角色可能引用同一个骨架定义
    pub skeleton_pool: SlotMap<SkeletonKey, Skeleton>,

    // === 环境和全局设置 ===
    pub environment: Environment,
    pub background: Option<Vec4>,
    pub active_camera: Option<NodeHandle>,

    // === GPU 资源描述 ===
    pub(crate) light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    pub(crate) uniforms_buffer: CpuBuffer<EnvironmentUniforms>,
    light_data_cache: RefCell<Vec<GpuLightStorage>>,
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

            // 密集组件
            names: SecondaryMap::new(),

            // 稀疏组件（直接存储组件）
            meshes: SparseSecondaryMap::new(),
            cameras: SparseSecondaryMap::new(),
            lights: SparseSecondaryMap::new(),
            skins: SparseSecondaryMap::new(),
            morph_weights: SparseSecondaryMap::new(),
            morph_target_nodes: Vec::with_capacity(16),

            // 资源池（仅保留真正共享的资源）
            skeleton_pool: SlotMap::with_key(),

            environment: Environment::new(),
            background: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),

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

            light_data_cache: RefCell::new(Vec::with_capacity(16)),
        }
    }

    // ========================================================================
    // 节点管理 API
    // ========================================================================

    /// 创建一个新节点并返回句柄
    pub fn create_node(&mut self) -> NodeHandle {
        self.nodes.insert(Node::new())
    }

    /// 创建一个带名称的新节点
    pub fn create_node_with_name(&mut self, name: &str) -> NodeHandle {
        let handle = self.nodes.insert(Node::new());
        self.names.insert(handle, Cow::Owned(name.to_string()));
        handle
    }

    /// 添加一个节点到场景 (默认放在根节点)
    pub fn add_node(&mut self, node: Node) -> NodeHandle {
        let handle = self.nodes.insert(node);
        self.root_nodes.push(handle);
        handle
    }

    /// 添加一个节点作为指定父节点的子节点
    pub fn add_to_parent(&mut self, child: Node, parent_handle: NodeHandle) -> NodeHandle {
        let handle = self.nodes.insert(child);

        // 建立父子关系
        if let Some(parent) = self.nodes.get_mut(parent_handle) {
            parent.children.push(handle);
        }
        if let Some(child_node) = self.nodes.get_mut(handle) {
            child_node.parent = Some(parent_handle);
        }

        handle
    }

    /// 移除节点 (递归移除所有子节点)
    pub fn remove_node(&mut self, handle: NodeHandle) {
        // 1. 收集需要删除的所有节点（深度优先）
        let mut to_remove = Vec::new();
        self.collect_subtree(handle, &mut to_remove);

        // 2. 处理父节点关系
        if let Some(node) = self.nodes.get(handle) {
            if let Some(parent_handle) = node.parent {
                if let Some(parent) = self.nodes.get_mut(parent_handle) {
                    parent.children.retain(|&h| h != handle);
                }
            } else {
                self.root_nodes.retain(|&h| h != handle);
            }
        }

        // 3. 删除所有节点及其组件
        for node_handle in to_remove {
            // 清理组件（直接删除，无需再从资源池移除）
            self.meshes.remove(node_handle);
            self.cameras.remove(node_handle);
            self.lights.remove(node_handle);
            self.skins.remove(node_handle);
            self.morph_weights.remove(node_handle);
            self.names.remove(node_handle);

            // 删除节点
            self.nodes.remove(node_handle);
        }
    }

    /// 收集子树中的所有节点
    fn collect_subtree(&self, handle: NodeHandle, result: &mut Vec<NodeHandle>) {
        result.push(handle);
        if let Some(node) = self.nodes.get(handle) {
            for &child in &node.children {
                self.collect_subtree(child, result);
            }
        }
    }

    /// 建立父子关系 (Attach)
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

    /// 获取只读引用
    #[inline]
    pub fn get_node(&self, handle: NodeHandle) -> Option<&Node> {
        self.nodes.get(handle)
    }

    /// 获取可变引用
    #[inline]
    pub fn get_node_mut(&mut self, handle: NodeHandle) -> Option<&mut Node> {
        self.nodes.get_mut(handle)
    }

    // ========================================================================
    // 组件管理 API (ECS 风格)
    // ========================================================================

    /// 设置节点名称
    pub fn set_name(&mut self, handle: NodeHandle, name: &str) {
        self.names.insert(handle, Cow::Owned(name.to_string()));
    }

    /// 获取节点名称
    pub fn get_name(&self, handle: NodeHandle) -> Option<&str> {
        self.names.get(handle).map(|s| s.as_ref())
    }

    /// 为节点设置 Mesh 组件
    pub fn set_mesh(&mut self, handle: NodeHandle, mesh: Mesh) {
        self.meshes.insert(handle, mesh);
    }

    /// 获取节点的 Mesh 组件引用
    pub fn get_mesh(&self, handle: NodeHandle) -> Option<&Mesh> {
        self.meshes.get(handle)
    }

    /// 获取节点的 Mesh 组件可变引用
    pub fn get_mesh_mut(&mut self, handle: NodeHandle) -> Option<&mut Mesh> {
        self.meshes.get_mut(handle)
    }

    /// 为节点设置 Camera 组件
    pub fn set_camera(&mut self, handle: NodeHandle, camera: Camera) {
        self.cameras.insert(handle, camera);
    }

    /// 获取节点的 Camera 组件引用
    pub fn get_camera(&self, handle: NodeHandle) -> Option<&Camera> {
        self.cameras.get(handle)
    }

    /// 获取节点的 Camera 组件可变引用
    pub fn get_camera_mut(&mut self, handle: NodeHandle) -> Option<&mut Camera> {
        self.cameras.get_mut(handle)
    }

    /// 为节点设置 Light 组件
    pub fn set_light(&mut self, handle: NodeHandle, light: Light) {
        self.lights.insert(handle, light);
    }

    /// 获取节点的 Light 组件引用
    pub fn get_light(&self, handle: NodeHandle) -> Option<&Light> {
        self.lights.get(handle)
    }

    /// 获取节点的 Light 组件可变引用
    pub fn get_light_mut(&mut self, handle: NodeHandle) -> Option<&mut Light> {
        self.lights.get_mut(handle)
    }

    /// 为节点绑定骨架
    pub fn bind_skeleton(&mut self, handle: NodeHandle, skeleton_key: SkeletonKey, bind_mode: BindMode) {
        if let Some(node) = self.nodes.get(handle) {
            let bind_matrix_inv = node.transform.world_matrix.inverse();
            self.skins.insert(handle, SkinBinding {
                skeleton: skeleton_key,
                bind_mode,
                bind_matrix_inv,
            });
        }
    }

    /// 获取节点的骨骼绑定
    pub fn get_skin(&self, handle: NodeHandle) -> Option<&SkinBinding> {
        self.skins.get(handle)
    }

    /// 设置形变权重
    pub fn set_morph_weights(&mut self, handle: NodeHandle, weights: Vec<f32>) {
        self.morph_weights.insert(handle, weights);
    }

    /// 获取形变权重
    pub fn get_morph_weights(&self, handle: NodeHandle) -> Option<&Vec<f32>> {
        self.morph_weights.get(handle)
    }

    /// 获取可变形变权重引用
    pub fn get_morph_weights_mut(&mut self, handle: NodeHandle) -> Option<&mut Vec<f32>> {
        self.morph_weights.get_mut(handle)
    }

    /// 设置节点的形变权重（从 POD 数据）
    pub fn set_morph_weights_from_pod(&mut self, handle: NodeHandle, data: &crate::animation::values::MorphWeightData) {
        let weights = self.morph_weights.entry(handle)
            .unwrap()
            .or_insert_with(|| vec![0.0; MAX_MORPH_TARGETS]);
        
        if weights.len() < MAX_MORPH_TARGETS {
            weights.resize(MAX_MORPH_TARGETS, 0.0);
        }
        weights.copy_from_slice(&data.weights);
    }

    // ========================================================================
    // 迭代场景中所有活跃的灯光
    // ========================================================================
    
    pub fn iter_active_lights(&self) -> impl Iterator<Item = (&Light, &Affine3A)> {
        self.lights.iter().filter_map(move |(node_handle, light)| {
            let node = self.nodes.get(node_handle)?;
            Some((light, &node.transform.world_matrix))
        })
    }

    // ========================================================================
    // 组件查询 API
    // ========================================================================

    /// 获取主相机的 (Transform, Camera) 组合
    pub fn query_main_camera_bundle(&mut self) -> Option<(&mut Transform, &mut Camera)> {
        let node_handle = self.active_camera?;
        self.query_camera_bundle(node_handle)
    }

    pub fn query_camera_bundle(&mut self, node_handle: NodeHandle) -> Option<(&mut Transform, &mut Camera)> {
        // 检查是否存在相机组件
        if !self.cameras.contains_key(node_handle) {
            return None;
        }
        
        // 使用指针以避免同时借用 nodes 和 cameras 的冲突
        let transform_ptr = self.nodes.get_mut(node_handle)
            .map(|n| &mut n.transform as *mut Transform)?;
        let camera = self.cameras.get_mut(node_handle)?;
        
        // SAFETY: transform 和 camera 是不相交的内存区域
        unsafe { Some((&mut *transform_ptr, camera)) }
    }

    /// 查询指定节点的 Transform 和 Light
    pub fn query_light_bundle(&mut self, node_handle: NodeHandle) -> Option<(&mut Transform, &Light)> {
        let light = self.lights.get(node_handle)?;
        let transform = &mut self.nodes.get_mut(node_handle)?.transform;
        Some((transform, light))
    }

    /// 查询指定节点的 Transform 和 Mesh
    pub fn query_mesh_bundle(&mut self, node_handle: NodeHandle) -> Option<(&mut Transform, &Mesh)> {
        let mesh = self.meshes.get(node_handle)?;
        let transform = &mut self.nodes.get_mut(node_handle)?.transform;
        Some((transform, mesh))
    }

    // ========================================================================
    // 矩阵更新流水线
    // ========================================================================

    /// 更新整个场景的世界矩阵
    pub fn update_matrix_world(&mut self) {
        transform_system::update_hierarchy_iterative(
            &mut self.nodes,
            &mut self.cameras,
            &self.root_nodes,
        );
    }

    /// 更新指定子树的世界矩阵
    pub fn update_subtree(&mut self, root_handle: NodeHandle) {
        transform_system::update_subtree(
            &mut self.nodes,
            &mut self.cameras,
            root_handle,
        );
    }

    // ========================================================================
    // 资源管理 API
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

    pub fn get_features(&self) -> SceneFeatures {
        let mut features = SceneFeatures::empty();
        if self.environment.has_env_map() {
            features |= SceneFeatures::USE_ENV_MAP;
        }
        features
    }

    /// 更新场景状态（每帧调用）
    pub fn update(&mut self) {
        self.update_matrix_world();
        self.update_skeletons();
        self.sync_morph_weights();
        self.sync_gpu_buffers();
    }

    /// 同步 GPU Buffer 数据
    pub fn sync_gpu_buffers(&mut self) {
        self.sync_light_buffer();
        self.sync_environment_buffer();
    }

    /// 同步灯光数据到 GPU Buffer
    fn sync_light_buffer(&mut self) {
        {
            let mut cache = self.light_data_cache.borrow_mut();
            cache.clear();

            for (light, world_matrix) in self.iter_active_lights() {
                let pos = world_matrix.translation.to_vec3();
                let dir = world_matrix.transform_vector3(-Vec3::Z).normalize();

                let mut gpu_light = GpuLightStorage {
                    color: light.color,
                    intensity: light.intensity,
                    position: pos,
                    direction: dir,
                    ..Default::default()
                };

                match &light.kind {
                    LightKind::Point(point) => {
                        gpu_light.range = point.range;
                    }
                    LightKind::Spot(spot) => {
                        gpu_light.range = spot.range;
                        gpu_light.inner_cone_cos = spot.inner_cone.cos();
                        gpu_light.outer_cone_cos = spot.outer_cone.cos();
                    }
                    LightKind::Directional(_) => {}
                }

                cache.push(gpu_light);
            }

            if cache.is_empty() {
                cache.push(GpuLightStorage::default());
            }
        }

        let current_data = self.light_storage_buffer.read();
        let cache_ref = self.light_data_cache.borrow();
        if current_data.as_slice() != cache_ref.as_slice() {
            self.light_storage_buffer.write().clone_from(&cache_ref);
        }
    }

    /// 同步环境数据到 GPU Buffer
    fn sync_environment_buffer(&mut self) {
        let env = &self.environment;
        let light_count = self.lights.len();

        let new_uniforms = EnvironmentUniforms {
            ambient_light: env.ambient_color,
            num_lights: light_count as u32,
            env_map_intensity: env.intensity,
            env_map_max_mip_level: env.env_map_max_mip_level,
            ..Default::default()
        };

        let current = self.uniforms_buffer.read();
        if current != &new_uniforms {
            *self.uniforms_buffer.write() = new_uniforms;
        }
    }

    // ========================================================================
    // GPU 资源访问接口
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
                // 惰性计算包围盒（仅在首次需要时计算）
                if skeleton.local_bounds.is_none() {
                    skeleton.compute_local_bounds(&self.nodes);
                }
            }
        }
    }

    pub fn sync_morph_weights(&mut self) {
        self.morph_target_nodes.clear();
        
        for (handle, _) in &self.morph_weights {
            if self.meshes.contains_key(handle) {
                self.morph_target_nodes.push(handle);
            }
        }

        for handle in &self.morph_target_nodes {            
            if let Some(weights) = self.morph_weights.get(*handle) {
                if weights.is_empty() { continue; }
                
                let weights_slice = weights.as_slice();
                
                if let Some(mesh) = self.meshes.get_mut(*handle) {
                    mesh.set_morph_target_influences(weights_slice);
                    mesh.update_morph_uniforms();
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

    fn get_bbox_of_one_node(&self, node_handle: NodeHandle, assets: &AssetServer) -> Option<BoundingBox> {
        let node = self.get_node(node_handle)?;
        if node.visible == false {
            return None;
        }
        let mesh = self.meshes.get(node_handle)?;
        if mesh.visible == false {
            return None;
        }
        let geometry = assets.get_geometry(mesh.geometry)?;

        // 有骨骼绑定时使用 Skeleton 的包围盒
        if let Some(skeleton_binding) = self.skins.get(node_handle) {
            if let Some(skeleton) = self.skeleton_pool.get(skeleton_binding.skeleton) {
                return skeleton.compute_tight_world_bounds(&self.nodes);
            }
        }

        // 无骨骼绑定时使用 Geometry 的包围盒
        let local_bbox = if let Some(bbox) = geometry.bounding_box.borrow().as_ref() {
            *bbox
        } else {
            geometry.compute_bounding_volume();
            *geometry.bounding_box.borrow().as_ref()?
        };

        Some(local_bbox.transform(&node.transform.world_matrix))
    }

    pub fn get_bbox_of_node(&self, node_handle: NodeHandle, assets: &AssetServer) -> Option<BoundingBox> {
        let mut combined_bbox = self.get_bbox_of_one_node(node_handle, assets);

        let node = self.get_node(node_handle)?;
        for &child_handle in &node.children {
            if let Some(child_bbox) = self.get_bbox_of_node(child_handle, assets) {
                combined_bbox = match combined_bbox {
                    Some(existing_bbox) => Some(existing_bbox.union(&child_bbox)),
                    None => Some(child_bbox),
                };
            }
        }

        combined_bbox
    }

    /// 开始构建一个节点
    pub fn build_node(&mut self, name: &str) -> NodeBuilder<'_> {
        NodeBuilder::new(self, name)
    }

    /// 根据名称查找节点
    pub fn find_node_by_name(&self, name: &str) -> Option<NodeHandle> {
        for (handle, node_name) in &self.names {
            if node_name.as_ref() == name {
                return Some(handle);
            }
        }
        None
    }

    /// 获取节点的全局变换矩阵
    pub fn get_global_transform(&self, handle: NodeHandle) -> Affine3A {
        self.nodes.get(handle)
            .map(|n| n.transform.world_matrix)
            .unwrap_or(Affine3A::IDENTITY)
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

    pub fn with_position(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.nodes.get_mut(self.handle) {
            node.transform.position = glam::Vec3::new(x, y, z);
        }
        self
    }

    pub fn with_scale(self, s: f32) -> Self {
        if let Some(node) = self.scene.nodes.get_mut(self.handle) {
            node.transform.scale = glam::Vec3::splat(s);
        }
        self
    }

    pub fn with_parent(mut self, parent: NodeHandle) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn with_mesh(mut self, mesh: Mesh) -> Self {
        self.mesh = Some(mesh);
        self
    }

    pub fn build(self) -> NodeHandle {
        let handle = self.handle;

        // 设置 Mesh 组件
        if let Some(mesh) = self.mesh {
            self.scene.meshes.insert(handle, mesh);
        }

        // 处理父子关系
        if let Some(parent) = self.parent {
            self.scene.attach(handle, parent);
        } else {
            self.scene.root_nodes.push(handle);
        }

        handle
    }
}
