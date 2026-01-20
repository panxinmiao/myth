use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use thunderdome::{Arena};
use slotmap::SlotMap;
use glam::{Affine3A, Vec3, Vec4}; 
use bitflags::bitflags;
use crate::AssetServer;
use crate::resources::BoundingBox;
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{EnvironmentUniforms, GpuLightStorage};
use crate::scene::node::Node;
use crate::scene::skeleton::{BindMode, Skeleton};
use crate::scene::transform::Transform;
use crate::scene::transform_system;
use crate::scene::light::LightKind;
use crate::resources::mesh::Mesh;
use crate::scene::camera::Camera;
use crate::scene::light::Light;
use crate::scene::environment::Environment;

use crate::scene::{CameraKey, LightKey, MeshKey, NodeIndex, SkeletonKey};

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct SceneFeatures: u32 {
        const USE_ENV_MAP  = 1 << 0;
        // const USE_SHADOW_MAP = 1 << 0;
    }
}

static NEXT_SCENE_ID: AtomicU32 = AtomicU32::new(1);

/// 场景图结构
/// 
/// Scene 是纯数据层，存储场景图逻辑和组件数据。
/// 同时维护自身的 GPU 资源描述（CpuBuffer），可被 ResourceManager 统一管理。
/// 
pub struct Scene {
    pub id: u32,

    pub nodes: Arena<Node>,
    pub root_nodes: Vec<NodeIndex>,

    // ====组件/资源池====
    pub meshes: SlotMap<MeshKey, Mesh>,
    pub cameras: SlotMap<CameraKey, Camera>,
    pub lights: SlotMap<LightKey, Light>,

    pub skins: SlotMap<SkeletonKey, Skeleton>,

    // 环境和全局设置
    pub environment: Environment,
    
    // 暂时简单用 RGBA，后面可以用 Texture
    pub background: Option<Vec4>,

    pub active_camera: Option<NodeIndex>,

    // ====GPU 资源描述（自管理）====
    /// 灯光存储 Buffer（Storage Buffer）
    pub(crate) light_storage_buffer: CpuBuffer<Vec<GpuLightStorage>>,
    /// 环境 Uniform Buffer
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

            nodes: Arena::new(),
            root_nodes: Vec::new(),
            meshes: SlotMap::with_key(),
            cameras: SlotMap::with_key(),
            lights: SlotMap::with_key(),

            skins: SlotMap::with_key(),

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

    /// 迭代场景中所有活跃的灯光
    pub fn iter_active_lights(&self) -> impl Iterator<Item = (&Light, &Affine3A)> {
        self.lights.iter().filter_map(move |(light_key, light)| {
            // 查找关联该 Light 的 Node
            let node_opt = self.nodes.iter().find_map(|(_, node)| {
                if node.light == Some(light_key) {
                    Some(node)
                } else {
                    None
                }
            });
            node_opt.map(|node| (light, &node.transform.world_matrix))
        })
    }

    /// 开始构建一个节点
    pub fn build_node(&'_ mut self, name: &str) -> NodeBuilder<'_> {
        NodeBuilder::new(self, name)
    }

    /// 添加一个节点到场景 (默认放在根节点)
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        let idx = self.nodes.insert(node);
        self.root_nodes.push(idx);
        idx
    }

    pub fn add_to_parent(&mut self, child: Node, parent_idx: NodeIndex) -> NodeIndex {
        let idx = self.nodes.insert(child);

        // 建立父子关系
        if let Some(p) = self.nodes.get_mut(parent_idx) {
            p.children.push(idx);
        }
        if let Some(c) = self.nodes.get_mut(idx) {
            c.parent = Some(parent_idx);
        }

        idx
    }

    /// 移除节点 (递归移除所有子节点)
    pub fn remove_node(&mut self, idx: NodeIndex) {
        // 1. 先把它的 children 列表拿出来，避免借用冲突
        let children = if let Some(node) = self.nodes.get(idx) {
            node.children.clone()
        } else {
            return;
        };

        // 2. 递归移除子节点
        for child in children {
            self.remove_node(child);
        }

        // 3. 处理父节点关系
        // 获取该节点的 parent ID
        let parent_opt = self.nodes.get(idx).and_then(|n| n.parent);

        if let Some(parent_idx) = parent_opt {
            // 从父节点的 children 列表中移除自己
            if let Some(parent) = self.nodes.get_mut(parent_idx)
                && let Some(pos) = parent.children.iter().position(|&x| x == idx) {
                    parent.children.remove(pos);
                }
        } else {
            // 如果是根节点，从 root_nodes 移除
            if let Some(pos) = self.root_nodes.iter().position(|&x| x == idx) {
                self.root_nodes.remove(pos);
            }
        }

        // 3. === 清理组件 ===
        if let Some(node) = self.nodes.get(idx) {
            // Mesh
            if let Some(mesh_idx) = node.mesh {
                self.meshes.remove(mesh_idx);
            }
            // Camera
            if let Some(cam_idx) = node.camera {
                self.cameras.remove(cam_idx);
            }
            // Light
            if let Some(light_idx) = node.light {
                self.lights.remove(light_idx);
            }
        }

        // 4. 彻底删除数据
        self.nodes.remove(idx);
    }

    /// 核心逻辑：建立父子关系 (Attach)
    pub fn attach(&mut self, child_idx: NodeIndex, parent_idx: NodeIndex) {
        if child_idx == parent_idx {{
            log::warn!("Cannot attach node to itself!");
            return;
        }}
        // 1. Detach from old
        let old_parent = self.nodes.get(child_idx).and_then(|n| n.parent);
        if let Some(p) = old_parent {
            if let Some(n) = self.nodes.get_mut(p)
                && let Some(i) = n.children.iter().position(|&x| x == child_idx) {
                    n.children.remove(i);
                }
        } else if let Some(i) = self.root_nodes.iter().position(|&x| x == child_idx) {
            self.root_nodes.remove(i);
        }

        // 2. Attach to new
        if let Some(p) = self.nodes.get_mut(parent_idx) {
            p.children.push(child_idx);
        } else {
            log::error!("Parent node not found during attach!");
            // 恢复 child 到 root_nodes 防止数据丢失（可选策略）
            self.root_nodes.push(child_idx);
            return;
        }
        
        // 3. Update child
        if let Some(c) = self.nodes.get_mut(child_idx) {
            c.parent = Some(parent_idx);
            c.transform.mark_dirty(); // 强制标记脏，确保矩阵更新
        }
    }

    /// 获取只读引用
    pub fn get_node(&self, idx: NodeIndex) -> Option<&Node> {
        self.nodes.get(idx)
    }

    /// 获取可变引用 (用于修改 TRS)
    pub fn get_node_mut(&mut self, idx: NodeIndex) -> Option<&mut Node> {
        self.nodes.get_mut(idx)
    }

    // ========================================================================
    // 组件查询 API (Component Query)
    // ========================================================================

    /// 获取主相机的 (Transform, Camera) 组合
    pub fn query_main_camera_bundle(&mut self) -> Option<(&mut Transform, &mut Camera)> {
        let node_id = self.active_camera?;
        self.query_camera_bundle(node_id)
    }

    pub fn query_camera_bundle(&mut self, node_id: NodeIndex) -> Option<(&mut Transform, &mut Camera)> {
        let camera_key = self.nodes.get(node_id)?.camera?;
        let camera = self.cameras.get_mut(camera_key)?;
        let transform = &mut self.nodes.get_mut(node_id)?.transform;

        Some((transform, camera))
    }

    /// 查询指定节点的 Transform 和 Light
    pub fn query_light_bundle(&mut self, node_id: NodeIndex) -> Option<(&mut Transform, &Light)> {
        let light_key = self.nodes.get(node_id)?.light?;
        let light = self.lights.get(light_key)?;
        let transform = &mut self.nodes.get_mut(node_id)?.transform;
        Some((transform, light))
    }

    /// 查询指定节点的 Transform 和 Mesh
    pub fn query_mesh_bundle(&mut self, node_id: NodeIndex) -> Option<(&mut Transform, &Mesh)> {
        let mesh_key = self.nodes.get(node_id)?.mesh?;
        let mesh = self.meshes.get(mesh_key)?;
        let transform = &mut self.nodes.get_mut(node_id)?.transform;
        Some((transform, mesh))
    }


    // ========================================================================
    // 矩阵更新流水线 (Affine3A 版)
    // ========================================================================

    /// 更新整个场景的世界矩阵
    /// 这是每帧渲染前必须调用的
    /// 
    /// 使用解耦的变换系统，只借用必要的数据结构
    pub fn update_matrix_world(&mut self) {
        // 使用迭代版本避免深层级场景的栈溢出
        transform_system::update_hierarchy_iterative(
            &mut self.nodes,
            &mut self.cameras,
            &self.root_nodes,
        );
    }

    /// 更新指定子树的世界矩阵
    /// 用于局部更新场景图的一部分
    pub fn update_subtree(&mut self, root_idx: NodeIndex) {
        transform_system::update_subtree(
            &mut self.nodes,
            &mut self.cameras,
            root_idx,
        );
    }


    // === 资源管理 API ===
    pub fn add_mesh(&mut self, mesh: Mesh) -> NodeIndex {
        let mut node = crate::scene::node::Node::new(&mesh.name);
        node.mesh = Some(self.meshes.insert(mesh));
        self.add_node(node)
    }

    pub fn add_mesh_to_parent(&mut self, mesh: Mesh, parent: NodeIndex) -> NodeIndex {
        let mut node = crate::scene::node::Node::new(&mesh.name);
        node.mesh = Some(self.meshes.insert(mesh));
        self.add_to_parent(node, parent)
    }

    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> SkeletonKey {
        self.skins.insert(skeleton)
    }

    pub fn add_camera(&mut self, camera: Camera) ->  NodeIndex {
        // 1. 创建 Node
        let mut node = Node::new("Camera");

        node.camera = Some(self.cameras.insert(camera));

        // 2. 插入 Node
        self.add_node(node)

    }

    pub fn add_camera_to_parent(&mut self, camera: Camera, parent: NodeIndex) -> NodeIndex {
        let mut node = Node::new("Camera");
        node.camera = Some(self.cameras.insert(camera));
        self.add_to_parent(node, parent)
    }

    pub fn add_light(&mut self, light: Light) -> NodeIndex {
        // 1. 创建 Node
        let mut node = Node::new("Light");

        node.light = Some(self.lights.insert(light));

        // 2. 插入 Node
        self.add_node(node)
    }

    pub fn add_light_to_parent(&mut self, light: Light, parent: NodeIndex) -> NodeIndex {
        let mut node = Node::new("Light");
        node.light = Some(self.lights.insert(light));
        self.add_to_parent(node, parent)
    }

    pub fn get_features(&self) -> SceneFeatures {
        
        // 示例：未来可以根据场景内容设置标志位
        // if self.lights.len() > 0 {
        //     features |= SceneFeatures::USE_SHADOW_MAP;
        // }
        let mut features = SceneFeatures::empty();

        if self.environment.has_env_map() {
            features |= SceneFeatures::USE_ENV_MAP;
        }
        features
    }

    /// 更新场景状态（每帧调用）
    /// 
    /// 只更新逻辑层数据，GPU 数据同步由 ResourceManager 负责
    pub fn update(&mut self) {
        self.update_matrix_world();
        self.update_skeletons();
        self.sync_gpu_buffers();
    }

    /// 同步 GPU Buffer 数据
    /// 
    /// 将灯光和环境数据同步到 CpuBuffer，等待 ResourceManager 上传到 GPU。
    /// 使用 CpuBuffer 的版本追踪机制，只有数据变化时才会触发上传。
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

            // 确保至少有一个占位灯光（Shader 要求）
            if cache.is_empty() {
                cache.push(GpuLightStorage::default());
            }
        }

        let current_data = self.light_storage_buffer.read();
        let cache_ref = self.light_data_cache.borrow();
        // 比较
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
        
        // 使用 CpuBuffer 的 write() 方法，自动追踪版本
        let current = self.uniforms_buffer.read();
        if current != &new_uniforms {
            *self.uniforms_buffer.write() = new_uniforms;
        }
    }
    
    // ========================================================================
    // GPU 资源访问接口
    // ========================================================================
    
    /// 获取灯光存储 Buffer 的引用
    pub fn light_storage(&self) -> &CpuBuffer<Vec<GpuLightStorage>> {
        &self.light_storage_buffer
    }
    
    /// 获取环境 Uniform Buffer 的引用
    pub fn environment_uniforms(&self) -> &CpuBuffer<EnvironmentUniforms> {
        &self.uniforms_buffer
    }

    pub fn update_skeletons(&mut self) {
        // 步骤 1: 收集“任务” (我们需要更新哪些 Skeleton，以及用什么参数)
        let mut tasks = Vec::new();

        for (_, node) in &self.nodes {
            if let Some(binding) = &node.skin {
                // 根据 BindMode 决定 root_inverse
                let root_inv = match binding.bind_mode {
                    BindMode::Attached => node.transform.world_matrix.inverse(),
                    BindMode::Detached => binding.bind_matrix_inv,
                };
                
                tasks.push((binding.skeleton, root_inv));
            }
        }

        // 步骤 2: 执行“任务” (修改 Skeleton)
        // 此时我们只需要只读访问 nodes，可变访问 skeletons
        let nodes = &self.nodes; // Immutable borrow
        
        for (skeleton_id, root_inv) in tasks {
            if let Some(skeleton) = self.skins.get_mut(skeleton_id) {
                skeleton.compute_joint_matrices(nodes, root_inv);
            }
        }
    }

    pub fn sync_morph_weights(&mut self) {
        let mut updates = Vec::new();
        
        for node in self.nodes.iter().map(|(_, n)| n) {
            if let Some(mesh_key) = node.mesh {
                if !node.morph_weights.is_empty() {
                    updates.push((mesh_key, node.morph_weights.clone()));
                }
            }
        }
        
        for (mesh_key, weights) in updates {
            if let Some(mesh) = self.meshes.get_mut(mesh_key) {
                mesh.set_morph_target_influences(&weights);
                mesh.update_morph_uniforms();
            }
        }
    }

    pub fn main_camera_node_mut(&mut self) -> Option<&mut Node> {
        let id = self.active_camera?; 
        self.get_node_mut(id)
    }

    pub fn main_camera_node(&self) -> Option<&Node> {
        let id = self.active_camera?;
        self.get_node(id)
    }

    fn get_bbox_of_one_node(&self, node_id: NodeIndex, assets: &AssetServer) -> Option<BoundingBox> {

        let node = self.get_node(node_id)?;

        let mesh_key = node.mesh?;
        let mesh = self.meshes.get(mesh_key)?;  
        let geometry = assets.get_geometry(mesh.geometry)?;

        let local_bbox = if let Some(bbox) = geometry.bounding_box.borrow().as_ref() {
            bbox.clone()
        } else {
            geometry.compute_bounding_volume(); 
            geometry.bounding_box.borrow().as_ref()?.clone()
        };


        if let Some(skeleton_binding) = &node.skin 
            && let Some(skeleton) = self.skins.get(skeleton_binding.skeleton) {
            // 如果骨骼存在

            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            let mut bone_found = false;

            for &bone_idx in &skeleton.bones {
                if let Some(bone_node) = self.get_node(bone_idx) {
                    // 这里的 translation 必须是 World Position
                    let pos = bone_node.transform.world_matrix.translation.to_vec3();
                    min = min.min(pos);
                    max = max.max(pos);
                    bone_found = true;
                }
            }

            if !bone_found {
                return None;
            }

            let approximate_radius = 0.15;
            
            Some(BoundingBox { min, max }.inflate(approximate_radius))

        }else{
            // 如果没有骨骼，直接使用 Mesh 的几何体
            let world_matrix = &node.transform.world_matrix;
            Some(local_bbox.transform(world_matrix))
        }

    }

    pub fn get_bbox_of_node(&self, node_id: NodeIndex, assets: &AssetServer) -> Option<BoundingBox> {
        //let mut combined_bbox: Option<BoundingBox> = None;
        let mut combined_bbox = self.get_bbox_of_one_node(node_id, assets);        

        let node = self.get_node(node_id)?;
        // 递归检查子节点
        for &child_id in &node.children {
            if let Some(child_bbox) = self.get_bbox_of_node(child_id, assets) {
                combined_bbox = match combined_bbox {
                    Some(existing_bbox) => Some(existing_bbox.union(&child_bbox)),
                    None => Some(child_bbox),
                };
            }
        }

        combined_bbox
    }

}


pub struct NodeBuilder<'a> {
    scene: &'a mut Scene,
    node: Node, // 暂存正在构建的 Node 数据
    parent: Option<NodeIndex>, // 暂存父节点 ID
}

impl<'a> NodeBuilder<'a> {
    pub fn new(scene: &'a mut Scene, name: &str) -> Self {
        Self {
            scene,
            node: Node::new(name),
            parent: None,
        }
    }

    // === 链式配置方法 ===

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.node.transform.position = glam::Vec3::new(x, y, z);
        self
    }

    pub fn with_scale(mut self, s: f32) -> Self {
        self.node.transform.scale = glam::Vec3::splat(s);
        self
    }

    /// 设置父节点
    pub fn with_parent(mut self, parent: NodeIndex) -> Self {
        self.parent = Some(parent);
        self
    }

    /// 关联 Mesh (传入 Mesh 句柄)
    pub fn with_mesh(mut self, mesh: crate::scene::MeshKey) -> Self {
        self.node.mesh = Some(mesh);
        self
    }

    // === 终结方法 ===

    /// 完成构建，将 Node 插入 Scene，返回 Index
    pub fn build(self) -> NodeIndex {
        // 1. 插入 Node Arena
        let node_idx = self.scene.nodes.insert(self.node);

        // 2. 处理父子关系
        if let Some(parent_idx) = self.parent {
            self.scene.attach(node_idx, parent_idx);
        } else {
            // 如果没父节点，默认加入 root
            self.scene.root_nodes.push(node_idx);
        }

        node_idx
    }
}