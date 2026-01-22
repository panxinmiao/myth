//! 渲染提取阶段 (Extract Phase)
//!
//! 在渲染开始前，从 Scene 中提取当前帧所需的精简数据。
//! Extract 完成后，Scene 就可以被释放，后续渲染准备工作不再依赖 Scene 的借用。
//!
//! # 设计原则
//! - 只复制渲染所需的"精简数据"，不复制实际的 Mesh/Material 资源
//! - 视锥剔除前置，只提取可见物体
//! - 使用 Copy 类型尽可能减少开销
//! - 携带缓存 ID 以避免每帧重复查找

use glam::Mat4;

use crate::scene::scene::SceneFeatures;
use crate::scene::{NodeHandle, Scene, SkeletonKey, MeshKey};
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::scene::camera::Camera;

/// 精简的渲染项，只包含 GPU 需要的数据
/// 
/// 使用 Clone 而非 Copy，因为 SkinBinding 包含非 Copy 类型
#[derive(Clone)]
pub struct ExtractedRenderItem {
    /// 节点句柄（用于调试和回溯）
    pub node_handle: NodeHandle,
    /// Mesh 的 Key（用于回写缓存）
    pub mesh_key: MeshKey,
    /// 世界变换矩阵 (64 bytes)
    pub world_matrix: Mat4,
    /// 几何体句柄 (8 bytes)
    pub geometry: GeometryHandle,
    /// 材质句柄 (8 bytes)
    pub material: MaterialHandle,
    /// 蒙皮绑定信息（可选）
    pub skeleton: Option<SkeletonKey>,
    /// 到相机的距离平方（用于排序）
    pub distance_sq: f32,
    /// 缓存的 BindGroup ID（快速路径）
    pub cached_bind_group_id: Option<u64>,
    /// 缓存的 Pipeline ID（快速路径）
    pub cached_pipeline_id: Option<u16>,
}

/// 提取的骨骼数据
#[derive(Clone)]
pub struct ExtractedSkeleton {
    pub skeleton_key: SkeletonKey,
}

/// 提取的场景数据
/// 
/// 这是一个轻量级的结构，只包含当前帧渲染所需的最小数据集。
/// 在 Extract 阶段填充，之后可以安全地释放 Scene 的借用。
pub struct ExtractedScene {
    /// 可见的渲染项列表（已经过视锥剔除）
    pub render_items: Vec<ExtractedRenderItem>,
    /// 需要上传的骨骼数据
    pub skeletons: Vec<ExtractedSkeleton>,
    /// 背景颜色
    pub background: Option<glam::Vec4>,
    /// 场景特性标志
    pub scene_features: SceneFeatures,

    pub scene_id: u32,
}

impl ExtractedScene {
    /// 创建空的提取场景
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            skeletons: Vec::new(),
            background: None,
            scene_features: SceneFeatures::empty(),
            scene_id: 0,
        }
    }

    /// 预分配容量
    pub fn with_capacity(item_capacity: usize, skeleton_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            skeletons: Vec::with_capacity(skeleton_capacity),
            background: None,
            scene_features: SceneFeatures::empty(),
            scene_id: 0,
        }
    }

    /// 清空数据以便重用
    pub fn clear(&mut self) {
        self.render_items.clear();
        self.skeletons.clear();
        self.scene_id = 0;
    }

    /// 复用当前实例的内存，从 Scene 中提取数据
    /// 
    /// 这是性能优化的关键：避免每帧分配新内存
    pub fn extract_into(&mut self, scene: &Scene, camera: &Camera, assets: &AssetServer) {
        self.clear();
        self.extract_render_items(scene, camera, assets);
        self.extract_skeletons(scene);
        self.extract_environment(scene);
    }

    /// 提取可见的渲染项
    fn extract_render_items(&mut self, scene: &Scene, camera: &Camera, assets: &AssetServer) {
        let frustum = camera.frustum;
        let camera_pos = camera.world_matrix.translation;

        for (node_handle, node) in scene.nodes.iter() {
            // 跳过不可见节点
            if !node.visible {
                continue;
            }

            // 获取 Mesh 组件（从组件存储中查询）
            let Some(&mesh_key) = scene.meshes.get(node_handle) else {
                continue;
            };
            let Some(mesh) = scene.mesh_pool.get(mesh_key) else {
                continue;
            };

            // 跳过不可见的 Mesh
            if !mesh.visible {
                continue;
            }

            let geo_handle = mesh.geometry;
            let mat_handle = mesh.material;

            // 验证资源存在性（安全检查）
            let Some(geometry) = assets.get_geometry(geo_handle) else {
                log::warn!("Node {:?} refers to missing Geometry {:?}", node_handle, geo_handle);
                continue;
            };

            // 材质可以稍后检查，但最好在这里也验证
            if assets.get_material(mat_handle).is_none() {
                log::warn!("Node {:?} refers to missing Material {:?}", node_handle, mat_handle);
                continue;
            }

            let node_world = node.transform.world_matrix;
            let world_matrix = Mat4::from(node_world);

            // 视锥剔除
            if let Some(bs) = geometry.bounding_sphere.borrow().as_ref() {
                let scale = node.transform.scale.max_element();
                let center = node_world.transform_point3(bs.center);
                if !frustum.intersects_sphere(center, bs.radius * scale) {
                    continue;
                }
            }

            // 计算到相机的距离（用于排序）
            let distance_sq = camera_pos.distance_squared(node_world.translation);

            // 从 Mesh 的渲染缓存中获取缓存的 ID
            let cached_bind_group_id = mesh.render_cache.bind_group_id;
            let cached_pipeline_id = mesh.render_cache.pipeline_id;

            // 获取骨骼绑定（从组件存储中查询）
            let skeleton = scene.skins.get(node_handle).map(|skin| skin.skeleton);

            self.render_items.push(ExtractedRenderItem {
                node_handle,
                mesh_key,
                world_matrix,
                geometry: geo_handle,
                material: mat_handle,
                skeleton,
                distance_sq,
                cached_bind_group_id,
                cached_pipeline_id,
            });
        }
    }

    /// 提取骨骼数据
    fn extract_skeletons(&mut self, scene: &Scene) {
        for (skel_key, _skeleton) in &scene.skeleton_pool {
            self.skeletons.push(ExtractedSkeleton {
                skeleton_key: skel_key,
            });
        }
    }

    /// 提取环境数据
    fn extract_environment(&mut self, scene: &Scene) {
        self.background = scene.background;
        self.scene_features = scene.get_features();
        self.scene_id = scene.id;
    }

    /// 获取渲染项数量
    #[inline]
    pub fn render_item_count(&self) -> usize {
        self.render_items.len()
    }

    /// 检查是否为空
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
        // 确保结构体大小合理
        let size = std::mem::size_of::<ExtractedRenderItem>();
        println!("ExtractedRenderItem size: {} bytes", size);
        // 应该在合理范围内（不超过 256 bytes）
        assert!(size < 256, "ExtractedRenderItem is too large: {} bytes", size);
    }
}
