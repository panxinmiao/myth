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

use std::collections::HashSet;

use glam::Mat4;

use crate::renderer::core::{BindGroupContext, ResourceManager};
use crate::resources::shader_defines::ShaderDefines;
use crate::scene::environment::Environment;
use crate::scene::{NodeHandle, Scene, SkeletonKey};
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::scene::camera::RenderCamera;

/// 精简的渲染项，只包含 GPU 需要的数据
/// 
/// 使用 Clone 而非 Copy，因为 SkinBinding 包含非 Copy 类型
#[derive(Clone)]
pub struct ExtractedRenderItem {
    /// 节点句柄（用于调试和回写缓存）
    pub node_handle: NodeHandle,
    /// 世界变换矩阵 (64 bytes)
    pub world_matrix: Mat4,

    pub object_bind_group: BindGroupContext,
    /// 几何体句柄 (8 bytes)
    pub geometry: GeometryHandle,
    /// 材质句柄 (8 bytes)
    pub material: MaterialHandle,

    pub item_variant_flags: u32,

    pub item_shader_defines: ShaderDefines,

    /// 到相机的距离平方（用于排序）
    pub distance_sq: f32,

    // 缓存的 BindGroup ID（快速路径）
    // pub cached_bind_group_id: Option<u64>,
    // /// 缓存的 Pipeline ID（快速路径）
    // pub cached_pipeline_id: Option<u16>,
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
    /// 场景的 Shader 宏定义
    pub scene_defines: ShaderDefines,
    pub scene_id: u32,
    pub background: Option<glam::Vec4>,
    pub envvironment: Environment,

    collected_meshes: Vec<CollectedMesh>,
    collected_skeleton_keys: HashSet<SkeletonKey>,

}

struct CollectedMesh {
    pub node_handle: NodeHandle,
    pub skeleton: Option<SkeletonKey>,
}

impl ExtractedScene {
    /// 创建空的提取场景
    pub fn new() -> Self {
        Self {
            render_items: Vec::new(),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            collected_meshes: Vec::new(),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// 预分配容量
    pub fn with_capacity(item_capacity: usize) -> Self {
        Self {
            render_items: Vec::with_capacity(item_capacity),
            scene_defines: ShaderDefines::new(),
            scene_id: 0,
            background: None,
            envvironment: Environment::default(),
            collected_meshes: Vec::with_capacity(item_capacity),
            collected_skeleton_keys: HashSet::default(),
        }
    }

    /// 清空数据以便重用
    pub fn clear(&mut self) {
        self.render_items.clear();
        // self.skeletons.clear();
        self.scene_defines.clear();
        self.scene_id = 0;


        self.collected_meshes.clear();
        self.collected_skeleton_keys.clear();
    }

    /// 复用当前实例的内存，从 Scene 中提取数据
    pub fn extract_into(&mut self, scene: &mut Scene, camera: &RenderCamera, assets: &AssetServer, resource_manager: &mut ResourceManager) {
        self.clear();
        self.extract_render_items(scene, camera, assets , resource_manager);
        self.extract_environment(scene);
    }

    /// 提取可见的渲染项
    fn extract_render_items(&mut self, scene: &mut Scene, camera: &RenderCamera, assets: &AssetServer, resource_manager: &mut ResourceManager) {
        let frustum = camera.frustum;
        let camera_pos = camera.position;

        // let mut collected_meshes = Vec::new();
        // let mut collected_skeleton_keys = HashSet::new();

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
            let mat_handle = mesh.material;

            let Some(geometry) = assets.get_geometry(geo_handle) else {
                log::warn!("Node {:?} refers to missing Geometry {:?}", node_handle, geo_handle);
                continue;
            };

            if assets.get_material(mat_handle).is_none() {
                log::warn!("Node {:?} refers to missing Material {:?}", node_handle, mat_handle);
                continue;
            }

            let node_world = node.transform.world_matrix;
            let skin_binding = scene.skins.get(node_handle);

            // 视锥剔除：根据是否有骨骼绑定选择不同的包围盒
            let is_visible = if let Some(binding) = skin_binding {
                // 有骨骼绑定：使用 Skeleton 的包围盒
                if let Some(skeleton) = scene.skeleton_pool.get(binding.skeleton) {
                    if let Some(local_bounds) = skeleton.local_bounds() {
                        let world_bounds = local_bounds.transform(&node_world);
                        frustum.intersects_box(world_bounds.min, world_bounds.max)
                    } else {
                        // 包围盒尚未计算，默认可见
                        true
                    }
                } else {
                    true
                }
            } else {
                // 无骨骼绑定：使用 Geometry 的包围盒
                if let Some(bbox) = geometry.bounding_box.borrow().as_ref() {
                    let world_bounds = bbox.transform(&node_world);
                    frustum.intersects_box(world_bounds.min, world_bounds.max)
                } else if let Some(bs) = geometry.bounding_sphere.borrow().as_ref() {
                    // 回退到包围球
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

        // 准备骨骼数据
        for skeleton_key in &self.collected_skeleton_keys {
            if let Some(skeleton) = scene.skeleton_pool.get(*skeleton_key){
                resource_manager.prepare_skeleton(skeleton);
            }
        }

        // 确保模型缓冲区容量
        resource_manager.ensure_model_buffer_capacity(self.collected_meshes.len());

        // 更新并填充渲染项
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

            self.render_items.push(ExtractedRenderItem {
                node_handle,
                world_matrix,
                object_bind_group,
                geometry: mesh.geometry,
                material: mesh.material,
                item_variant_flags: {
                    // if mesh.morph_targets.len() > 0 {
                    //     flags |= crate::renderer::pipeline::PipelineItemVariants::HAS_MORPH_TARGETS as u32;
                    // }
                    // Todo: Define the flags properly
                    skeleton.is_some() as u32
                },
                item_shader_defines,
                distance_sq,
            });
        }
    }



    /// 提取环境数据
    fn extract_environment(&mut self, scene: &Scene) {
        // self.background = scene.background;
        self.scene_defines = scene.shader_defines();
        self.scene_id = scene.id;
        self.envvironment = scene.environment.clone();
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
