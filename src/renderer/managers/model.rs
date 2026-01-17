//! 模型变换矩阵管理器
//!
//! 管理动态 Uniform Buffer 和 Object BindGroup
//! 
//! # 优化策略
//! - 使用紧凑的缓存键减少 HashMap 查找开销
//! - 区分静态和动态几何体的缓存策略
//! - 支持预缓存的 BindGroup ID 以减少每帧查找

use wgpu::ShaderStages;
use rustc_hash::FxHashMap;

use crate::Mesh;
use crate::renderer::core::builder::WgslStructName;
use crate::renderer::managers::skeleton::SkeletonManager;
use crate::renderer::core::resources::ResourceManager;
use crate::resources::uniforms::{DynamicModelUniforms, MorphUniforms};
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::renderer::core::builder::ResourceBuilder;
use crate::renderer::core::binding::Bindings;
use crate::assets::GeometryHandle;
use crate::resources::buffer::CpuBuffer;
use crate::scene::{SkeletonKey};
use crate::scene::skeleton::SkinBinding;

/// 紧凑的 BindGroup 缓存键
/// 
/// 优化点：使用更小的键结构减少哈希计算开销
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObjectBindGroupKey {
    /// 几何体句柄（对于静态几何体为 None）
    geo_id: Option<GeometryHandle>,
    /// Model Buffer 的 ID（用于检测 buffer 重建）
    model_buffer_id: u64,
    /// 骨骼 ID（如果使用蒙皮）
    skeleton_id: Option<SkeletonKey>,
}

/// 预缓存的 BindGroup 查找键
/// 
/// 用于在 RenderItem 中存储，避免每帧重新计算完整的缓存键
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CachedBindGroupId {
    pub bind_group_id: u64,
    pub model_buffer_id: u64,
}

impl CachedBindGroupId {
    /// 检查缓存是否仍然有效
    #[inline]
    pub fn is_valid(&self, current_model_buffer_id: u64) -> bool {
        self.model_buffer_id == current_model_buffer_id
    }
}

pub struct ModelManager {
    model_buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    current_capacity: usize,
    cache: FxHashMap<ObjectBindGroupKey, ObjectBindingData>,
    /// 按 BindGroup ID 的快速查找表
    id_lookup: FxHashMap<u64, ObjectBindingData>,
}

#[derive(Clone)]
pub struct ObjectBindingData {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub binding_wgsl: String,
    /// 用于快速验证缓存有效性
    pub cached_id: CachedBindGroupId,
}

impl ModelManager {
    pub fn new(resource_manager: &mut ResourceManager) -> Self {
        let initial_capacity = 128;
        let initial_data = vec![DynamicModelUniforms::default(); initial_capacity];

        let model_buffer = CpuBuffer::new(
            initial_data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer")
        );

        resource_manager.write_buffer(model_buffer.handle(), model_buffer.as_bytes());

        Self {
            model_buffer,
            current_capacity: initial_capacity,
            cache: FxHashMap::default(),
            id_lookup: FxHashMap::default(),
        }
    }

    /// 获取当前 Model Buffer 的 ID，用于缓存验证
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_buffer.handle().id
    }

    /// 通过缓存的 ID 快速获取 BindGroup 数据
    /// 
    /// 这是一个 O(1) 查找，比完整的 prepare_bind_group 更快
    #[inline]
    pub fn get_cached_bind_group(&self, cached_id: CachedBindGroupId) -> Option<&ObjectBindingData> {
        if cached_id.is_valid(self.model_buffer_id()) {
            self.id_lookup.get(&cached_id.bind_group_id)
        } else {
            None
        }
    }

    pub fn write_uniforms(&mut self, resource_manager: &mut ResourceManager, data: Vec<DynamicModelUniforms>) {
        if data.is_empty() { return; }

        if data.len() > self.current_capacity {
            let new_cap = (data.len() * 2).max(128);
            log::info!("Model Buffer expanding capacity: {} -> {}", self.current_capacity, new_cap);

            let mut new_data = data;
            new_data.resize(new_cap, DynamicModelUniforms::default());

            self.model_buffer = CpuBuffer::new(
                new_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer")
            );
            self.current_capacity = new_cap;
            // Buffer 重建后，所有缓存都失效
            self.cache.clear();
            self.id_lookup.clear();
        } else {
            *self.model_buffer.write() = data;
        }

        resource_manager.write_buffer(self.model_buffer.handle(), self.model_buffer.as_bytes());
    }

    pub fn prepare_bind_group(
        &mut self,
        resource_manager: &mut ResourceManager,
        skeleton_manager: &SkeletonManager,
        geometry_handle: GeometryHandle,
        geometry: &Geometry,
        mesh: &Mesh,
        skin_binding: Option<&SkinBinding>,
    ) -> ObjectBindingData {
        let features = geometry.get_features();
        let is_static = !features.intersects(GeometryFeatures::USE_MORPHING | GeometryFeatures::USE_SKINNING);

        let has_skin_binding = skin_binding.is_some();
        let geo_supports_skinning = features.contains(GeometryFeatures::USE_SKINNING);
        let use_skinning = geo_supports_skinning && has_skin_binding;

        let skeleton_id = if use_skinning {
            skin_binding.map(|s| s.skeleton)
        } else {
            None
        };

        let key = ObjectBindGroupKey {
            geo_id: if is_static { None } else { Some(geometry_handle) },
            model_buffer_id: self.model_buffer.handle().id,
            skeleton_id,
        };

        if let Some(binding_data) = self.cache.get(&key) {
            return binding_data.clone();
        }

        let mut builder = ResourceBuilder::new();

        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model",
            &self.model_buffer,
            std::mem::size_of::<DynamicModelUniforms>() as u64,
            ShaderStages::VERTEX
        );

        mesh.define_bindings(&mut builder);
        
        let use_morphing = features.contains(GeometryFeatures::USE_MORPHING);

        if use_morphing {
            builder.add_uniform::<MorphUniforms>(
                "morph_targets",
                &mesh.morph_uniforms,
                ShaderStages::VERTEX
            );
        }

        geometry.define_bindings(&mut builder);

        if use_skinning {
            if let Some(skel_id) = skeleton_id {
                if let Some(buffer) = skeleton_manager.get_buffer(skel_id) {
                    builder.add_storage_buffer(
                        "skins",
                        buffer.handle(),
                        buffer.as_bytes(),
                        true,
                        ShaderStages::VERTEX,
                        Some(WgslStructName::Name("mat4x4<f32>".into()))
                    );
                }
            }
        }

        let binding_wgsl = builder.generate_wgsl(2);
        let (layout, _layout_id) = resource_manager.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bind_group_id) = resource_manager.create_bind_group(&layout, &builder.resources);

        let cached_id = CachedBindGroupId {
            bind_group_id,
            model_buffer_id: self.model_buffer.handle().id,
        };

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl,
            cached_id,
        };

        // 同时更新两个缓存
        self.cache.insert(key, data.clone());
        self.id_lookup.insert(bind_group_id, data.clone());
        
        data
    }
}
