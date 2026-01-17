//! 模型变换矩阵管理器
//!
//! 管理动态 Uniform Buffer 和 Object BindGroup

use wgpu::ShaderStages;
use rustc_hash::FxHashMap;

use crate::renderer::core::builder::WgslStructName;
use crate::renderer::managers::skeleton::SkeletonManager;
use crate::renderer::core::resources::ResourceManager;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::renderer::core::builder::ResourceBuilder;
use crate::renderer::core::binding::Bindings;
use crate::assets::GeometryHandle;
use crate::resources::buffer::CpuBuffer;
use crate::scene::SkeletonKey;
use crate::scene::skeleton::SkinBinding;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectBindGroupKey {
    geo_id: Option<GeometryHandle>,
    model_buffer_id: u64,
    skeleton_id: Option<SkeletonKey>,
}

pub struct ModelManager {
    model_buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    current_capacity: usize,
    cache: FxHashMap<ObjectBindGroupKey, ObjectBindingData>,
}

#[derive(Clone)]
pub struct ObjectBindingData {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub binding_wgsl: String,
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
            self.cache.clear();
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

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl,
        };

        self.cache.insert(key, data.clone());
        data
    }
}
