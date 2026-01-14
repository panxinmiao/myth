use std::collections::HashMap;
use wgpu::ShaderStages;

use crate::render::resources::manager::ResourceManager;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::resources::buffer::BufferRef;
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::render::resources::builder::ResourceBuilder;
use crate::render::resources::binding::Bindings;
use crate::assets::GeometryHandle;

// 缓存 Key：决定了 BindGroup 是否可以复用
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectBindGroupKey {
    geo_id: Option<GeometryHandle>,
    model_buffer_id: u64,   // 全局 Model Buffer ID (区分扩容)
}

pub struct ModelBufferManager {
    // === 数据源 ===
    model_buffer: BufferRef,
    last_model_buffer_id: u64,
    
    // === 缓存 ===
    // 我们缓存 (BindGroup, BindGroupId, Layout) 三元组，方便渲染时直接取用
    cache: HashMap<ObjectBindGroupKey, ObjectBindingData>,
}

#[derive(Clone)]
pub struct ObjectBindingData {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub binding_wgsl: String, 
}

impl ModelBufferManager {
    pub fn new(resource_manager: &mut ResourceManager) -> Self {
        let initial_capacity = 128;
        let model_buffer = BufferRef::with_capacity(
            initial_capacity * std::mem::size_of::<DynamicModelUniforms>(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer")
        );
        let initial_data = vec![DynamicModelUniforms::default(); initial_capacity];
        let bytes = bytemuck::cast_slice(&initial_data);
        let last_model_buffer_id = resource_manager.write_buffer(&model_buffer, bytes);

        Self {
            model_buffer,
            last_model_buffer_id,
            cache: HashMap::new(),
        }
    }

    /// 每一帧开始时调用：上传所有矩阵数据
    pub fn write_uniforms(&mut self, resource_manager: &mut ResourceManager, data: &[DynamicModelUniforms]) {
        if data.is_empty() { return; }

        let bytes = bytemuck::cast_slice(data);

        // 1. 如果长度超过当前 capacity，需要替换 BufferRef
        if bytes.len() > self.model_buffer.size() {
            // 扩容为当前所需的两倍，避免频繁扩容
            let new_cap = bytes.len() * 2;
            self.model_buffer = BufferRef::with_capacity(new_cap, self.model_buffer.usage(), self.model_buffer.label());
        }

        // 2. 同步 GPU (写入并获取 GPU 资源 ID)
        let new_id = resource_manager.write_buffer(&self.model_buffer, bytes);
        if new_id != self.last_model_buffer_id {
            // Buffer 重建了！所有 BindGroup 失效 (因为它们引用了旧 Buffer)
            log::info!("Model Buffer resized ({} -> {}), clearing ObjectBindGroup cache", self.last_model_buffer_id, new_id);
            self.cache.clear();
            self.last_model_buffer_id = new_id;
        }
    }

    /// 获取 Group 2 资源
    pub fn prepare_bind_group(
        &mut self,
        resource_manager: &mut ResourceManager,
        geometry_handle: GeometryHandle,
        geometry: &Geometry,
    ) -> ObjectBindingData {
        
        // 1. 计算 Key
        // 对于普通物体（无 Morph/Skin），我们可以认为它们共享同一个 "Static" 特征
        let features = geometry.get_features();
        
        // 如果是纯静态物体(无额外Bind)，我们可以让所有静态物体共享一个 Key，减少 Cache 大小
        let is_static = !features.intersects(GeometryFeatures::USE_MORPHING | GeometryFeatures::USE_SKINNING);
        
        let key = ObjectBindGroupKey {
            geo_id: if is_static {None } else { Some(geometry_handle) },
            model_buffer_id: self.last_model_buffer_id,
        };

        // 2. 查缓存
        if let Some(binding_data) = self.cache.get(&key) {
            return binding_data.clone();
        }

        // 3. 缓存未命中：开始构建
        
        // 3.1 收集 Bindings
        let mut builder = ResourceBuilder::new();
        
        // Binding 0: Model Matrix (Dynamic Uniform)
        // 复用 ResourceBuilder，它生成的 layout_entries 会自动兼容 RM 的缓存机制
        builder.add_dynamic_uniform::<DynamicModelUniforms>(
            "model", 
            &self.model_buffer, 
            std::mem::size_of::<DynamicModelUniforms>() as u64, 
            ShaderStages::VERTEX
        );

        // Binding 1..N: Geometry 贡献的 (Morph/Skin)
        // 我们需要 Geometry 的特定逻辑来填充 builder
        geometry.define_bindings(&mut builder);


        let binding_wgsl = builder.generate_wgsl(2); // Group 2

        // 我们不自己管理 Layout Cache，而是把 entries 扔给 RM，它会返回一个全局去重的 Layout
        let (layout, _layout_id) = resource_manager.get_or_create_layout(&builder.layout_entries);

        // 3.3 创建 BindGroup
        // 这里调用 RM 的工厂方法，它会处理 Buffer/Texture 的 GPU 句柄查找
        let (bind_group, bind_group_id) = resource_manager.create_bind_group(&layout, &builder.resources);

        let data = ObjectBindingData {
            layout,
            bind_group,
            bind_group_id,
            binding_wgsl,
        };

        // 4. 写入缓存
        self.cache.insert(key, data.clone());

        data
    }
}