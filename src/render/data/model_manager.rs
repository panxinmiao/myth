use std::collections::HashMap;
use wgpu::ShaderStages;

use crate::render::resources::manager::ResourceManager;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::render::resources::builder::ResourceBuilder;
use crate::render::resources::binding::Bindings;
use crate::assets::GeometryHandle;
use crate::resources::buffer::CpuBuffer;

// 缓存 Key：决定了 BindGroup 是否可以复用
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectBindGroupKey {
    geo_id: Option<GeometryHandle>,
    model_buffer_id: u64,   // 全局 Model Buffer ID (区分扩容)
}

pub struct ModelBufferManager {
    // === 数据源 ===
    model_buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    
    // 记录当前的分配容量（元素个数），用于判断是否需要重建 Buffer
    current_capacity: usize,
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
        let initial_data = vec![DynamicModelUniforms::default(); initial_capacity];

        let model_buffer = CpuBuffer::new(
            initial_data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("GlobalModelBuffer")
        );

        // 立即上传初始 Buffer 到 GPU，确保后续 prepare_bind_group 时 Buffer 已存在
        resource_manager.write_buffer(
            model_buffer.handle(), 
            model_buffer.as_bytes()
        );

        Self {
            model_buffer,
            current_capacity: initial_capacity,
            cache: HashMap::new(),
        }
    }

    /// 每一帧开始时调用：上传所有矩阵数据
    pub fn write_uniforms(&mut self, resource_manager: &mut ResourceManager, data: &[DynamicModelUniforms]) {
        if data.is_empty() { return; }

        // 1. 检查是否需要扩容
        if data.len() > self.current_capacity {
            let new_cap = (data.len() * 2).max(128);
            log::info!("Model Buffer expanding capacity: {} -> {}", self.current_capacity, new_cap);
            
            // 为了保持 GPU Buffer 的大容量，我们需要用 padded 数据初始化新的 CpuBuffer
            let mut new_data = data.to_vec();
            new_data.resize(new_cap, DynamicModelUniforms::default());

            // 【关键】创建新的 CpuBuffer -> 生成新的 BufferRef ID
            self.model_buffer = CpuBuffer::new(
                new_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("GlobalModelBuffer")
            );
            self.current_capacity = new_cap;

            // ID 变了，所有引用旧 Buffer 的 BindGroup 都失效了
            self.cache.clear();
            
            // 新 Buffer，ResourceManager 会在 write_buffer 时自动处理创建
        } else {
            // 2. 容量足够：直接更新数据
            // 我们通过 write() 获取 Guard，将数据拷贝进去
            // 注意：CpuBuffer 内部的 Vec 长度会变成 data.len()。
            // 只要 data.len() <= current_capacity (即 GPU Buffer 的 size)，
            // ResourceManager 的 write_buffer 就能正常工作（部分写入）。
            
            let mut guard = self.model_buffer.write();
            *guard = data.to_vec(); // 这里会分配内存拷贝，但对于几十KB的数据量通常可接受
            
            // Guard Drop 时，version 会自动 +1
        }

        // 3. 显式触发上传
        // 这一步是必须的，因为 ModelBufferManager 的 BindGroup 缓存可能命中，
        // 导致 prepare_bind_group 不会被调用，或者 RM 不会检查这个资源。
        // 我们利用 ResourceManager 现有的 write_buffer (它支持盲写)
        resource_manager.write_buffer(
            self.model_buffer.handle(), 
            self.model_buffer.as_bytes()
        );

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
            geo_id: if is_static { None } else { Some(geometry_handle) },
            // 直接使用 CpuBuffer 的 ID
            model_buffer_id: self.model_buffer.handle().id,
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