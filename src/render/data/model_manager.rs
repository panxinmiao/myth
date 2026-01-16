use wgpu::ShaderStages;
use rustc_hash::{FxHashMap};

use crate::render::resources::builder::WgslStructName;
use crate::render::data::skeleton_manager::{SkeletonManager};
use crate::render::resources::manager::ResourceManager;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::render::resources::builder::ResourceBuilder;
use crate::render::resources::binding::Bindings;
use crate::assets::GeometryHandle;
use crate::resources::buffer::CpuBuffer;
use crate::scene::SkeletonKey;
use crate::scene::skeleton::SkinBinding;

// 缓存 Key：决定了 BindGroup 是否可以复用
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectBindGroupKey {
    geo_id: Option<GeometryHandle>,
    model_buffer_id: u64,   // 全局 Model Buffer ID (区分扩容)

    skeleton_id: Option<SkeletonKey>,
}

pub struct ModelManager {
    // === 数据源 ===
    model_buffer: CpuBuffer<Vec<DynamicModelUniforms>>,
    
    // 记录当前的分配容量（元素个数），用于判断是否需要重建 Buffer
    current_capacity: usize,
    // === 缓存 ===
    // 我们缓存 (BindGroup, BindGroupId, Layout) 三元组，方便渲染时直接取用
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

        // 立即上传初始 Buffer 到 GPU，确保后续 prepare_bind_group 时 Buffer 已存在
        resource_manager.write_buffer(
            model_buffer.handle(), 
            model_buffer.as_bytes()
        );

        Self {
            model_buffer,
            current_capacity: initial_capacity,
            cache: FxHashMap::default(),
        }
    }

    /// 每一帧开始时调用：上传所有矩阵数据
    pub fn write_uniforms(&mut self, resource_manager: &mut ResourceManager, data: Vec<DynamicModelUniforms>) {
        if data.is_empty() { return; }

        // 1. 检查是否需要扩容
        if data.len() > self.current_capacity {
            let new_cap = (data.len() * 2).max(128);
            log::info!("Model Buffer expanding capacity: {} -> {}", self.current_capacity, new_cap);
            
            let mut new_data = data;
            new_data.resize(new_cap, DynamicModelUniforms::default());

            // 【关键】创建新的 CpuBuffer -> 生成新的 BufferRef ID
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

        // 3. 显式触发上传
        resource_manager.write_buffer(
            self.model_buffer.handle(), 
            self.model_buffer.as_bytes()
        );

    }

    /// 获取 Group 2 资源
    pub fn prepare_bind_group(
        &mut self,
        resource_manager: &mut ResourceManager,
        skeleton_manager: &SkeletonManager,
        geometry_handle: GeometryHandle,
        geometry: &Geometry,
        skin_binding: Option<&SkinBinding>,
    ) -> ObjectBindingData {
        
        // 1. 计算 Key
        // 对于普通物体（无 Morph/Skin），我们可以认为它们共享同一个 "Static" 特征
        let features = geometry.get_features();
        
        // 如果是纯静态物体(无额外Bind)，我们可以让所有静态物体共享一个 Key，减少 Cache 大小
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
            // 直接使用 CpuBuffer 的 ID
            model_buffer_id: self.model_buffer.handle().id,

            skeleton_id,
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


        if use_skinning {
            if let Some(skel_id) = skeleton_id {
                if let Some(buffer) = skeleton_manager.get_buffer(skel_id) {
                    builder.add_buffer(
                        "skins",
                        buffer.handle(), 
                        Some(buffer.as_bytes()),
                        ShaderStages::VERTEX,
                        Some(WgslStructName::Name("array<mat4x4<f32>, 64>".into()))
                    );
                }
            }
        }


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