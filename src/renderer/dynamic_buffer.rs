use std::sync::{Arc};
use wgpu::{BufferUsages, ShaderStages};
use crate::renderer::resource_manager::ResourceManager;
use crate::core::uniforms::DynamicModelUniforms;
use crate::core::buffer::{DataBuffer, BufferRef};
use crate::core::binding::{ResourceBuilder};

/// 管理动态 Uniform Buffer (Group 2)
/// 职责：
/// 1. CPU 端：收集每一帧的 Model Uniforms 数据
/// 2. GPU 端：利用 ResourceManager 自动管理 Buffer 生命周期
/// 3. Binding：维护支持 Dynamic Offset 的 BindGroup
pub struct DynamicBuffer {
    label: String,
    
    // CPU 端数据源
    pub cpu_buffer: BufferRef,
    
    // 渲染需要的资源
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,

    pub layout: Arc<wgpu::BindGroupLayout>,

    last_buffer_id: u64,
}

impl DynamicBuffer {
    pub fn new(resource_manager: &mut ResourceManager, label: &str) -> Self {
        // 1. 初始化 CPU 数据 (预分配一些空间，比如 128 个物体)
        let initial_capacity = 128;
        let initial_data = vec![DynamicModelUniforms::default(); initial_capacity];
        
        let cpu_buffer = BufferRef::new(DataBuffer::new(
            &initial_data,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            Some(label)
        ));
        // 2. 立即在 GPU 上准备 Buffer (这会在 RM 中注册并上传数据)
        let buffer_id = resource_manager.prepare_buffer(&cpu_buffer.read());

        let (layout, bind_group, bg_id) = Self::recreate_resources(resource_manager, &cpu_buffer);

        Self {
            label: label.to_string(),
            cpu_buffer,
            bind_group,
            bind_group_id: bg_id,
            layout,
            last_buffer_id: buffer_id,
        }
    }

    /// 每一帧调用：上传数据，如果容量不足则自动扩容
    pub fn write_and_expand(&mut self, resource_manager: &mut ResourceManager, data: &[DynamicModelUniforms]) {
        if data.is_empty() { return; }

        // 1. 更新 CPU Buffer
        // Vec 的自动扩容由 CpuBuffer::update 内部处理
        self.cpu_buffer.write().update(data);

        // 2. 委托 RM 同步 GPU Buffer
        let buffer_ref = self.cpu_buffer.read();
        let new_buffer_id = resource_manager.prepare_buffer(&buffer_ref);

        // 3. 检查物理 Buffer 是否发生了变化 (扩容导致重建)
        if new_buffer_id != self.last_buffer_id {
            log::info!("Recreating BindGroup for {} (Buffer Resized)", self.label);
            
            // 重新生成 BindGroup
            let (layout, bind_group, bg_id) = Self::recreate_resources(resource_manager, &self.cpu_buffer);

            if !Arc::ptr_eq(&self.layout, &layout) {
                // 通常不会发生变化，警告？
                self.layout = layout;
            }
            self.bind_group = bind_group;
            self.bind_group_id = bg_id;
            self.last_buffer_id = new_buffer_id;
        }
    }
    
    // 内部 helper：根据 CpuBuffer 描述生成资源
    fn recreate_resources(
        rm: &mut ResourceManager, 
        cpu_buffer: &BufferRef,
    ) -> (Arc<wgpu::BindGroupLayout>, wgpu::BindGroup, u64) {

        let mut builder = ResourceBuilder::new();

        // 这里的 256 是 min_binding_size，对于动态 Uniform 通常是单个结构体大小
        // dynamic offset 已经在 add_dynamic_uniform 内部处理了 (has_dynamic_offset = true)
        builder.add_dynamic_uniform(
            "DynamicModel", 
            cpu_buffer, 
            std::mem::size_of::<DynamicModelUniforms>() as u64, // 自动计算大小
            ShaderStages::VERTEX
        );

        // 2. 调用 RM 的通用接口创建 Layout
        // 这里 DynamicBuffer 和 Material 享受同等的 Layout 缓存待遇
        let layout = rm.get_or_create_layout(&builder.layout_entries); 
        
        // 2. 创建 BindGroup
        let (bind_group, bg_id) = rm.create_bind_group(&layout, &builder.resources);
        
        (layout, bind_group, bg_id)
    }
}