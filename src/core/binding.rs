// src/core/binding.rs
use uuid::Uuid;
use wgpu::ShaderStages;

/// 定义绑定的具体类型 (Schema)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// Uniform Buffer (通常用于材质参数，全局变量)
    UniformBuffer,
    
    /// Storage Buffer (只读/读写，用于骨骼矩阵、粒子等)
    StorageBuffer { read_only: bool },
    
    /// 纹理
    Texture {
        sample_type: wgpu::TextureSampleType,
        view_dimension: wgpu::TextureViewDimension,
        multisampled: bool,
    },
    
    /// 采样器
    Sampler {
        type_: wgpu::SamplerBindingType,
    },
}

/// 单个绑定槽位的描述符 (用于生成 Layout)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingDescriptor {
    /// Shader 中的变量名 (主要用于调试和 Shader生成器的变量映射)
    /// 例如: "map" -> "t_map", "s_map"
    pub name: &'static str,
    
    /// 绑定槽位索引 (binding index)
    pub index: u32,
    
    /// 绑定类型
    pub bind_type: BindingType,
    
    /// 可见性
    pub visibility: ShaderStages,
}

/// 实际的绑定资源数据 (用于生成 BindGroup)
/// Core 层只持有 ID 或 数据引用，不持有 GPU 句柄
#[derive(Debug, Clone)]
pub enum BindingResource<'a> {
    /// Buffer 数据 (直接提供 bytes，由 Renderer 写入 GPU Buffer)
    Buffer(&'a [u8]),
    
    /// Buffer ID (引用已存在的 GPU Buffer，如骨骼矩阵)
    BufferId(Uuid),
    
    /// 纹理 ID (可能为空，意味着需要使用缺省纹理)
    Texture(Option<Uuid>),
    
    /// 采样器 ID (通常跟随纹理，但也可以独立)
    Sampler(Option<Uuid>),
}

/// 核心 Trait：所有能被绑定的对象都要实现此接口
pub trait Bindable {
    /// 获取绑定的布局描述 (Schema)
    /// 这决定了 PipelineLayout 的结构
    fn get_bindings(&self) -> (Vec<BindingDescriptor>, Vec<BindingResource<'_>>);


    // fn get_binding_resources(&self) -> Vec<BindingResource<'_>>;
}