use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use md5;

use crate::core::material::Material;
use crate::renderer::shader_generator::ShaderGenerator;
use crate::renderer::shader_generator::ShaderContext;
use crate::renderer::resource_manager::{generate_resource_id, GPUGeometry};

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征 (慢，但唯一)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub shader_hash: String,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<wgpu::BlendState>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,

    // 是否需要 BindGroup Layout 的变化？
    // 实际上，只要 Shader （shader_hash）代码不变，Layout 结构通常也不变？
    // pub layout_ids: Vec<wgpu::Id<wgpu::BindGroupLayout>>,
}

/// L1 缓存 Key: 基于 ID 和版本号 (极快，无堆分配)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
struct FastPipelineKey {
    material_id: Uuid,
    material_version: u64,
    geometry_id: Uuid,     // 逻辑 ID
    geometry_version: u64, // 布局版本
    topology: wgpu::PrimitiveTopology, // 拓扑结构也可能变化
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderModuleKey {
    code_hash: String, // 代码内容的哈希
    stage: wgpu::ShaderStages, // Vertex 或 Fragment
}

pub struct PipelineCache {
    // L1: 快速查找 (命中率 99%+)
    // 存储的是 Arc，方便克隆引用
    fast_cache: HashMap<FastPipelineKey, (Arc<wgpu::RenderPipeline>, u64)>,

    // L2: 规范查找 (用于去重)
    canonical_cache: HashMap<PipelineKey, (Arc<wgpu::RenderPipeline>, u64)>,

    // Shader Module 缓存 (避免重复创建)
    module_cache: HashMap<ShaderModuleKey, Arc<wgpu::ShaderModule>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            fast_cache: HashMap::new(),
            canonical_cache: HashMap::new(),
            module_cache: HashMap::new(),
        }
    }

    // 获取或创建 Module 的辅助函数
    fn get_or_create_module(
        &mut self, 
        device: &wgpu::Device, 
        code: String, 
        code_hash: String,
        stage: wgpu::ShaderStages,
        label: Option<&str>,
    ) -> Arc<wgpu::ShaderModule> {

        let key = ShaderModuleKey {
            code_hash,
            stage,
        };

        if let Some(module) = self.module_cache.get(&key) {
            return module.clone();
        }

        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(code.into()),
        }));

        self.module_cache.insert(key, module.clone());
        module
    }



    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        material: &Material,
        gpu_geometry: &GPUGeometry, 
        geometry_id: Uuid,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat, 
        bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> (Arc<wgpu::RenderPipeline>, u64) { // 返回 Arc 更好管理生命周期
        
        // =========================================================
        // 1. 快速路径 (Fast Path) - 热路径优化
        // =========================================================
        let topology = gpu_geometry.topology;

        let fast_key = FastPipelineKey {
            material_id: material.id,
            material_version: material.version,
            geometry_id,
            geometry_version: gpu_geometry.version,
            topology,
        };

        // 如果命中 L1 缓存，直接返回 (零堆内存分配)
        if let Some(cached) = self.fast_cache.get(&fast_key) {
            return cached.clone();
        }


        // =========================================================
        // 生成 Shader 代码
        // =========================================================
        
        // 1. Context
        let mut base_context = ShaderContext::new();
        if material.shader_name() == "MeshStandard" {
            base_context = base_context.define("USE_PBR", true);
        }

        // 2. Generate Code
        // 我们需要先生成代码，才能计算 shader_hash
        // 这比之前稍微慢一点点（因为每次 L1 Miss 都要生成字符串），但对于 L2 命中检查是必要的
        let vs_code = ShaderGenerator::generate_vertex(
            &base_context,
            &gpu_geometry.layout_info,
            "standard_vert.wgsl"
        );
        let fs_code = ShaderGenerator::generate_fragment(
            &base_context,
            material,
            material.shader_name()
        );

        // 计算代码 Hash
        let vs_hash = md5::compute(&vs_code); // 或者其他 hash
        let fs_hash = md5::compute(&fs_code);
        let shader_hash_str = format!("{:x}_{:x}", vs_hash, fs_hash);

        // 收集 Layout IDs 用于 Cache Key
        // wgpu::BindGroupLayout 没有直接暴露 ID，但在 Rust 中我们可以利用 global_id 方法或者仅仅依赖 shader_hash?
        // 实际上，只要 Shader 代码不变，Layout 结构通常也不变。
        // 为了安全起见，PipelineKey 应该包含 BlendState 等渲染状态。

        let canonical_key = PipelineKey {
            shader_hash: shader_hash_str.clone(),
            topology,
            cull_mode: material.cull_mode,
            depth_write: material.depth_write,
            depth_compare: if material.depth_test { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
            blend_state: if material.transparent {
                Some(wgpu::BlendState::ALPHA_BLENDING)
            } else {
                None
            },
            color_format,
            depth_format,
        };

        if let Some(cached) = self.canonical_cache.get(&canonical_key) {
            self.fast_cache.insert(fast_key, cached.clone());
            return cached.clone();
        }


        // =========================================================
        // 3. 创建 Pipeline (昂贵操作)
        // =========================================================

        let vs_hash_str = format!("{:x}", vs_hash);
        let fs_hash_str = format!("{:x}", fs_hash);
        let vs_module = self.get_or_create_module(device, vs_code, vs_hash_str, wgpu::ShaderStages::VERTEX, Some("Vertex Shader"));
        let fs_module = self.get_or_create_module(device, fs_code, fs_hash_str, wgpu::ShaderStages::FRAGMENT, Some("Fragment Shader"));

        // 自动创建 PipelineLayout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts,
            immediate_size: 0,
        });

        // 3.3 创建 Pipeline
        let vertex_layouts: Vec<_> = gpu_geometry.layout_info.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&canonical_key.shader_hash),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: canonical_key.blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: canonical_key.topology,
                cull_mode: canonical_key.cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: canonical_key.depth_write,
                depth_compare: canonical_key.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let id = generate_resource_id();
        let result = (Arc::new(pipeline), id);

        // 4. 同时存入 L1 和 L2 缓存
        self.canonical_cache.insert(canonical_key, result.clone());
        self.fast_cache.insert(fast_key, result.clone());

        result
    }


    // 只读获取
    pub fn get_pipeline(
        &self, 
        material: &Material, 
        gpu_geometry: &GPUGeometry, 
        geometry_id: uuid::Uuid,
        // ... 其他 Key 参数 ... (为了计算 L1 Key)
    ) -> Option<(&wgpu::RenderPipeline, u64)> {
        
        // 1. 尝试 L1 快速查找
        let topology = gpu_geometry.topology;
        let fast_key = FastPipelineKey {
            material_id: material.id,
            material_version: material.version,
            geometry_id,
            geometry_version: gpu_geometry.version,
            topology,
        };

        if let Some((arc, id)) = self.fast_cache.get(&fast_key) {
            return Some((arc.as_ref(), *id));
        }

        // 理论上 Render 阶段不应该 L1 Miss (因为 Prepare 阶段已经创建了)
        // 是否需要尝试 L2 查找？这里选择不尝试，直接返回 None
        None 
    }
}