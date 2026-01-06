use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::core::material::Material;
use crate::renderer::layout_generator::GeneratedLayout;
use crate::renderer::shader_generator::ShaderGenerator;
use crate::renderer::resource_manager::{generate_resource_id, GPUGeometry}; // 假设引入了 GPUGeometry

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征 (慢，但唯一)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub shader_hash: String,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<wgpu::BlendState>,
    pub layout_signature: String,
    pub depth_format: wgpu::TextureFormat, // 新增：深度格式也是 Pipeline 的一部分
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

pub struct PipelineCache {
    // L1: 快速查找 (命中率 99%+)
    // 存储的是 Arc，方便克隆引用
    fast_cache: HashMap<FastPipelineKey, (Arc<wgpu::RenderPipeline>, u64)>,

    // L2: 规范查找 (用于去重)
    canonical_cache: HashMap<PipelineKey, (Arc<wgpu::RenderPipeline>, u64)>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            fast_cache: HashMap::new(),
            canonical_cache: HashMap::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        material: &Material,
        gpu_geometry: &GPUGeometry, 
        geometry_id: Uuid,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat, 
        global_layout: &wgpu::BindGroupLayout,
        material_layout: &wgpu::BindGroupLayout,
        model_layout: &wgpu::BindGroupLayout,
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
        // 2. 慢速路径 (Slow Path) - 首次创建或状态变更
        // =========================================================
        
        // 2.1 构建完整的 PipelineKey
        let features = material.features();
        let geometry_layout = &gpu_geometry.layout_info;

        let canonical_key = PipelineKey {
            shader_hash: format!("{}_{:?}", material.shader_name(), features),
            topology,
            cull_mode: material.cull_mode,
            depth_write: material.depth_write,
            depth_compare: if material.depth_test { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
            blend_state: if material.transparent {
                Some(wgpu::BlendState::ALPHA_BLENDING)
            } else {
                None
            },
            layout_signature: geometry_layout.shader_code.clone(),
            depth_format, // 使用传入的格式
        };

        // 2.2 检查 L2 缓存 (去重：不同物体如果材质几何特征完全一致，复用 Pipeline)
        if let Some(cached) = self.canonical_cache.get(&canonical_key) {
            // 更新 L1 缓存以便下次快速命中
            self.fast_cache.insert(fast_key, cached.clone());
            return cached.clone();
        }

        // =========================================================
        // 3. 创建 Pipeline (昂贵操作)
        // =========================================================
        
        // 3.1 生成 Shader (提取为独立函数)
        let shader_source = Self::generate_shader_source(material, geometry_layout);
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&canonical_key.shader_hash),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // 3.2 创建 Layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[global_layout, material_layout, model_layout],
            immediate_size: 0,
        });

        // 3.3 创建 Pipeline
        let vertex_layouts: Vec<_> = geometry_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&canonical_key.shader_hash),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
                format: depth_format, // <--- 使用传入的 depth_format
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

    // 将 Shader 生成逻辑抽离，保持主流程整洁
    fn generate_shader_source(material: &Material, layout: &GeneratedLayout) -> String {
        
        let fragment_template = material.shader_name();

        let shader_code = ShaderGenerator::generate_shader(material, layout,fragment_template);
        
        shader_code
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