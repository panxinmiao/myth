use std::collections::HashMap;
use md5;

use crate::resources::geometry::{Geometry, GeometryFeatures};
use crate::resources::material::{Material, MaterialFeatures};
use crate::scene::scene::{Scene, SceneFeatures};
use crate::assets::{GeometryHandle, MaterialHandle};

use crate::render::pipeline::shader_gen::ShaderGenerator;
use crate::render::pipeline::shader_gen::ShaderCompilationOptions;
use crate::render::pipeline::vertex::GeneratedVertexLayout;
use crate::render::resources::manager::GpuMaterial;
use crate::render::resources::manager::GpuEnvironment;
use crate::render::data::model_manager::ObjectBindingData;

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征 (慢，但唯一)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    // pub shader_hash: String,
    // 1. 逻辑特征 (Logic Features)
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    pub scene_features: SceneFeatures,

    // 2. 渲染状态 (Render States)
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<wgpu::BlendState>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,

    // 是否需要 BindGroup Layout 的变化？
    // BindGroupLayout 的结构通常由 Features 决定, 所以只要 Features 相同，Layout 结构大概率相同
    // pub layout_ids: Vec<wgpu::Id<wgpu::BindGroupLayout>>,
}

/// L1 缓存 Key: 基于 ID 和版本号 (极快，无堆分配)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastPipelineKey {
    pub material_handle: MaterialHandle,
    pub material_version: u64,
    pub geometry_handle: GeometryHandle, 
    pub geometry_version: u64, 

    pub render_state_id: u32,
    pub scene_id: u32,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderModuleKey {
    code_hash: String, // 代码内容的哈希
    stage: wgpu::ShaderStages, // Vertex 或 Fragment
}

pub struct PipelineCache {
    // L1: 快速查找 (命中率 99%+)
    fast_cache: HashMap<FastPipelineKey, (wgpu::RenderPipeline, u16)>,
    // L2: 规范查找 (用于去重)
    canonical_cache: HashMap<PipelineKey, (wgpu::RenderPipeline, u16)>,
    // Shader Module 缓存 (避免重复创建)
    module_cache: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    next_id: u16, // 下一个 Pipeline ID
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            fast_cache: HashMap::new(),
            canonical_cache: HashMap::new(),
            module_cache: HashMap::new(),
            next_id: 0,
        }
    }

    // 获取或创建 Module 的辅助函数
    fn create_shader_module(
        &mut self, 
        device: &wgpu::Device, 
        code: &str, 
        stage: wgpu::ShaderStages,
        label: Option<&str>,
    ) -> wgpu::ShaderModule {

        let code_hash = format!("{:x}", md5::compute(code));
        let key = ShaderModuleKey {
            code_hash,
            stage,
        };

        if let Some(module) = self.module_cache.get(&key) {
            return module.clone();
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });

        self.module_cache.insert(key, module.clone());
        module
    }


    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,

        fast_key: FastPipelineKey,

        geometry: &Geometry,
        material: &Material,
        scene: &Scene,

        vertex_layout: &GeneratedVertexLayout, 
        gpu_material: &GpuMaterial, 
        object_data: &ObjectBindingData,
        gpu_environment: &GpuEnvironment,

        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat, 
    ) -> (wgpu::RenderPipeline, u16) {

        // =========================================================
        // 1. L1 Cache: 快速路径 (Fast Path)
        // =========================================================

        // 如果命中 L1 缓存，直接返回 (零堆内存分配)
        if let Some(cached) = self.fast_cache.get(&fast_key) {
            return cached.clone();
        }

        // =========================================================
        // 2. L2 Cache: 规范路径 (Canonical Path)
        // =========================================================

        // 2.1 收集 Features (极快，位运算)
        let mat_features = material.get_features();
        let geo_features = geometry.get_features();
        let scene_features = scene.get_features();
        let topology = geometry.topology;

        // 2.2 构建 Canonical Key (无堆分配，如果 bitflags 和 enum 都是 copy)
        let canonical_key = PipelineKey {
            mat_features,
            geo_features,
            scene_features,
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
            sample_count: 1, // 暂时写死
        };

        // 2.3 查 L2 缓存
        if let Some(cached) = self.canonical_cache.get(&canonical_key) {
            // L2 命中！更新 L1 并返回
            self.fast_cache.insert(fast_key, cached.clone());
            return cached.clone();
        }

        // =========================================================
        // 3. Cache Miss - 生成 Shader 代码并编译 (慢路径)
        // =========================================================

        // 0. 收集编译选项 (Material + Geometry)
        let options = ShaderCompilationOptions {
            mat_features,
            geo_features,
            scene_features,
        };

        let template_name = material.shader_name();

        // 2. Generate Code
        let shader_source = ShaderGenerator::generate_shader(
            &vertex_layout,
            &gpu_environment.binding_wgsl,
            &gpu_material.binding_wgsl,
            &object_data.binding_wgsl,
            &template_name,
            &options,
        );

        // Debug 输出
        if cfg!(feature = "debug_shader") {
            println!("================= Generated Shader Code {} ==================\n {}", template_name, shader_source);
        }

        // 3.3 创建 Modules
        let shader_module = self.create_shader_module(device, &shader_source, wgpu::ShaderStages::VERTEX, Some(template_name));

        // 自动创建 PipelineLayout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &gpu_environment.layout,
                &gpu_material.layout, 
                &object_data.layout 
            ],
            immediate_size: 0,
        });

        // 3.3 创建 Pipeline
        let vertex_buffers_layout: Vec<_> = vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Auto-Generated Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
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

        let id = self.next_id;

        // 注意：RenderKey 只分配了 14 位 (16384)，超过这个值会导致排序 key 冲突，但不影响渲染正确性
        self.next_id = self.next_id.wrapping_add(1);

        let result = (pipeline, id);

        // 4. 同时存入 L1 和 L2 缓存
        self.canonical_cache.insert(canonical_key, result.clone());
        self.fast_cache.insert(fast_key, result.clone());

        result
    }

}