use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use md5;

use crate::core::geometry::{Geometry, GeometryFeatures};
use crate::core::material::{Material, MaterialFeatures};
use crate::core::scene::SceneFeatures;
use crate::renderer::shader_generator::ShaderGenerator;
use crate::renderer::shader_generator::ShaderContext;
use crate::renderer::resource_manager::{generate_resource_id};
use crate::renderer::shader_generator::ShaderCompilationOptions;
use crate::renderer::vertex_layout::GeneratedVertexLayout;
use crate::renderer::resource_manager::GPUMaterial;
use crate::renderer::object_manager::ObjectBindingData;

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征 (慢，但唯一)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    // pub shader_hash: String,
    // 1. 逻辑特征 (Logic Features)
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    pub scene_features: SceneFeatures,

    // [新增] 数值特征 (Numerical Features)
    // 直接影响 Shader 宏定义，如 #define NUM_POINT_LIGHTS 3
    pub num_dir_lights: u8,
    pub num_point_lights: u8,
    pub num_spot_lights: u8,

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
struct FastPipelineKey {
    material_id: Uuid,
    material_version: u64,
    geometry_id: Uuid,     // 逻辑 ID
    geometry_version: u64, // 布局版本
    topology: wgpu::PrimitiveTopology, // 拓扑结构也可能变化

    scene_hash: u64, // 场景状态哈希
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
    fn create_shader_module(
        &mut self, 
        device: &wgpu::Device, 
        code: &str, 
        stage: wgpu::ShaderStages,
        label: Option<&str>,
    ) -> Arc<wgpu::ShaderModule> {

        let code_hash = format!("{:x}", md5::compute(code));
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
        geometry: &Geometry,
        scene: &crate::core::scene::Scene,

        gpu_material: &GPUMaterial, 
        object_data: &ObjectBindingData,
        global_layout: &wgpu::BindGroupLayout,

        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat, 
        vertex_layout: &GeneratedVertexLayout, 
    ) -> (Arc<wgpu::RenderPipeline>, u64) { // 返回 Arc 更好管理生命周期
        
        // 提取场景特征
        let (scene_features, num_dir, num_point, num_spot) = scene.get_render_stats();

        let scene_hash = (num_dir as u64) 
            | ((num_point as u64) << 8) 
            | ((num_spot as u64) << 16) 
            | ((scene_features.bits() as u64) << 24);

        // =========================================================
        // 1. L1 Cache: 快速路径 (Fast Path)
        // =========================================================
        let topology = geometry.topology;

        let fast_key = FastPipelineKey {
            material_id: material.id,
            material_version: material.version,
            geometry_id: geometry.id,
            geometry_version: geometry.version,
            topology,
            scene_hash,
        };

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

        // 2.2 构建 Canonical Key (无堆分配，如果 bitflags 和 enum 都是 copy)
        let canonical_key = PipelineKey {
            mat_features,
            geo_features,
            scene_features,
            num_dir_lights: num_dir,
            num_point_lights: num_point,
            num_spot_lights: num_spot,
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
            num_dir_lights: num_dir,
            num_point_lights: num_point,
            num_spot_lights: num_spot,
        };
        
        // 1. Context
        let mut base_context = ShaderContext::new();

        // 注入 defines
        for (k, v) in options.to_defines() {
            base_context = base_context.set_value(&k, v);
        }

        // if material.shader_name() == "MeshStandard" {
        //     base_context = base_context.define("USE_PBR", true);
        // }

        // 2. Generate Code
        let vs_code = ShaderGenerator::generate_vertex(
            &base_context,
            vertex_layout,
            &object_data.binding_wgsl,
            "mesh_basic.wgsl"
        );
        let fs_code = ShaderGenerator::generate_fragment(
            &base_context,
            &gpu_material.binding_wgsl,
            material.shader_name()
        );

        // Debug 输出
        if cfg!(debug_assertions) {
            println!("=== Vertex Shader ===\n{}", vs_code);
            println!("=== Fragment Shader ===\n{}", fs_code);
        }

        // 3.3 创建 Modules
        let vs_module = self.create_shader_module(device, &vs_code, wgpu::ShaderStages::VERTEX, Some("Vertex Shader"));
        let fs_module = self.create_shader_module(device, &fs_code, wgpu::ShaderStages::FRAGMENT, Some("Fragment Shader"));

        // 自动创建 PipelineLayout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                global_layout,
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
                module: &vs_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
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

}