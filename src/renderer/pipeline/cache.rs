//! 管线缓存
//!
//! L1 快速缓存 + L2 规范缓存
//!
//! # 缓存策略
//! 
//! - **L1 快速缓存**: 基于资源 Handle 和 Layout ID 的极快路径
//! - **L2 规范缓存**: 基于完整 Pipeline 特征的规范化缓存

use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_128;

use crate::renderer::core::ObjectBindingData;
use crate::resources::geometry::GeometryFeatures;
use crate::resources::material::MaterialFeatures;
use crate::scene::scene::SceneFeatures;
use crate::assets::{GeometryHandle, MaterialHandle};

use crate::renderer::pipeline::shader_gen::{ShaderGenerator, ShaderCompilationOptions};
use crate::renderer::pipeline::vertex::GeneratedVertexLayout;
use crate::renderer::core::resources::GpuMaterial;

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    pub scene_features: SceneFeatures,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<wgpu::BlendState>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
}

/// L1 缓存 Key: 基于资源 Handle 和物理 Layout ID (极快)
/// 
/// 使用 GPU 端的 layout_id 而非 CPU 端的 version，更精确地反映 Pipeline 兼容性
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastPipelineKey {
    pub material_handle: MaterialHandle,
    /// Material 的 BindGroupLayout ID（物理资源 ID）
    pub material_version: u64,
    pub material_layout_id: u64,
    pub geometry_handle: GeometryHandle,
    /// Geometry 的结构版本（影响 VertexLayout）
    pub geometry_layout_version: u64,
    /// 实例变体标志（如是否有骨骼）
    pub instance_variants: u32,
    /// 场景特性 ID
    pub scene_id: u32,
    /// 渲染状态 ID
    pub render_state_id: u32,
}

pub struct PipelineCache {
    fast_cache: FxHashMap<FastPipelineKey, (wgpu::RenderPipeline, u16)>,
    canonical_cache: FxHashMap<PipelineKey, (wgpu::RenderPipeline, u16)>,
    module_cache: FxHashMap<u128, wgpu::ShaderModule>,
    next_id: u16,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            fast_cache: FxHashMap::default(),
            canonical_cache: FxHashMap::default(),
            module_cache: FxHashMap::default(),
            next_id: 0,
        }
    }

    pub fn get_pipeline_fast(&self, fast_key: FastPipelineKey) -> Option<&(wgpu::RenderPipeline, u16)> {
        self.fast_cache.get(&fast_key)
    }

    pub fn insert_pipeline_fast(&mut self, fast_key: FastPipelineKey, pipeline: (wgpu::RenderPipeline, u16)) {
        self.fast_cache.insert(fast_key, pipeline);
    }

    pub fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        template_name: &str,
        canonical_key: PipelineKey,
        vertex_layout: &GeneratedVertexLayout,
        gpu_material: &GpuMaterial,
        object_data: &ObjectBindingData,
        global_binding_wgsl: &str,
        global_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::RenderPipeline, u16) {
        if let Some(cached) = self.canonical_cache.get(&canonical_key) {
            return cached.clone();
        }

        let options = ShaderCompilationOptions {
            mat_features: canonical_key.mat_features,
            geo_features: canonical_key.geo_features,
            scene_features: canonical_key.scene_features,
        };

        let shader_source = ShaderGenerator::generate_shader(
            vertex_layout,
            global_binding_wgsl,
            &gpu_material.binding_wgsl,
            &object_data.binding_wgsl,
            template_name,
            &options,
        );

        if cfg!(feature = "debug_shader") {
            fn normalize_newlines(s: &str) -> String {
                let mut result = String::with_capacity(s.len());
                let mut last_was_newline = false;

                for c in s.chars() {
                    if c == '\n' {
                        if !last_was_newline {
                            result.push('\n');
                            last_was_newline = true;
                        }
                    } else {
                        result.push(c);
                        last_was_newline = false;
                    }
                }

                result
            }
            
            println!("================= Generated Shader Code {} ==================\n {}", template_name, normalize_newlines(&shader_source));
        }

        let code_hash = xxh3_128(shader_source.as_bytes());

        let shader_module = self.module_cache.entry(code_hash).or_insert(
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Shader Module {}", template_name)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            })
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                global_layout,
                &gpu_material.layout,
                &object_data.layout
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> = vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Auto-Generated Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: canonical_key.color_format,
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
                format: canonical_key.depth_format,
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
        self.next_id = self.next_id.wrapping_add(1);

        self.canonical_cache.insert(canonical_key, (pipeline.clone(), id));

        (pipeline, id)
    }
}
