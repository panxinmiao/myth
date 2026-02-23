//! 管线缓存
//!
//! L1 快速缓存 + L2 规范缓存
//!
//! # 缓存策略
//!
//! - **L1 快速缓存**: 基于资源 Handle 和 Layout ID 的极快路径
//! - **L2 规范缓存**: 基于完整 Pipeline 特征的规范化缓存

use rustc_hash::FxHashMap;
use std::hash::Hash;
use xxhash_rust::xxh3::xxh3_128;

use crate::assets::{GeometryHandle, MaterialHandle};
use crate::renderer::core::BindGroupContext;

use crate::renderer::core::resources::{GpuGlobalState, GpuMaterial};
use crate::renderer::graph::context::FrameResources;
use crate::renderer::pipeline::shader_gen::{ShaderCompilationOptions, ShaderGenerator};
use crate::renderer::pipeline::vertex::GeneratedVertexLayout;

/// L2 缓存 Key: 完整描述 Pipeline 的所有特征
///
/// 使用 `shader_hash` 替代原来的三个 Features 枚举，
/// 实现更灵活的宏定义组合。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub shader_hash: u64,
    pub vertex_layout_id: u64,
    // [Global, Material, Object, Screen]
    pub bind_group_layout_ids: [u64; 4],
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub blend_state: Option<wgpu::BlendState>,
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
    pub alpha_to_coverage: bool,
    pub front_face: wgpu::FrontFace,
}

/// L1 缓存 Key: 基于资源 Handle 和物理 Layout ID (极快)
///
/// 使用 GPU 端的 `layout_id` 而非 CPU 端的 version，更精确地反映 Pipeline 兼容性
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastPipelineKey {
    /// Material 句柄
    pub material_handle: MaterialHandle,
    /// Material 的 version
    pub material_version: u64,

    /// 几何体句柄
    pub geometry_handle: GeometryHandle,
    /// Geometry 的 version
    pub geometry_version: u64,

    /// 实例变体标志（如是否有骨骼）
    pub instance_variants: u32,

    /// GPU World state ID
    pub global_state_id: u32,

    /// 轻量级场景变体标志（如是否启用 SSAO）
    pub scene_variants: u32,

    /// Pipeline settings version (HDR, MSAA changes)
    /// This ensures cache invalidation when these settings change
    pub pipeline_settings_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastShadowPipelineKey {
    pub material_handle: MaterialHandle,
    pub material_version: u64,
    pub geometry_handle: GeometryHandle,
    pub geometry_version: u64,
    pub instance_variants: u32,
    pub pipeline_settings_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShadowPipelineKey {
    pub shader_hash: u64,
    pub topology: wgpu::PrimitiveTopology,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_format: wgpu::TextureFormat,
    pub front_face: wgpu::FrontFace,
}

pub struct PipelineCache {
    fast_cache: FxHashMap<FastPipelineKey, (wgpu::RenderPipeline, u16)>,
    canonical_cache: FxHashMap<PipelineKey, (wgpu::RenderPipeline, u16)>,
    fast_shadow_cache: FxHashMap<FastShadowPipelineKey, wgpu::RenderPipeline>,
    canonical_shadow_cache: FxHashMap<ShadowPipelineKey, wgpu::RenderPipeline>,
    module_cache: FxHashMap<u128, wgpu::ShaderModule>,
    next_id: u16,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fast_cache: FxHashMap::default(),
            canonical_cache: FxHashMap::default(),
            fast_shadow_cache: FxHashMap::default(),
            canonical_shadow_cache: FxHashMap::default(),
            module_cache: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Clears all cached pipelines.
    ///
    /// Call this when render settings change (MSAA, HDR) that affect pipeline compatibility.
    /// Shader modules are preserved since they don't depend on these settings.
    pub fn clear(&mut self) {
        self.fast_cache.clear();
        self.canonical_cache.clear();
        self.fast_shadow_cache.clear();
        self.canonical_shadow_cache.clear();
        // Note: module_cache is NOT cleared since shader code doesn't depend on MSAA/HDR settings
        // This saves expensive shader recompilation
    }

    #[must_use]
    pub fn get_pipeline_fast(
        &self,
        fast_key: FastPipelineKey,
    ) -> Option<&(wgpu::RenderPipeline, u16)> {
        self.fast_cache.get(&fast_key)
    }

    pub fn insert_pipeline_fast(
        &mut self,
        fast_key: FastPipelineKey,
        pipeline: (wgpu::RenderPipeline, u16),
    ) {
        self.fast_cache.insert(fast_key, pipeline);
    }

    #[must_use]
    pub fn get_shadow_pipeline_fast(
        &self,
        fast_key: FastShadowPipelineKey,
    ) -> Option<&wgpu::RenderPipeline> {
        self.fast_shadow_cache.get(&fast_key)
    }

    pub fn insert_shadow_pipeline_fast(
        &mut self,
        fast_key: FastShadowPipelineKey,
        pipeline: wgpu::RenderPipeline,
    ) {
        self.fast_shadow_cache.insert(fast_key, pipeline);
    }

    #[allow(clippy::too_many_lines)]
    pub fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        template_name: &str,
        canonical_key: PipelineKey,
        options: &ShaderCompilationOptions,
        vertex_layout: &GeneratedVertexLayout,
        gpu_material: &GpuMaterial,
        object_bind_group: &BindGroupContext,
        gpu_world: &GpuGlobalState,
        frame_resources: &FrameResources,
    ) -> (wgpu::RenderPipeline, u16) {
        if let Some(cached) = self.canonical_cache.get(&canonical_key) {
            return cached.clone();
        }

        let binding_code = format!(
            "{}\n{}\n{}",
            &gpu_world.binding_wgsl, &gpu_material.binding_wgsl, &object_bind_group.binding_wgsl
        );

        let shader_source = ShaderGenerator::generate_shader(
            &vertex_layout.vertex_input_code,
            &binding_code,
            template_name,
            options,
        );

        // 调试输出生成的 Shader 代码
        if cfg!(debug_assertions) {
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

            println!(
                "================= Generated Shader Code {} ==================\n {}",
                template_name,
                normalize_newlines(&shader_source)
            );
        }

        let code_hash = xxh3_128(shader_source.as_bytes());

        let shader_module =
            self.module_cache
                .entry(code_hash)
                .or_insert(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("Shader Module {template_name}")),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                }));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &gpu_world.layout,
                &gpu_material.layout,
                &object_bind_group.layout,
                &frame_resources.screen_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> =
            vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Auto-Generated Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: canonical_key.color_format,
                    blend: canonical_key.blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: canonical_key.topology,
                front_face: canonical_key.front_face,
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
            multisample: wgpu::MultisampleState {
                count: canonical_key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        self.canonical_cache
            .insert(canonical_key, (pipeline.clone(), id));

        (pipeline, id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_shadow_pipeline(
        &mut self,
        device: &wgpu::Device,
        canonical_key: ShadowPipelineKey,
        options: &ShaderCompilationOptions,
        vertex_layout: &GeneratedVertexLayout,
        shadow_global_layout: &wgpu::BindGroupLayout,
        shadow_binding_wgsl: &str,
        gpu_material: &GpuMaterial,
        object_bind_group: &BindGroupContext,
    ) -> wgpu::RenderPipeline {
        if let Some(cached) = self.canonical_shadow_cache.get(&canonical_key) {
            return cached.clone();
        }

        let binding_code = format!(
            "{}\n{}\n{}",
            shadow_binding_wgsl, &gpu_material.binding_wgsl, &object_bind_group.binding_wgsl
        );

        let shader_source = ShaderGenerator::generate_shader(
            &vertex_layout.vertex_input_code,
            &binding_code,
            "passes/depth",
            options,
        );

        let code_hash = xxh3_128(shader_source.as_bytes());

        let shader_module =
            self.module_cache
                .entry(code_hash)
                .or_insert(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Shadow Shader Module"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                }));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[
                shadow_global_layout,
                &gpu_material.layout,
                &object_bind_group.layout,
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> =
            vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: canonical_key.topology,
                front_face: canonical_key.front_face,
                cull_mode: canonical_key.cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: canonical_key.depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.canonical_shadow_cache
            .insert(canonical_key, pipeline.clone());

        pipeline
    }
}
