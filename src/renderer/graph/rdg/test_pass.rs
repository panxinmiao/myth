use super::builder::PassBuilder;
use super::context::RdgExecuteContext;
use super::node::PassNode;
use super::types::TextureNodeId;
use crate::FxaaQuality;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::context::RdgPrepareContext;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use wgpu::CommandEncoder;

type FxaaL1CacheKey = (FxaaQuality, wgpu::TextureFormat);

pub struct RdgFxaaPass {
    // 瞬态/外部驱动数据
    pub input_tex: TextureNodeId,
    pub output_tex: TextureNodeId,
    pub target_quality: FxaaQuality,

    // 持久化内部缓存
    l1_cache_key: Option<FxaaL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    current_bind_group_key: Option<BindGroupKey>,
}

impl RdgFxaaPass {
    /// Creates a new FXAA pass.
    ///
    /// Only allocates the bind group layout and sampler. Pipelines are
    /// lazily created on first use (or when quality changes).
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),
            target_quality: FxaaQuality::High,
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
            current_bind_group_key: None,
        }
    }
}

impl PassNode for RdgFxaaPass {
    fn name(&self) -> &'static str {
        "RDG_FXAA_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // 声明读写意图
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // --------------------------------------------------------
        // 1. 惰性初始化 BindGroupLayout (仅发生一次)
        // --------------------------------------------------------
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FXAA BindGroup Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            self.bind_group_layout = Some(Tracked::new(layout));
        }

        // --------------------------------------------------------
        // 2. L1 Cache Diffing：极速拦截管线变更
        // --------------------------------------------------------
        let output_desc = &ctx.graph.resources[self.output_tex.0 as usize].desc;
        let output_format = output_desc.format;
        let current_key = (self.target_quality, output_format);

        if self.l1_cache_key != Some(current_key) {
            // L1 Miss: 目标画质或输出屏幕格式发生了改变，向 L2 请求新管线

            let mut options = ShaderCompilationOptions::default();

            // Only Low and High need explicit defines; Medium is the default in the shader
            if self.target_quality != FxaaQuality::Medium {
                options.add_define(self.target_quality.define_key(), "1");
            }

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                ctx.device,
                "passes/fxaa",
                &options,
                "",
                "",
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("FXAA Pipeline Layout"),
                        bind_group_layouts: &[&self.bind_group_layout.as_ref().unwrap()],
                        immediate_size: 0,
                    });

            // 向全局 PipelineCache 索取管线 ID
            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                &format!("FXAA Pipeline {:?}", self.target_quality),
            );
            self.pipeline_id = Some(id);

            // 更新 L1 Cache
            self.l1_cache_key = Some(current_key);

            // 管线/格式变了，强制使当前绑定的 BindGroup 失效
            self.current_bind_group_key = None;
        }

        // --------------------------------------------------------
        // 3. 物理别名感知与全局 BindGroup 去重缓存
        // --------------------------------------------------------

        let input_view = ctx.get_physical_texture(self.input_tex);
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        let current_key = BindGroupKey::new(bind_group_layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id());

        if self.current_bind_group_key.as_ref() != Some(&current_key) {
            // 只确保 Cache 中存在，不保留实体引用
            if ctx.global_bind_group_cache.get(&current_key).is_none() {
                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FXAA BindGroup"),
                    // 🌟 修正：解引用 Tracked
                    layout: &**bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&**input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&**sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache
                    .insert(current_key.clone(), new_bg);
            }

            self.current_bind_group_key = Some(current_key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut CommandEncoder) {
        // 1. 从 RDG 上下文中获取真实的 wgpu::TextureView
        let output_view = ctx.get_texture_view(self.output_tex);

        // 2.我们确信 prepare 阶段已经装填好了所需资源
        let pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.pipeline_id.expect("Pipeline not initialized!"));

        let bind_group_key = self
            .current_bind_group_key
            .as_ref()
            .expect("BindGroupKey should have been set in prepare!");
        let bind_group = ctx
            .global_bind_group_cache
            .get(&bind_group_key)
            .expect("BindGroup should have been prepared!");

        // 3. 录制原生的 WGPU 渲染指令
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG FXAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
