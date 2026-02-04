//! 渲染上下文
//!
//! `RenderContext` 在渲染图的各个 Pass 之间传递共享数据，
//! 避免参数列表过长，统一数据访问方式。

use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::scene::camera::RenderCamera;
use crate::scene::{Scene};
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::pipeline::PipelineCache;
use crate::assets::AssetServer;
use crate::renderer::graph::{RenderState, ExtractedScene};
use crate::renderer::graph::frame::RenderLists;
use crate::renderer::core::resources::Tracked;

/// 渲染上下文
/// 
/// 在 RenderGraph 执行期间，所有 RenderNode 共享此上下文。
/// 包含 GPU 上下文、资源管理器、场景数据等。
/// 
/// # 性能考虑
/// - 所有字段都是引用，避免数据复制
/// - `surface_view` 每帧更新，指向当前交换链纹理
/// - 通过借用规则确保线程安全
pub struct RenderContext<'a> {
    /// WGPU 核心上下文（device, queue, surface）
    pub wgpu_ctx: &'a WgpuContext,
    /// GPU 资源管理器
    pub resource_manager: &'a mut ResourceManager,
    /// Pipeline 缓存
    pub pipeline_cache: &'a mut PipelineCache,
    /// 资产服务器
    pub assets: &'a AssetServer,
    /// 当前场景
    pub scene: &'a mut Scene,
    /// 相机
    pub camera: &'a RenderCamera,
    /// 当前帧的 Surface View
    pub surface_view: &'a wgpu::TextureView,
    /// 渲染状态
    pub render_state: &'a RenderState,
    /// 提取的场景数据
    pub extracted_scene: &'a ExtractedScene,
    /// 渲染列表（由 SceneCullPass 填充，供各个绘制 Pass 消费）
    pub render_frame: RenderFrameRef<'a>,
    /// 帧资源
    pub frame_resources: &'a FrameResources,
    /// 当前时间
    pub time: f32,


    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    // === Post Process 状态机 ===

    // pub current_color_texture_view: &'a Tracked<wgpu::TextureView>,
    // 内部计数器，用于决定下一个 Output 是 ping_pong[0] 还是 [1]
    pub(crate) color_view_flip_flop: usize,
}

/// RenderFrame 的可变引用包装
/// 
/// 用于在 RenderContext 中安全地访问 RenderLists
pub struct RenderFrameRef<'a> {
    pub render_lists: &'a mut RenderLists,
}


impl<'a> RenderContext<'a> {
    /// 获取 Post Process 的 Input 和 Output
    /// 
    /// 自动实现 Ping-Pong 切换：
    /// - Input: 上一个 Pass 的输出
    /// - Output: 下一个空闲的缓冲
    /// 
    /// 调用此方法后，Context 的 current_color_texture_view 会自动更新指向 Output，
    /// 供下一个 Pass 使用。
    pub fn acquire_pass_io(
        &mut self
    ) -> (&Tracked<wgpu::TextureView>, &wgpu::TextureView) {

        let current_idx = self.color_view_flip_flop;
    
        // 1. 确定输入
        let input = &self.frame_resources.scene_color_view[current_idx];

        // 2. 确定输出
        let output = &self.frame_resources.scene_color_view[1 - current_idx];

        // 3. 状态流转
        self.color_view_flip_flop = 1 - self.color_view_flip_flop; // 翻转索引

        (input, output)
    }

    pub fn get_scene_render_target_view(&self) -> &wgpu::TextureView {
        // 逻辑：如果是直连模式 ? Surface : SceneColor[0]
        if self.wgpu_ctx.enable_hdr {
            &self.frame_resources.scene_color_view[0]
        } else {
            self.surface_view
        }
    }

    // pub fn get_output_format(
    //     &self,
    //     output_to_screen: bool,
    // ) -> wgpu::TextureFormat {
    //     if output_to_screen {
    //         self.wgpu_ctx.surface_view_format
    //     } else {
    //         self.wgpu_ctx.color_format
    //     }
    // }

    pub fn get_scene_render_target_format(&self) -> wgpu::TextureFormat {
        if self.wgpu_ctx.enable_hdr {
            // 强制使用 HDR 格式 (推荐 Rgba16Float)
            // wgpu::TextureFormat::Rgba16Float 
            crate::renderer::HDR_TEXTURE_FORMAT
        } else {
            self.wgpu_ctx.surface_view_format
        }
    }

    // /// 获取场景渲染的目标 View
    // /// 
    // /// - 如果开启后处理：返回 FrameResources.scene_color_view (HDR)
    // /// - 如果关闭后处理：返回 Surface View (LDR/sRGB)
    // pub fn scene_output_view(&self) -> &'a wgpu::TextureView {
    //     // 这里的逻辑需要在 begin_frame 构建 Context 时确定
    //     // 或者在 Context 里存一个字段指向当前的 target
    //     self.current_render_target
    // }
    
    // /// 获取场景渲染的格式
    // pub fn scene_output_format(&self) -> wgpu::TextureFormat {
    //      // 同上，可能是 Rgba16Float 或 Bgra8UnormSrgb
    //      self.current_render_target_format 
    // }

}


pub struct FrameResources {
    // MSAA 缓冲 (可选)
    pub scene_msaa_view: Option<Tracked<wgpu::TextureView>>,

    // 场景主颜色缓冲 (HDR)
    // ping-pong 机制, 当非 straightforward 模式时，使用两个交替的缓冲，作为后处理输入输出
    pub scene_color_view: [Tracked<wgpu::TextureView>; 2],

    // 深度缓冲
    pub depth_view: Tracked<wgpu::TextureView>,

    pub transmission_view: Option<Tracked<wgpu::TextureView>>,

    pub screen_bind_group: Tracked<wgpu::BindGroup>,
    pub screen_bind_group_layout: Tracked<wgpu::BindGroupLayout>,
    screen_sampler: Tracked<wgpu::Sampler>,
    size: (u32, u32),
}

impl FrameResources {

    pub fn new(wgpu_ctx: &WgpuContext, size: (u32, u32)) -> Self {
        let device = &wgpu_ctx.device;

        // 1. 创建 Layout
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen/Transmission Layout"),
            entries: &[
                // Binding 0: Transmission Texture
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
                // Binding 1: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 2. 创建通用采样器
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Transmission Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });


        let placeholder_view = Self::create_texture(
            device,
            (1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
        1,
            "Placeholder Texture",
        );

        // 2. 创建初始 BindGroup (指向 Dummy)
        let initial_bind_group = Self::create_bind_group(
            &wgpu_ctx.device, 
            &layout, 
            &placeholder_view, 
            &sampler
        );

        let mut resources = Self {
            size: (0, 0),

            depth_view: Tracked::new(placeholder_view.clone()),
            scene_msaa_view: None,
            scene_color_view: [
                Tracked::new(placeholder_view.clone()),
                Tracked::new(placeholder_view.clone()),
            ],

            transmission_view: None,
            screen_bind_group: Tracked::new(initial_bind_group),
            screen_bind_group_layout: Tracked::new(layout),
            screen_sampler: Tracked::new(sampler),
        };
        
        resources.resize(&wgpu_ctx, size);
        resources
        
    }

    fn create_texture(
        device: &wgpu::Device,
        size: (u32, u32), 
        format: wgpu::TextureFormat, 
        usage: wgpu::TextureUsages, 
        sample_count: u32,
        mip_level_count: u32,
        label: &str
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_level_count,
            sample_count: sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        view
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen/Transmission BindGroup"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    pub fn resize(&mut self, wgpu_ctx: &WgpuContext, size: (u32, u32)){

        if self.size == size {
            return;
        }
        if size.0 == 0 || size.1 == 0 {
            return;
        }

        self.size = size;

        // Depth Texture
        let depth_view = Self::create_texture(
            &wgpu_ctx.device,
            size,
            wgpu_ctx.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu_ctx.msaa_samples,
            1,
            "Depth Texture",
        );
        self.depth_view = Tracked::new(depth_view);

        // Scene Color Texture(s) (ping-pong)
        if wgpu_ctx.enable_hdr {
            // 非直接渲染模式，创建两个 ping-pong 纹理
            let ping_pong_texture_0 = Self::create_texture(
                &wgpu_ctx.device,
                size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                1,
                1,
                "Ping-Pong Texture 0",
            );

            let ping_pong_texture_1 = Self::create_texture(
                &wgpu_ctx.device,
                size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                1,
                1,
                "Ping-Pong Texture 1",
            );
            self.scene_color_view = [
                Tracked::new(ping_pong_texture_0),
                Tracked::new(ping_pong_texture_1),
            ];
        }

        // MSAA Texture
        if wgpu_ctx.msaa_samples > 1 {
            // 创建 MSAA 纹理, 格式与主颜色纹理(Resolve target)相同

            let masaa_target_fromat = if wgpu_ctx.enable_hdr{
                crate::renderer::HDR_TEXTURE_FORMAT
            } else {
                wgpu_ctx.surface_view_format
            };

            let scene_msaa_view =Self::create_texture(
                &wgpu_ctx.device,
                size,
                masaa_target_fromat,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                wgpu_ctx.msaa_samples,
                1,
                "Scene MSAA Color Texture",
            );
            self.scene_msaa_view = Some(Tracked::new(scene_msaa_view));

        }

        if self.transmission_view.is_some() {
            let mip_level_count = ((size.0.max(size.1) as f32).log2().floor() as u32) + 1;
            // 1. 创建纹理 (Tracked::new 会生成新 ID)
            let texture_view = Self::create_texture(
                &wgpu_ctx.device,
                size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                1,
                mip_level_count,
                "Transmission Texture"
            );
            let tracked_view = Tracked::new(texture_view);

            // 2. 立即创建 BindGroup (显式管理)
            // 这里我们不做 Cache 查找，直接 New 一个，反正 Texture 变了 BindGroup 必须变
            let bind_group = wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Screen BindGroup"),
                layout: &self.screen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tracked_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.screen_sampler),
                    },
                ],
            });

            self.transmission_view = Some(tracked_view);
            self.screen_bind_group = Tracked::new(bind_group);
        }

    }

    pub fn ensure_transmission_resource(&mut self, device: &wgpu::Device) -> &wgpu::BindGroup {
        
        if self.transmission_view.is_none() {
            let mip_level_count = ((self.size.0.max(self.size.1) as f32).log2().floor() as u32) + 1;
            let texture_view = Self::create_texture(
                device,
                self.size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                1,
                mip_level_count,
                "Transmission Texture"
            );
            let tracked_view = Tracked::new(texture_view);

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Screen BindGroup"),
                layout: &self.screen_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tracked_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.screen_sampler),
                    },
                ],
            });

            self.transmission_view = Some(tracked_view);
            self.screen_bind_group = Tracked::new(bind_group);
        }

        &self.screen_bind_group

    }
}


pub struct PrepareContext<'a> {
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,
    pub assets: &'a AssetServer,
    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,
    pub wgpu_ctx: &'a WgpuContext,
}