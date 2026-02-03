//! 渲染上下文
//!
//! `RenderContext` 在渲染图的各个 Pass 之间传递共享数据，
//! 避免参数列表过长，统一数据访问方式。

use crate::RenderSettings;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::scene::camera::RenderCamera;
use crate::scene::{Scene};
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::pipeline::PipelineCache;
use crate::assets::AssetServer;
use crate::renderer::graph::{RenderState, ExtractedScene};
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
    /// 帧资源
    pub frame_resources: &'a FrameResources,
    /// 当前时间
    pub time: f32,


    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    // === Post Process 状态机 ===
    // 当前链条的输入源 (初始指向 frame_resources.scene_color_view)
    pub current_pp_input: &'a Tracked<wgpu::TextureView>,
    // 内部计数器，用于决定下一个 Output 是 ping_pong[0] 还是 [1]
    pub(crate) pp_flip_flop: usize,
}


impl<'a> RenderContext<'a> {
    /// 获取 Post Process 的 Input 和 Output
    /// 
    /// 自动实现 Ping-Pong 切换：
    /// - Input: 上一个 Pass 的输出 (或 SceneColor)
    /// - Output: 下一个空闲的缓冲
    /// 
    /// 调用此方法后，Context 的 current_input 会自动更新指向 Output，
    /// 供下一个 Pass 使用。
    pub fn acquire_post_process_io(
        &mut self
    ) -> (&'a Tracked<wgpu::TextureView>, &'a Tracked<wgpu::TextureView>) {
        // 1. 确定输入
        let input = self.current_pp_input;

        // 2. 确定输出 (从 PingPong 数组中选一个)
        // 简单的逻辑：如果当前输入是 ping_pong[0]，输出就是 [1]，反之亦然。
        // 如果当前输入是 scene_color，输出就是 ping_pong[0]。
        
        // 我们可以利用地址比较或 ID 比较，或者简单的 flip_flop 计数器
        let output_idx = self.pp_flip_flop;
        let output = &self.frame_resources.ping_pong_buffers[output_idx].1;

        // 3. 状态流转
        self.current_pp_input = output; // 下一个 Pass 读这个 Output
        self.pp_flip_flop = 1 - self.pp_flip_flop; // 翻转索引

        (input, output)
    }
    
    /// 获取当前的最终结果 (用于最后上屏 ToneMap)
    pub fn current_post_process_output(&self) -> &'a Tracked<wgpu::TextureView> {
        self.current_pp_input
    }

}


pub struct FrameResources {
    // MSAA 缓冲 (可选)
    pub scene_msaa_view: Option<Tracked<wgpu::TextureView>>,

    // 场景主颜色缓冲 (HDR)
    // MSAA 开启时作为 Resolve Target，关闭时作为直接渲染目标
    // 始终作为 Post Process 链的【初始输入】
    pub scene_color_view: Tracked<wgpu::TextureView>,

    // 深度缓冲
    pub depth_view: Tracked<wgpu::TextureView>,

    // === Ping-Pong 缓冲 (Post Process) ===
    
    // 两个纹理交替使用，格式通常与 scene_color 一致
    // 这里全流程 HDR，直到最后上屏
    pub ping_pong_buffers: [(wgpu::Texture, Tracked<wgpu::TextureView>); 2],

    pub transmission_view: Tracked<wgpu::TextureView>,

    pub screen_bind_group: wgpu::BindGroup, // Set 3
    pub screen_bind_group_layout: wgpu::BindGroupLayout,

    pub screen_bindings_code: String,

    size: (u32, u32),
}

impl FrameResources {

    pub fn new(device: &wgpu::Device, settings: &RenderSettings, size: (u32, u32)) -> Self {
        
        // let screen_bind_group_layout = Self::create_sreen_bind_group_layout(device);
        
        let mut resources = Self::create_placeholders(device);
        resources.resize(device, settings, size);
        resources
        
    }

    pub fn create_texture(
        device: &wgpu::Device,
        size: (u32, u32), 
        format: wgpu::TextureFormat, 
        usage: wgpu::TextureUsages, 
        sample_count: u32,
        label: &str
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        view
    }

    fn create_sreen_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen Space Bind Group Layout"),
            entries: &[
                // Transmission Color Texture
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
                // Transmission Color Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_screen_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        transmission_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout,
            entries: &[
                // Transmission Texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(transmission_view),
                },
                // Transmission Sampler
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("Screen Transmission Sampler"),
                        address_mode_u: wgpu::AddressMode::ClampToEdge,
                        address_mode_v: wgpu::AddressMode::ClampToEdge,
                        address_mode_w: wgpu::AddressMode::ClampToEdge,
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        mipmap_filter: wgpu::MipmapFilterMode::Linear,
                        ..Default::default()
                    })),
                },
            ],
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, settings: &RenderSettings, size: (u32, u32)){

        if self.size == size {
            return;
        }

        let scene_color_view = Self::create_texture(
            device,
            size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
            "Scene Color Texture",
        );

        // Depth Texture
        let depth_view = Self::create_texture(
            device,
            size,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            1,
            "Depth Texture",
        );

        // Transmission Texture
        let transmission_view = Self::create_texture(
            device,
            size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
            "Transmission Texture",
        );

        let screen_bind_group = Self::create_screen_bind_group(device, &self.screen_bind_group_layout, &transmission_view);

        let scene_msaa_view = if settings.msaa_samples > 1 {
            Some(Self::create_texture(
                device,
                size,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                settings.msaa_samples,
                "Scene MSAA Color Texture",
            ))
        } else {
            None
        };

        self.size = size;
        self.scene_color_view = Tracked::new(scene_color_view);
        self.depth_view = Tracked::new(depth_view);
        self.transmission_view = Tracked::new(transmission_view);
        self.screen_bind_group = screen_bind_group;
        self.scene_msaa_view = scene_msaa_view.map(Tracked::new);

    }


    fn create_placeholders(device: &wgpu::Device) -> Self {
        Self::create_dummy_instance(device) 
    }
    
    // 仅用于 new 初始化，防止 uninitialized 错误
    fn create_dummy_instance(device: &wgpu::Device) -> Self {
        let placeholder_view = Self::create_texture(
            device,
            (1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
            "Placeholder Texture",
        );

        let screen_bind_group_layout = Self::create_sreen_bind_group_layout(device);
        let screen_bind_group = Self::create_screen_bind_group(device, &screen_bind_group_layout, &placeholder_view);

        Self {
            scene_msaa_view: None,
            scene_color_view: Tracked::new(placeholder_view.clone()),
            depth_view: Tracked::new(placeholder_view.clone()),
            ping_pong_buffers: [
                (
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Ping-Pong Texture 0"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }),
                    Tracked::new(placeholder_view.clone())
                ),
                (
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Ping-Pong Texture 1"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }),
                    Tracked::new(placeholder_view.clone())
                ),
            ],
            transmission_view: Tracked::new(placeholder_view.clone()),
            screen_bind_group,
            screen_bind_group_layout,
            screen_bindings_code: String::new(),
            size: (0, 0),
        }
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