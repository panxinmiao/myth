//! System Textures
//!
//! Global pool of 1×1 fallback textures used as default bind-group fillers
//! when optional rendering features (SSAO, shadows, transmission, etc.) are
//! disabled.  Textures are classified by **data semantics** rather than by
//! the feature they serve, eliminating per-feature dummy proliferation.
//!
//! # Design Rationale
//!
//! GPU bind-group layouts are strict: every declared slot must be bound even
//! when the feature that writes it is inactive.  Instead of creating one
//! dedicated dummy per feature (which couples low-level resource management
//! to high-level pipeline knowledge), `SystemTextures` provides a small set
//! of mathematically-neutral textures:
//!
//! | Texture          | Value              | Typical Usage                   |
//! |------------------|--------------------|----------------------------------|
//! | `white_2d`       | `[255,255,255,255]`| Multiplicative identity (AO, shadow) |
//! | `black_2d`       | `[0,0,0,255]`      | Additive identity (emission)     |
//! | `transparent_2d` | `[0,0,0,0]`        | No contribution (transmission)   |
//! | `normal_2d`      | `[128,128,255,255]`| Tangent-space +Z (flat normal)   |
//! | `black_cube`     | 6×`[0,0,0,255]`    | Empty environment / IBL          |
//! | `white_r8`       | `[255]` R8Unorm    | SSAO fallback (fully lit)        |
//! | `black_hdr`      | Rgba16Float zero   | Transmission HDR fallback        |
//! | `depth_d2array`  | Depth32Float 1×1   | Shadow map D2Array fallback      |
//!
//! # Screen Bind Group (Group 3)
//!
//! The struct also owns the **layout**, **samplers**, and
//! **`build_bind_group`** logic for the screen-level descriptor set
//! (Group 3), consolidating everything previously scattered across
//! `ScreenBindGroupInfo` and `ResourceManager`.

use super::Tracked;

/// Global system fallback texture pool and Group 3 bind-group infrastructure.
///
/// Created once during renderer initialisation and shared (by reference)
/// with all render-graph contexts.  All textures are 1×1 (or 1×1×6 for
/// cubes) and never mutated after creation.
pub struct SystemTextures {
    // ─── Data-Semantic Fallback Textures ───────────────────────────
    /// 1×1 RGBA8 `[255,255,255,255]` — multiplicative identity.
    pub white_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[0,0,0,255]` — additive identity.
    pub black_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[0,0,0,0]` — fully transparent / no contribution.
    pub transparent_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[128,128,255,255]` — default tangent-space normal (+Z).
    pub normal_2d: Tracked<wgpu::TextureView>,

    /// 1×1×6 RGBA8 all-black cube map — empty environment / IBL.
    pub black_cube: Tracked<wgpu::TextureView>,

    /// 1×1 R8Unorm `[255]` — SSAO fallback (AO = 1.0, fully lit).
    pub white_r8: Tracked<wgpu::TextureView>,

    /// 1×1 Rgba16Float zero — HDR transmission fallback.
    pub black_hdr: Tracked<wgpu::TextureView>,

    /// 1×1 Depth32Float D2Array — shadow map fallback.
    pub depth_d2array: Tracked<wgpu::TextureView>,

    // ─── Screen BindGroup Infrastructure (Group 3) ─────────────────
    /// `BindGroupLayout` for Group 3 (transmission, SSAO, shadow).
    pub screen_layout: Tracked<wgpu::BindGroupLayout>,

    /// Linear-clamp sampler shared by transmission / SSAO sampling.
    pub screen_sampler: Tracked<wgpu::Sampler>,

    /// `LessEqual` comparison sampler for PCF shadow sampling.
    pub shadow_compare_sampler: Tracked<wgpu::Sampler>,
}

impl SystemTextures {
    /// Creates all system textures and Group 3 infrastructure.
    ///
    /// This is called once during renderer initialisation.  The total GPU
    /// memory footprint is negligible (a handful of 1×1 textures).
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // ── Data-Semantic Textures ─────────────────────────────────────

        let white_2d = create_1x1_rgba8(device, queue, [255, 255, 255, 255], "sys_white_2d");
        let black_2d = create_1x1_rgba8(device, queue, [0, 0, 0, 255], "sys_black_2d");
        let transparent_2d = create_1x1_rgba8(device, queue, [0, 0, 0, 0], "sys_transparent_2d");
        let normal_2d = create_1x1_rgba8(device, queue, [128, 128, 255, 255], "sys_normal_2d");
        let black_cube = create_1x1_cube(device, queue, [0, 0, 0, 255], "sys_black_cube");
        let white_r8 = create_1x1_r8(device, queue, 255, "sys_white_r8");
        let black_hdr = create_1x1_hdr(device, "sys_black_hdr");
        let depth_d2array = create_1x1_depth_d2array(device, "sys_depth_d2array");

        // ── Group 3 Layout ─────────────────────────────────────────────

        let screen_layout = Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Screen/Transient Layout (Group 3)"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            },
        ));

        // ── Samplers ───────────────────────────────────────────────────

        let screen_sampler = Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Screen Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        let shadow_compare_sampler =
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Comparison Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }));

        Self {
            white_2d,
            black_2d,
            transparent_2d,
            normal_2d,
            black_cube,
            white_r8,
            black_hdr,
            depth_d2array,
            screen_layout,
            screen_sampler,
            shadow_compare_sampler,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers — minimal 1×1 texture construction
// ═══════════════════════════════════════════════════════════════════════════

fn create_1x1_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

fn create_1x1_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 6,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // 1×1 pixel × 4 bytes × 6 faces
    let mut data = [0u8; 24];
    for face in 0..6 {
        data[face * 4..face * 4 + 4].copy_from_slice(&color);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }))
}

fn create_1x1_r8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    value: u8,
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[value],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// Rgba16Float 1×1 zero-initialised texture (GPU zero-fills by default).
fn create_1x1_hdr(device: &wgpu::Device, label: &str) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// Depth32Float 1×1 D2Array fallback for shadow-less frames.
fn create_1x1_depth_d2array(device: &wgpu::Device, label: &str) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    }))
}
