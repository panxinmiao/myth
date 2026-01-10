use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use glam::{Vec2, Mat3};
pub use wgpu::{TextureFormat, TextureDimension, TextureViewDimension, AddressMode, FilterMode, MipmapFilterMode};

// ============================================================================
// 1. 纹理数据源 (支持 Mipmaps 和 Array Layers)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct TextureSource {
    /// 原始二进制数据
    /// 为了兼容性，这里假设数据是紧凑排列的：
    /// Layer0[Mip0, Mip1...] -> Layer1[Mip0, Mip1...]
    pub data: Option<Vec<u8>>,

    pub width: u32,
    pub height: u32,
    /// 对于 2D 纹理是 1，对于 3D 纹理是深度
    pub depth: u32, 

    pub format: TextureFormat,
    
    pub mip_level_count: u32,
    pub array_layer_count: u32, // 对于 CubeMap 这里是 6
}

impl TextureSource {
    pub fn bytes_per_pixel(&self) -> u32 {
        // Todo: 确定逻辑是否正确
        self.format.block_copy_size(Some(wgpu::TextureAspect::All)).unwrap_or(4) as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSampler {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,

    // 高级功能：比较函数 (用于 Shadow Map PCF)
    pub compare: Option<wgpu::CompareFunction>,
    // 高级功能：各向异性过滤等级 (1 = 关闭)
    pub anisotropy_clamp: u16,

}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            compare: None,
            anisotropy_clamp: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureTransform {
    pub offset: Vec2,
    pub repeat: Vec2,
    pub rotation: f32,
    pub center: Vec2,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            repeat: Vec2::ONE,
            rotation: 0.0,
            center: Vec2::new(0.5, 0.5),
        }
    }
}

impl TextureTransform {
    /// 获取 3x3 UV 变换矩阵
    pub fn get_matrix(&self) -> Mat3 {
        let c = self.rotation.cos();
        let s = self.rotation.sin();
        let ox = self.offset.x;
        let oy = self.offset.y;
        let rx = self.repeat.x;
        let ry = self.repeat.y;
        let cx = self.center.x;
        let cy = self.center.y;

        Mat3::from_cols_array(&[
            c * rx,             s * rx,             0.0,
            -s * ry,            c * ry,             0.0,
            (c * -cx + s * -cy + cx) * rx + ox,
            (-s * -cx + c * -cy + cy) * ry + oy,
            1.0
        ])
    }
}

// ============================================================================
// 2. Texture Asset
// ============================================================================

#[derive(Debug)]
pub struct Texture {
    pub id: Uuid,
    pub name: String,
    
    pub source: TextureSource,
    pub dimension: TextureDimension, // D1, D2, D3

    // 例如：一个 D2 Texture 可能是 Cube View，也可能是 D2Array View
    pub view_dimension: TextureViewDimension,

    pub sampler: TextureSampler,
    pub transform: TextureTransform,
    
    pub version: AtomicU64,

    pub generation_id: AtomicU64,
}

impl Texture {
    pub fn new_2d(name: &str, width: u32, height: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            source: TextureSource {
                width, height, depth: 1,
                format,
                mip_level_count: 1,
                array_layer_count: 1,
                data,
            },
            dimension: TextureDimension::D2,
            view_dimension: TextureViewDimension::D2, // 默认 2D 视图
            sampler: TextureSampler::default(),
            transform: TextureTransform::default(),
            version: AtomicU64::new(0),
            generation_id: AtomicU64::new(0),
        }
    }

    /// 创建 Cube Map (天空盒/环境贴图)
    /// 假设 data 包含了 6 个面的数据，顺序通常是: +X, -X, +Y, -Y, +Z, -Z
    pub fn new_cube(name: &str, size: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            source: TextureSource {
                width: size, height: size, depth: 1,
                format,
                mip_level_count: 1,
                array_layer_count: 6, // 关键：6层
                data,
            },
            dimension: TextureDimension::D2, // 物理上是 2D 纹理数组
            view_dimension: TextureViewDimension::Cube, // 逻辑上是 Cube
            sampler: TextureSampler {
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            },
            transform: TextureTransform::default(),
            version: AtomicU64::new(0),
            generation_id: AtomicU64::new(0),
        }
    }

    /// 辅助：创建纯色纹理 (1x1)
    pub fn create_solid_color(name: &str, color: [u8; 4]) -> Texture {
        Self::new_2d(name, 1, 1, Some(color.to_vec()), wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub fn needs_update(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// 改变尺寸 (结构性变更)
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.source.width == width && self.source.height == height {
            return;
        }
        self.source.width = width;
        self.source.height = height;
        // 结构变了，内容肯定也变了（或失效了）
        self.generation_id.fetch_add(1, Ordering::Relaxed);
        self.needs_update();
    }


    /// 创建一个棋盘格测试纹理
    pub fn create_checkerboard(name: &str, width: u32, height: u32, check_size: u32) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        
        let color_a = [255, 255, 255, 255]; // 白
        let color_b = [0, 0, 0, 255];       // 黑 (或者用粉色 [255, 0, 255, 255] 方便调试)

        for y in 0..height {
            for x in 0..width {
                // 简单的异或逻辑生成棋盘格
                let cx = x / check_size;
                let cy = y / check_size;
                let is_a = (cx + cy) % 2 == 0;
                
                if is_a {
                    data.extend_from_slice(&color_a);
                } else {
                    data.extend_from_slice(&color_b);
                }
            }
        }

        Self::new_2d(name, width, height, Some(data), wgpu::TextureFormat::Rgba8Unorm)
    }
}