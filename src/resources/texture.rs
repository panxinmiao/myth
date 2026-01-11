use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use glam::{Vec2, Mat3};
use wgpu::{TextureFormat, TextureDimension, TextureViewDimension, AddressMode};
use crate::resources::image::Image;

// ============================================================================
// 1. 纹理数据源 (支持 Mipmaps 和 Array Layers)
// ===========================================================================

// #[derive(Debug)]
// pub struct SourceInner {
//     pub id: Uuid, // 或 u64
//     pub label: String,

//     // 1. 物理尺寸
//     pub width: u32,
//     pub height: u32,
    
//     // 关键：统一了 "深度 (3D)" 和 "层数 (Array/Cube)"
//     // wgpu 的 Extent3d 也是这么设计的 (depth_or_array_layers)
//     pub depth_or_array_layers: u32, 

//     // 2. 物理维度 (消除歧义)
//     // 决定了显存是如何开辟的：D1, D2, D3
//     pub dimension: wgpu::TextureDimension, 

//     // 3. 格式与层级
//     pub format: wgpu::TextureFormat,
//     pub mip_level_count: u32,

//     // 4. 数据内容 (CPU 副本)
//     pub data: RwLock<Option<Vec<u8>>>,
//     pub version: AtomicU64,
// }

// impl SourceInner {
//     pub fn bytes_per_pixel(&self) -> u32 {
//         // Todo: 确定逻辑是否正确
//         self.format.block_copy_size(Some(wgpu::TextureAspect::All)).unwrap_or(4) as u32
//     }
// }


// #[derive(Debug, Clone)]
// pub struct TextureSource(Arc<SourceInner>);

// impl TextureSource {
//     pub fn new(width: u32, height: u32, format: wgpu::TextureFormat, dimension: wgpu::TextureDimension, data: Option<Vec<u8>>) -> Self {
//         Self(Arc::new(SourceInner {
//             id: Uuid::new_v4(),
//             label: "Image".into(),
//             width, height, depth_or_array_layers: 1, format, mip_level_count: 1,
//             dimension,
//             data: RwLock::new(data),
//             version: AtomicU64::new(0),
//         }))
//     }

//     // 类似 BufferRef 的更新机制
//     pub fn update_data(&self, new_data: Vec<u8>) {
//         let mut lock = self.0.data.write().unwrap();
//         *lock = Some(new_data);
//         self.0.version.fetch_add(1, Ordering::Relaxed);
//     }
    
//     pub fn version(&self) -> u64 {
//         self.0.version.load(Ordering::Relaxed)
//     }
    
//     // ... getters ...
// }

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
    pub uuid: Uuid,
    pub name: String,
    
    pub image: Image,

    pub view_dimension: TextureViewDimension,

    pub sampler: TextureSampler,
    pub transform: TextureTransform,
    
    pub version: AtomicU64,
}

impl Texture {
    /// 基础构造：从现有 Image 创建 Texture
    pub fn new(name: &str, image: Image, view_dimension: TextureViewDimension) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            image,
            view_dimension,
            sampler: TextureSampler::default(),
            transform: TextureTransform::default(),
            version: AtomicU64::new(0),
        }
    }

    /// 辅助构造：创建 2D 纹理 (自动创建 Image)
    pub fn new_2d(name: &str, width: u32, height: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        let image = Image::new(
            name, width, height, 1, 
            TextureDimension::D2, 
            format, data
        );
        Self::new(name, image, TextureViewDimension::D2)
    }

    /// 辅助构造：创建 Cube Map
    pub fn new_cube(name: &str, size: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        let image = Image::new(
            name, size, size, 6, // 6 layers
            TextureDimension::D2, // 物理维度是 2D
            format, data
        );
        let mut tex = Self::new(name, image, TextureViewDimension::Cube);
        // Cube Map 默认使用 Clamp 采样
        tex.sampler.address_mode_u = AddressMode::ClampToEdge;
        tex.sampler.address_mode_v = AddressMode::ClampToEdge;
        tex.sampler.address_mode_w = AddressMode::ClampToEdge;
        tex
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }


    /// 辅助：创建纯色纹理 (1x1)
    pub fn create_solid_color(name: &str, color: [u8; 4]) -> Texture {
        Self::new_2d(name, 1, 1, Some(color.to_vec()), wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub fn needs_update(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
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