use glam::Vec3;
use std::num::NonZeroU8;

pub const STENCIL_FEATURE_SSS: u32 = 1 << 0;
pub const STENCIL_FEATURE_SSR: u32 = 1 << 1;
pub const STENCIL_WRITE_MASK: u32 = 0x0F;
// ============================================================================
// 1. 基础类型：全局稳定的 8-bit ID
// ============================================================================

/// 强类型的特性 ID，内部封装 NonZeroU8 使得 Option<FeatureId> 仅占 1 字节。
/// 有效范围: 1 ~ 255。 0 代表无特性。
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureId(pub NonZeroU8);

impl FeatureId {
    /// 将强类型的 ID 转换为底层着色器所需的 u32
    #[inline]
    pub fn to_u32(self) -> u32 {
        self.0.get() as u32
    }

    /// 尝试从着色器的 u32 数据还原为强类型的 ID (0 会自动变为 None)
    #[inline]
    pub fn from_u32(val: u32) -> Option<Self> {
        std::num::NonZeroU8::new(val as u8).map(FeatureId)
    }
}

// ============================================================================
// 2. SSS (次表面散射) 专用结构与注册表
// ============================================================================

/// GPU 端 SSS 数据结构 (16 字节)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct SssProfileData {
    pub scatter_color: [f32; 3], // 散射颜色
    pub scatter_radius: f32,     // 散射半径 (屏幕空间百分比或世界空间单位)
}

impl Default for SssProfileData {
    fn default() -> Self {
        Self {
            scatter_color: [0.0; 3],
            scatter_radius: 0.0,
        }
    }
}

/// 用户侧的 SSS 配置文件资产
#[derive(Clone, Debug)]
pub struct SssProfile {
    pub scatter_color: Vec3,
    pub scatter_radius: f32,
}

impl SssProfile {
    pub fn new(scatter_color: Vec3, scatter_radius: f32) -> Self {
        Self {
            scatter_color,
            scatter_radius,
        }
    }

    pub fn to_gpu_data(&self) -> SssProfileData {
        SssProfileData {
            scatter_color: self.scatter_color.into(),
            scatter_radius: self.scatter_radius,
        }
    }
}

/// SSS 专用全局定长分配器
pub struct SssRegistry {
    /// 严格对齐 GPU 布局的数组，ID 0 永远是 default
    pub buffer_data: [SssProfileData; 256],
    /// 空闲 ID 列表
    free_list: Vec<u8>,
    /// 版本号，用于触发 GPU 显存上传 (Diff-Sync)
    pub version: u64,
}

impl SssRegistry {
    pub fn new() -> Self {
        // 从 255 到 1，保证 pop() 时优先分配小号 ID
        let free_list = (1..=255).rev().collect();
        Self {
            buffer_data: [SssProfileData::default(); 256],
            free_list,
            version: 1,
        }
    }

    /// 注册一个新的 SSS Profile，返回全局稳定的 ID
    pub fn add(&mut self, profile: &SssProfile) -> Option<FeatureId> {
        if let Some(id) = self.free_list.pop() {
            self.buffer_data[id as usize] = profile.to_gpu_data();
            self.version += 1;
            Some(FeatureId(NonZeroU8::new(id).unwrap()))
        } else {
            log::warn!("SssRegistry is full (max 255 profiles).");
            None
        }
    }

    /// 更新已存在的 Profile (如 UI 动态调整)
    pub fn update(&mut self, id: FeatureId, profile: &SssProfile) {
        self.buffer_data[id.0.get() as usize] = profile.to_gpu_data();
        self.version += 1;
    }

    /// 移除 Profile，回收 ID 供后续使用
    pub fn remove(&mut self, id: FeatureId) {
        let index = id.0.get();
        self.buffer_data[index as usize] = SssProfileData::default();
        self.free_list.push(index);
        self.version += 1;
    }
}

// ============================================================================
// 3. 场景级全局开关
// ============================================================================
#[derive(Default, Clone, Debug)]
pub struct ScreenSpaceSettings {
    pub enable_sss: bool,
    pub enable_ssr: bool, // 预留给未来
}
