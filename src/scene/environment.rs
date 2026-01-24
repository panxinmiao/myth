//! Environment - 纯数据结构
//! 
//! 描述 IBL/天空盒配置

use crate::resources::texture::{Texture, TextureSource};

/// IBL 环境贴图配置
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Environment {
    /// 用户设置的原始环境贴图 (可能是 2D HDR 或 Cube)
    pub source_env_map: Option<TextureSource>,
    /// 标准化后的 CubeMap 源
    /// 如果 source_env_map 是 Cube，则此字段等于 source_env_map
    /// 如果 source_env_map 是 2D，则此字段指向转换后的 CubeMap
    pub(crate) processed_env_map: Option<TextureSource>,
    /// 预过滤的环境贴图 (PMREM, 用于 PBR Specular IBL)
    pub pmrem_map: Option<TextureSource>,
    /// BRDF LUT 贴图
    pub brdf_lut: Option<TextureSource>,
    /// 环境贴图的最大 mip 级别 (用于 roughness LOD)
    pub env_map_max_mip_level: f32,
    /// 环境光强度
    pub intensity: f32,
    /// 环境贴图旋转角度 (弧度)
    pub rotation: f32,
    /// 环境光颜色 (ambient)
    pub ambient_color: glam::Vec3,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            source_env_map: None,
            processed_env_map: None,
            pmrem_map: None,
            brdf_lut: None,
            env_map_max_mip_level: 0.0,
            intensity: 1.0,
            rotation: 0.0,
            ambient_color: glam::Vec3::ZERO,
        }
    }
    
    /// 设置环境贴图
    pub fn set_env_map(&mut self, texture_bundle: Option<(TextureSource, &Texture)>) {
        let new_handle = texture_bundle.map(|(h, _)| h);

        if self.source_env_map != new_handle {
            self.source_env_map = new_handle;
            self.processed_env_map = None;
            self.pmrem_map = None; 
            self.env_map_max_mip_level = 0.0;
        }
    }
    
    /// 设置 BRDF LUT
    pub fn set_brdf_lut(&mut self, handle: Option<TextureSource>) {
        self.brdf_lut = handle;
    }
    
    /// 设置环境光强度
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }
    
    /// 设置环境光颜色
    pub fn set_ambient_color(&mut self, color: glam::Vec3) {
        self.ambient_color = color;
    }
    
    /// 是否有有效的环境贴图
    pub fn has_env_map(&self) -> bool {
        self.source_env_map.is_some()
    }
    
    /// 获取处理后的环境贴图 (用于 Skybox 等需要 CubeMap 的地方)
    /// 只返回 processed_env_map，不回退到 source_env_map
    /// 因为 source_env_map 可能是 2D 纹理，而 Skybox 需要 CubeMap
    pub fn get_processed_env_map(&self) -> Option<&TextureSource> {
        self.processed_env_map.as_ref()
    }
}