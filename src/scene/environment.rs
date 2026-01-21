//! Environment - 纯数据结构
//! 
//! 描述 IBL/天空盒配置

use crate::resources::texture::{Texture, TextureSource};

/// IBL 环境贴图配置
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Environment {
    /// 原始环境贴图 (Source Cubemap, 用于背景显示或重新生成 PMREM)
    pub source_env_map: Option<TextureSource>,
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
        // let new_handle = texture_bundle.map(|(h, _)| h);

        // if self.source_env_map != new_handle {
        //     self.source_env_map = new_handle;
        //     // 清空 PMREM 以触发重新生成
        //     self.pmrem_map = None; 
        //     self.env_map_max_mip_level = 0.0;
        // }

        self.source_env_map = texture_bundle.map(|(h, _)| h);
        self.pmrem_map = self.source_env_map.clone();

        self.env_map_max_mip_level = texture_bundle
            .map(|(_, texture)| (texture.mip_level_count() - 1) as f32)
            .unwrap_or(0.0);
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
}