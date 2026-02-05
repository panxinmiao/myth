//! 渲染阶段定义
//!
//! `RenderStage` 定义渲染管线的标准阶段顺序，
//! 允许用户在指定阶段插入自定义渲染节点。

/// 渲染阶段枚举
///
/// 定义渲染管线的执行顺序。每个阶段可以包含多个渲染节点，
/// 同阶段内的节点按添加顺序执行。
///
/// # 阶段说明
///
/// | 阶段 | 用途 | 典型内容 |
/// |------|------|----------|
/// | `PreProcess` | 资源上传、计算预处理 | BRDF LUT 计算、IBL 预滤波 |
/// | `ShadowMap` | 阴影贴图渲染 | 级联阴影、点光源阴影 |
/// | `Opaque` | 不透明物体渲染 | Forward/Deferred 渲染 |
/// | `Skybox` | 天空盒渲染 | 环境贴图、程序化天空 |
/// | `Transparent` | 半透明物体渲染 | Alpha 混合物体 |
/// | `PostProcess` | 后处理效果 | ToneMapping、Bloom、FXAA |
/// | `UI` | 用户界面 | egui、调试信息 |
///
/// # 示例
///
/// ```ignore
/// // 在 PostProcess 阶段前插入描边效果
/// frame_builder.add_node(RenderStage::PostProcess, &outline_pass);
///
/// // 在 UI 阶段插入 egui 渲染
/// frame_builder.add_node(RenderStage::UI, &ui_pass);
/// ```
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
#[repr(u8)]
pub enum RenderStage {
    /// 预处理阶段：资源上传、计算着色器预处理
    ///
    /// 适用于：BRDF LUT 生成、IBL 预滤波、GPU 粒子计算
    PreProcess = 0,

    /// 阴影贴图渲染阶段
    ///
    /// 适用于：定向光阴影、点光源阴影、级联阴影贴图
    ShadowMap = 1,

    /// 不透明物体渲染阶段（G-Buffer 或 Forward）
    ///
    /// 适用于：标准 PBR 渲染、Deferred G-Buffer 填充
    Opaque = 2,

    /// 天空盒渲染阶段
    ///
    /// 适用于：Cubemap 天空盒、程序化天空、大气散射
    Skybox = 3,

    /// 半透明物体渲染阶段
    ///
    /// 适用于：Alpha 混合物体、粒子系统、玻璃/水面
    Transparent = 4,

    /// 后处理阶段
    ///
    /// 适用于：色调映射、泛光、景深、FXAA/TAA
    PostProcess = 5,

    /// 用户界面阶段（最后执行）
    ///
    /// 适用于：egui、ImGui、调试覆盖层
    UI = 6,
}

impl RenderStage {
    /// 获取阶段的数值索引（用于排序）
    #[inline]
    #[must_use]
    pub const fn order(self) -> u8 {
        self as u8
    }

    /// 阶段名称（用于调试）
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PreProcess => "PreProcess",
            Self::ShadowMap => "ShadowMap",
            Self::Opaque => "Opaque",
            Self::Skybox => "Skybox",
            Self::Transparent => "Transparent",
            Self::PostProcess => "PostProcess",
            Self::UI => "UI",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_ordering() {
        assert!(RenderStage::PreProcess < RenderStage::ShadowMap);
        assert!(RenderStage::ShadowMap < RenderStage::Opaque);
        assert!(RenderStage::Opaque < RenderStage::Skybox);
        assert!(RenderStage::Skybox < RenderStage::Transparent);
        assert!(RenderStage::Transparent < RenderStage::PostProcess);
        assert!(RenderStage::PostProcess < RenderStage::UI);
    }
}
