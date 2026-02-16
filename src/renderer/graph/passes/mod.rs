//! 渲染 Pass 实现
//!
//! 包含各种具体的渲染 Pass 实现。
//!
//! # Pass 分类
//!
//! ## 数据准备 Pass
//! - [`SceneCullPass`][]: 场景剔除、命令生成与排序
//!
//! ## 绘制 Pass (Simple Path - LDR)
//! - [`SimpleForwardPass`][]: 单 Pass 完成不透明+透明绘制
//!
//! ## 绘制 Pass (PBR Path - HDR)
//! - [`OpaquePass`][]: 仅绘制不透明物体
//! - [`TransmissionCopyPass`][]: 复制场景颜色供 Transmission 使用
//! - [`TransparentPass`][]: 仅绘制透明物体
//!
//! ## 天空/背景 Pass
//! - [`SkyboxPass`]: 渲染天空盒/渐变/全景贴图背景
//!
//! ## 计算 Pass
//! - [`BRDFLutComputePass`]: BRDF LUT 预计算
//! - [`IBLComputePass`]: IBL 预滤波
//!
//! ## 后处理 Pass
//! - [`ToneMapPass`]: 色调映射（HDR → LDR）
//!
//! # 渲染管线拓扑
//!
//! ## Simple Path (LDR/Fast)
//! ```text
//! SceneCullPass → SimpleForwardPass → Surface
//! ```
//!
//! ## PBR Path (HDR/Physical)
//! ```text
//! SceneCullPass → OpaquePass → [SkyboxPass] → [TransmissionCopyPass] → TransparentPass → ToneMapPass → Surface
//! ```

mod brdf_lut_compute;
mod cull;
mod ibl_compute;
mod opaque;
mod shadow;
mod simple_forward;
mod skybox;
mod tone_mapping;
mod transmission_copy;
mod transparent;

pub use brdf_lut_compute::BRDFLutComputePass;
pub use cull::SceneCullPass;
pub use ibl_compute::IBLComputePass;
pub use opaque::OpaquePass;
pub use shadow::ShadowPass;
pub use simple_forward::SimpleForwardPass;
pub use skybox::SkyboxPass;
pub use tone_mapping::ToneMapPass;
pub use transmission_copy::TransmissionCopyPass;
pub use transparent::TransparentPass;
