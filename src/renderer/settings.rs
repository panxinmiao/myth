//! 渲染设置

#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub power_preference: wgpu::PowerPreference,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
    pub clear_color: wgpu::Color,
    pub depth_format: wgpu::TextureFormat,
    pub vsync: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            clear_color: wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            depth_format: wgpu::TextureFormat::Depth32Float,
            vsync: true,
        }
    }
}
