use glam::{Vec3};
use uuid::Uuid;



#[derive(Debug, Clone)]
pub struct ShadowConfig {
    pub bias: f32,
    pub normal_bias: f32,
    pub map_size: u32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            bias: 0.005,
            normal_bias: 0.02,
            map_size: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    // cascades: u32,
}

#[derive(Debug, Clone)]
pub struct PointLight {
    pub range: f32,
}

#[derive(Debug, Clone)]
pub struct SpotLight {
    pub range: f32,
    pub inner_cone: f32,
    pub outer_cone: f32,
}

// 高层抽象：场景中的光源组件
#[derive(Debug, Clone)]
pub enum LightKind {
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
}

#[derive(Debug, Clone)]
pub struct Light {
    pub uuid: Uuid,
    pub color: Vec3,
    pub intensity: f32, // 建议：明确单位，例如 PBR 中 Point 用 Candela，Directional 用 Lux
    pub kind: LightKind,

    pub shadow: Option<ShadowConfig>,
}

impl Light {
    pub fn new_directional(color: Vec3, intensity: f32) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            color,
            intensity,
            kind: LightKind::Directional(DirectionalLight {
                // cascades: 4,
            }),
            shadow: Some(ShadowConfig::default()),
        }
    }

    pub fn new_point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            color,
            intensity,
            kind: LightKind::Point(PointLight {
                range,
            }),
            shadow: Some(ShadowConfig::default()),
        }
    }

    pub fn new_spot(color: Vec3, intensity: f32, range: f32, inner_cone: f32, outer_cone: f32) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            color,
            intensity,
            kind: LightKind::Spot(SpotLight {
                range,
                inner_cone,
                outer_cone,
            }),
            shadow: Some(ShadowConfig::default()),
        }
    }
}