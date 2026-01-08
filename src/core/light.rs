use glam::{Vec3};
use thunderdome::Index;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

#[derive(Debug, Clone)]
pub struct Light {
    pub id: Uuid,
    pub node_id: Option<Index>,
    pub light_type: LightType,
    pub color: Vec3,
    pub intensity: f32,
    
    // Transform
    pub position: Vec3,
    pub direction: Vec3, // for Directional & Spot
    
    // Parameters
    pub range: f32,      // for Point & Spot
    pub inner_cone: f32, // for Spot
    pub outer_cone: f32, // for Spot
    
    pub cast_shadow: bool,
    pub shadow_bias: f32,
    // pub shadow_map: Option<Texture>, // 后续扩展
}

impl Light {
    pub fn new_directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            node_id: None,
            light_type: LightType::Directional,
            color,
            intensity,
            position: Vec3::ZERO,
            direction: direction.normalize(),
            range: 0.0,
            inner_cone: 0.0,
            outer_cone: 0.0,
            cast_shadow: false,
            shadow_bias: 0.005,
        }
    }

    pub fn new_point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            node_id: None,
            light_type: LightType::Point,
            color,
            intensity,
            position,
            direction: Vec3::ZERO,
            range,
            inner_cone: 0.0,
            outer_cone: 0.0,
            cast_shadow: false,
            shadow_bias: 0.005,
        }
    }
    
    // ... 可以自行添加 new_spot
}