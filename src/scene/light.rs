use glam::Vec3;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ShadowConfig {
    pub bias: f32,
    pub normal_bias: f32,
    pub map_size: u32,
    /// Number of cascades for directional light CSM (1-4, default 4).
    /// Ignored for spot/point lights.
    pub cascade_count: u32,
    /// Blend factor between logarithmic and uniform cascade split (0.0-1.0, default 0.5).
    pub cascade_split_lambda: f32,
    /// Maximum shadow distance for directional lights (default 100.0).
    /// Beyond this distance, no shadow is rendered.
    pub max_shadow_distance: f32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            bias: 0.001,
            normal_bias: 0.02,
            map_size: 1024,
            cascade_count: 4,
            cascade_split_lambda: 0.5,
            max_shadow_distance: 100.0,
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

// High-level abstraction: light component in the scene
#[derive(Debug, Clone)]
pub enum LightKind {
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
}

#[derive(Debug, Clone)]
pub struct Light {
    pub uuid: Uuid,
    pub id: u64,
    pub color: Vec3,
    pub intensity: f32, // Suggestion: specify units, e.g. in PBR: Point uses Candela, Directional uses Lux
    pub kind: LightKind,

    pub cast_shadows: bool,
    pub shadow: Option<ShadowConfig>,
}

impl Light {
    fn generate_id_from_uuid(uuid: &Uuid) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        uuid.hash(&mut hasher);
        hasher.finish()
    }

    #[must_use]
    pub fn new_directional(color: Vec3, intensity: f32) -> Self {
        let uuid = Uuid::new_v4();
        Self {
            uuid,
            id: Self::generate_id_from_uuid(&uuid),
            color,
            intensity,
            kind: LightKind::Directional(DirectionalLight {
                // cascades: 4,
            }),
            cast_shadows: false,
            shadow: Some(ShadowConfig::default()),
        }
    }

    #[must_use]
    pub fn new_point(color: Vec3, intensity: f32, range: f32) -> Self {
        let uuid = Uuid::new_v4();
        Self {
            uuid,
            id: Self::generate_id_from_uuid(&uuid),
            color,
            intensity,
            kind: LightKind::Point(PointLight { range }),
            cast_shadows: false,
            shadow: Some(ShadowConfig::default()),
        }
    }

    #[must_use]
    pub fn new_spot(
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_cone: f32,
        outer_cone: f32,
    ) -> Self {
        let uuid = Uuid::new_v4();
        Self {
            uuid,
            id: Self::generate_id_from_uuid(&uuid),
            color,
            intensity,
            kind: LightKind::Spot(SpotLight {
                range,
                inner_cone,
                outer_cone,
            }),
            cast_shadows: false,
            shadow: Some(ShadowConfig::default()),
        }
    }
}
