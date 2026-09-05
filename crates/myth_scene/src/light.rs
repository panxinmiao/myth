use glam::Vec3;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Light flag marking a directional light as the sun.
pub const LIGHT_FLAG_IS_SUN: u32 = 1 << 0;
/// Light flag marking a directional light as the moon.
pub const LIGHT_FLAG_IS_MOON: u32 = 1 << 1;

/// Which faces of an occluder are written into the shadow map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowFaces {
    /// Back faces (default). On a closed solid this puts the recorded surface
    /// a thickness away from any receiver outside it, which hides acne.
    Back,
    /// Front faces. Needed when a receiver is *inside* a closed solid, where
    /// the back face is the receiver and the depth test has nothing to
    /// separate. Costs the usual acne, so pair it with `bias`/`normal_bias`.
    Front,
    /// Both faces.
    Both,
}

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
    /// Which faces this light's shadow map records (default [`ShadowFaces::Back`]).
    pub faces: ShadowFaces,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            bias: 0.0,
            normal_bias: 0.02,
            map_size: 2048,
            cascade_count: 4,
            cascade_split_lambda: 0.5,
            max_shadow_distance: 100.0,
            faces: ShadowFaces::Back,
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
    uuid: Uuid,
    id: u64,
    /// Bit flags used by renderer-side specialization paths.
    pub flags: u32,
    pub color: Vec3,
    pub intensity: f32, // Suggestion: specify units, e.g. in PBR: Point uses Candela, Directional uses Lux
    pub kind: LightKind,

    pub cast_shadows: bool,
    pub shadow: Option<ShadowConfig>,
}

impl Light {
    /// Returns the unique identifier for this light.
    #[inline]
    #[must_use]
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Returns the hash-based id derived from uuid.
    #[inline]
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

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
            flags: 0,
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
            flags: 0,
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
            flags: 0,
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
