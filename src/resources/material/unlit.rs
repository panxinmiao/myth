use std::sync::atomic::AtomicU64;

use glam::Vec4;
use parking_lot::RwLock;

use crate::assets::TextureHandle;
use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{AlphaMode, MaterialSettings, Side, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::UnlitUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Clone, Default, Debug)]
pub struct UnlitTextureSet {
    pub map: TextureSlot,

    pub map_sampler: Option<SamplerSource>,
}

#[derive(Debug)]
pub struct UnlitMaterial {
    pub(crate) uniforms: CpuBuffer<UnlitUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures: RwLock<UnlitTextureSet>,

    pub auto_sync_texture_to_uniforms: bool,
}

impl UnlitMaterial {
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        let uniform_data = UnlitUniforms {
            color,
            ..Default::default()
        };

        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("UnlitUniforms"),
            ),
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),
            textures: RwLock::new(UnlitTextureSet::default()),
            auto_sync_texture_to_uniforms: false,
        }
    }

    // -- Builder pattern (chainable at construction time) --

    /// Sets the base color (builder).
    #[must_use]
    pub fn with_color(self, color: Vec4) -> Self {
        self.uniforms.write().color = color;
        self
    }

    /// Sets the opacity (builder).
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        self.uniforms.write().opacity = opacity;
        self
    }

    /// Sets the color map texture (builder).
    #[must_use]
    pub fn with_map(self, handle: TextureHandle) -> Self {
        self.set_map(Some(handle));
        self
    }

    /// Sets the face culling side (builder).
    #[must_use]
    pub fn with_side(self, side: Side) -> Self {
        self.set_side(side);
        self
    }

    /// Sets the alpha mode (builder).
    #[must_use]
    pub fn with_alpha_mode(self, mode: AlphaMode) -> Self {
        self.set_alpha_mode(mode);
        self
    }

    /// Sets depth write (builder).
    #[must_use]
    pub fn with_depth_write(self, enabled: bool) -> Self {
        self.set_depth_write(enabled);
        self
    }
}

impl_material_api!(
    UnlitMaterial,
    UnlitUniforms,
    uniforms: [
        (color,   Vec4, "Base color."),
        (opacity, f32,  "Opacity value."),
        (alpha_test, f32, "Alpha test threshold."),
    ],
    textures: [
        (map, "The color map."),
    ]
);

impl_material_trait!(
    UnlitMaterial,
    "templates/unlit",
    UnlitUniforms,
    textures: [
        map,
    ]
);

impl Default for UnlitMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
