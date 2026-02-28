mod macros;
mod phong;
mod physical;
mod unlit;
use parking_lot::RwLockWriteGuard;

pub use phong::PhongMaterial;
pub use physical::{PhysicalFeatures, PhysicalMaterial};
pub use unlit::UnlitMaterial;

use std::{
    any::Any,
    borrow::Cow,
    ops::Deref,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::renderer::core::builder::ResourceBuilder;
use crate::resources::buffer::BufferRef;
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use crate::resources::uniforms::Mat3Uniform;
use crate::{
    assets::TextureHandle,
    resources::screen_space::{STENCIL_FEATURE_SSR, STENCIL_FEATURE_SSS},
};
use glam::{Vec2, Vec4};
use uuid::Uuid;

// ============================================================================
// TextureSlot Architecture
// ============================================================================

/// UV texture transformation parameters.
///
/// Defines offset, rotation, and scale transformations applied to
/// texture coordinates before sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    /// UV offset (translation)
    pub offset: Vec2,
    /// Rotation angle in radians
    pub rotation: f32,
    /// UV scale factor
    pub scale: Vec2,
}

impl Default for TextureTransform {
    #[inline]
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }
}

/// A texture slot that holds a texture reference and its UV transformation.
///
/// This is the primary way materials reference textures. Each slot can optionally
/// contain a texture handle and UV transformation parameters.
///
/// # Example
///
/// ```rust,ignore
/// // Create a slot with a texture
/// let slot = TextureSlot::new(texture_handle);
///
/// // Create a slot with custom transform
/// let slot = TextureSlot::with_transform(texture_handle, TextureTransform {
///     offset: Vec2::new(0.5, 0.0),
///     rotation: 0.0,
///     scale: Vec2::ONE,
/// });
/// ```
#[derive(Clone, Debug, Default)]
pub struct TextureSlot {
    /// The texture handle (None if no texture is assigned)
    pub texture: Option<TextureHandle>,
    /// UV transformation applied to this texture
    pub transform: TextureTransform,
    /// UV channel index (for meshes with multiple UV sets)
    pub channel: u8,
}

impl TextureSlot {
    /// Creates a new texture slot with default transform.
    #[inline]
    #[must_use]
    pub fn new(handle: TextureHandle) -> Self {
        Self {
            texture: Some(handle),
            transform: TextureTransform::default(),
            channel: 0,
        }
    }

    /// Creates a new texture slot with a custom transform.
    #[inline]
    #[must_use]
    pub fn with_transform(handle: TextureHandle, transform: TextureTransform) -> Self {
        Self {
            texture: Some(handle),
            transform,
            channel: 0,
        }
    }

    /// Computes the UV transformation matrix (3x3).
    ///
    /// Transform order: Translate * Rotate * Scale
    ///
    /// The resulting matrix is in column-major order for WGSL compatibility.
    #[inline]
    #[must_use]
    pub fn compute_matrix(&self) -> Mat3Uniform {
        let (s, c) = (-self.transform.rotation).sin_cos();
        let sx = self.transform.scale.x;
        let sy = self.transform.scale.y;

        // Column-major matrix:
        // | sx*c   -sy*s   tx |
        // | sx*s    sy*c   ty |
        // |  0       0      1 |
        Mat3Uniform::from_cols_array(&[
            sx * c,
            sx * s,
            0.0,
            -sy * s,
            sy * c,
            0.0,
            self.transform.offset.x,
            self.transform.offset.y,
            1.0,
        ])
    }

    #[inline]
    #[must_use]
    pub fn is_some(&self) -> bool {
        self.texture.is_some()
    }

    #[inline]
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.texture.is_none()
    }

    /// Sets the texture handle.
    #[inline]
    pub fn set_texture(&mut self, handle: Option<TextureHandle>) {
        self.texture = handle;
    }
}

// ============================================================================
// TextureSlotGuard - Texture Slot Modification Guard
// ============================================================================

/// RAII guard for texture slot modifications.
///
/// When the texture presence state changes (affecting shader macros),
/// automatically increments the material version number.
///
/// This ensures pipeline cache invalidation when textures are added or removed.
pub struct TextureSlotGuard<'a> {
    slot: &'a mut TextureSlot,
    version: &'a mut u64,
    was_some: bool,
}

impl<'a> TextureSlotGuard<'a> {
    /// Creates a new texture slot guard.
    #[inline]
    pub fn new(slot: &'a mut TextureSlot, version: &'a mut u64) -> Self {
        let was_some = slot.texture.is_some();
        Self {
            slot,
            version,
            was_some,
        }
    }
}

impl std::ops::Deref for TextureSlotGuard<'_> {
    type Target = TextureSlot;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.slot
    }
}

impl std::ops::DerefMut for TextureSlotGuard<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slot
    }
}

impl Drop for TextureSlotGuard<'_> {
    fn drop(&mut self) {
        let is_some = self.slot.texture.is_some();
        if self.was_some != is_some {
            *self.version = self.version.wrapping_add(1);
        }
    }
}

impl From<TextureHandle> for TextureSlot {
    #[inline]
    fn from(handle: TextureHandle) -> Self {
        Self::new(handle)
    }
}

impl From<Option<TextureHandle>> for TextureSlot {
    #[inline]
    fn from(opt: Option<TextureHandle>) -> Self {
        Self {
            texture: opt,
            transform: TextureTransform::default(),
            channel: 0,
        }
    }
}

impl From<TextureSource> for TextureSlot {
    #[inline]
    fn from(source: TextureSource) -> Self {
        match source {
            TextureSource::Asset(handle) => Self::new(handle),
            TextureSource::Attachment(_, _) => Self::default(),
        }
    }
}

impl From<Option<TextureSource>> for TextureSlot {
    #[inline]
    fn from(opt: Option<TextureSource>) -> Self {
        match opt {
            Some(TextureSource::Asset(handle)) => Self::new(handle),
            _ => Self::default(),
        }
    }
}

/// Base trait for all material types.
///
/// This is the user-facing interface for materials. For everyday use,
/// simply treat materials as `dyn Material`.
///
/// # Note
///
/// Users typically don't need to implement this trait directly.
/// Use the built-in material types or implement [`RenderableMaterialTrait`]
/// for custom materials.
pub trait MaterialTrait: Any + Send + Sync + std::fmt::Debug {
    /// Returns self as `Any` for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns mutable self as `Any` for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Advanced rendering interface for materials.
///
/// This trait is for internal use by the rendering system and for
/// implementing custom material types. Regular users don't need to
/// import or use this trait directly.
///
/// # Implementing Custom Materials
///
/// To create a custom material type:
///
/// 1. Implement [`MaterialTrait`] for basic type support
/// 2. Implement this trait for rendering capabilities
/// 3. Define your uniform struct with `#[repr(C)]` and `bytemuck`
/// 4. Create a corresponding shader template
///
/// See `PhysicalMaterial` for a reference implementation.
pub trait RenderableMaterialTrait: MaterialTrait {
    /// Returns the shader template name.
    fn shader_name(&self) -> &'static str;
    /// Returns the material version (used for cache invalidation).
    fn version(&self) -> u64;
    /// Returns shader macro definitions based on current material state.
    fn shader_defines(&self) -> ShaderDefines;
    /// Returns material rendering settings.
    fn settings(&self) -> MaterialSettings;

    /// Visits all textures used by this material.
    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource));
    /// Defines GPU resource bindings for this material.
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
    /// Returns a reference to the uniform buffer.
    fn uniform_buffer(&self) -> BufferRef;
    /// Provides uniform data bytes to the callback.
    fn with_uniform_bytes(&self, f: &mut dyn FnMut(&[u8]));

    fn extra_defines(&self, _defines: &mut ShaderDefines) {
        // Default implementation does nothing
    }
}

/// Face culling mode for rendering.
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum Side {
    /// Render only front faces (counter-clockwise winding)
    Front,
    /// Render only back faces (clockwise winding)
    Back,
    /// Render both faces (no culling)
    Double,
}

/// Alpha blending mode for transparency handling.
#[derive(PartialEq, Clone, Debug, Copy)]
pub enum AlphaMode {
    /// Fully opaque, no transparency
    Opaque,
    /// Alpha cutoff (discard pixels below threshold)
    Mask(f32, bool), // (cutoff value, alpha to coverage)
    /// Standard alpha blending
    Blend,
}

/// Material render state settings.
///
/// These settings affect the GPU pipeline configuration and may
/// cause pipeline cache misses when changed.
#[derive(PartialEq, Clone, Debug, Copy)]
pub struct MaterialSettings {
    /// Alpha blending mode
    pub alpha_mode: AlphaMode,
    /// Whether to write to depth buffer
    pub depth_write: bool,
    /// Whether to perform depth testing
    pub depth_test: bool,
    /// Face culling mode
    pub side: Side,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            alpha_mode: AlphaMode::Opaque,
            depth_write: true,
            depth_test: true,
            side: Side::Front,
        }
    }
}

impl MaterialSettings {
    /// Generates shader macro definitions from settings.
    ///
    /// This is called internally by the rendering system to configure
    /// shader compilation based on material settings.
    pub(crate) fn generate_shader_defines(&self, defines: &mut ShaderDefines) {
        // Alpha Mode
        match self.alpha_mode {
            AlphaMode::Opaque => {
                defines.set("ALPHA_MODE", "OPAQUE");
            }
            AlphaMode::Mask(_cutoff, _alpha_to_coverage) => {
                defines.set("ALPHA_MODE", "MASK");
            }
            AlphaMode::Blend => {
                defines.set("ALPHA_MODE", "BLEND");
            }
        }
    }
}
/// RAII guard for material settings modifications.
///
/// Automatically increments the material version when settings change,
/// triggering pipeline cache invalidation.
pub struct SettingsGuard<'a> {
    guard: RwLockWriteGuard<'a, MaterialSettings>,
    version: &'a AtomicU64,
    initial_settings: MaterialSettings,
}

impl<'a> SettingsGuard<'a> {
    pub fn new(guard: RwLockWriteGuard<'a, MaterialSettings>, version: &'a AtomicU64) -> Self {
        // Save snapshot (MaterialSettings must implement Clone)
        let initial_settings = *guard;
        Self {
            guard,
            version,
            initial_settings,
        }
    }
}

impl std::ops::Deref for SettingsGuard<'_> {
    type Target = MaterialSettings;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl std::ops::DerefMut for SettingsGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for SettingsGuard<'_> {
    fn drop(&mut self) {
        if *self.guard != self.initial_settings {
            self.version.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Core Material Enum (Material Data Enum)
// ============================================================================

/// Material data enum with hybrid dispatch strategy.
///
/// Uses "static dispatch + dynamic escape hatch" approach:
/// - Built-in materials (Unlit/Phong/Physical) use static dispatch for performance
/// - Custom variant allows user-defined materials via dynamic dispatch
///
/// # Built-in Materials
///
/// - [`UnlitMaterial`]: Unlit, flat-shaded material
/// - [`PhongMaterial`]: Classic Blinn-Phong shading
/// - [`PhysicalMaterial`]: PBR material with metallic-roughness workflow, clearcoat, transmission, etc.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MaterialType {
    /// Unlit material (no lighting calculations)
    Unlit(UnlitMaterial),
    /// Classic Blinn-Phong shading model
    Phong(PhongMaterial),
    /// Advanced PBR material with additional features
    Physical(PhysicalMaterial),
    /// User-defined custom material
    Custom(Box<dyn RenderableMaterialTrait>),
}

impl MaterialTrait for MaterialType {
    fn as_any(&self) -> &dyn Any {
        match self {
            Self::Unlit(m) => m.as_any(),
            Self::Phong(m) => m.as_any(),
            Self::Physical(m) => m.as_any(),
            Self::Custom(m) => m.as_any(),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        match self {
            Self::Unlit(m) => m.as_any_mut(),
            Self::Phong(m) => m.as_any_mut(),
            Self::Physical(m) => m.as_any_mut(),
            Self::Custom(m) => m.as_any_mut(),
        }
    }
}

impl RenderableMaterialTrait for MaterialType {
    fn shader_name(&self) -> &'static str {
        match self {
            Self::Unlit(m) => m.shader_name(),
            Self::Phong(m) => m.shader_name(),
            Self::Physical(m) => m.shader_name(),
            Self::Custom(m) => m.shader_name(),
        }
    }

    fn version(&self) -> u64 {
        match self {
            Self::Unlit(m) => m.version(),
            Self::Phong(m) => m.version(),
            Self::Physical(m) => m.version(),
            Self::Custom(m) => m.version(),
        }
    }

    fn shader_defines(&self) -> ShaderDefines {
        match self {
            Self::Unlit(m) => m.shader_defines(),
            Self::Phong(m) => m.shader_defines(),
            Self::Physical(m) => m.shader_defines(),
            Self::Custom(m) => m.shader_defines(),
        }
    }

    fn settings(&self) -> MaterialSettings {
        match self {
            Self::Unlit(m) => m.settings(),
            Self::Phong(m) => m.settings(),
            Self::Physical(m) => m.settings(),
            Self::Custom(m) => m.settings(),
        }
    }

    fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource)) {
        match self {
            Self::Unlit(m) => m.visit_textures(visitor),
            Self::Phong(m) => m.visit_textures(visitor),
            Self::Physical(m) => m.visit_textures(visitor),
            Self::Custom(m) => m.visit_textures(visitor),
        }
    }

    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        match self {
            Self::Unlit(m) => m.define_bindings(builder),
            Self::Phong(m) => m.define_bindings(builder),
            Self::Physical(m) => m.define_bindings(builder),
            Self::Custom(m) => m.define_bindings(builder),
        }
    }

    fn uniform_buffer(&self) -> BufferRef {
        match self {
            Self::Unlit(m) => m.uniform_buffer(),
            Self::Phong(m) => m.uniform_buffer(),
            Self::Physical(m) => m.uniform_buffer(),
            Self::Custom(m) => m.uniform_buffer(),
        }
    }

    fn with_uniform_bytes(&self, visitor: &mut dyn FnMut(&[u8])) {
        match self {
            Self::Unlit(m) => m.with_uniform_bytes(visitor),
            Self::Phong(m) => m.with_uniform_bytes(visitor),
            Self::Physical(m) => m.with_uniform_bytes(visitor),
            Self::Custom(m) => m.with_uniform_bytes(visitor),
        }
    }
}

impl MaterialType {
    /// Tries to downcast to a concrete type (for Custom materials)
    pub fn as_custom<T: MaterialTrait + 'static>(&self) -> Option<&T> {
        match self {
            Self::Custom(m) => m.as_any().downcast_ref::<T>(),
            _ => None,
        }
    }

    /// Tries to downcast to a mutable reference of concrete type (for Custom materials)
    pub fn as_custom_mut<T: MaterialTrait + 'static>(&mut self) -> Option<&mut T> {
        match self {
            Self::Custom(m) => m.as_any_mut().downcast_mut::<T>(),
            _ => None,
        }
    }
}

// ============================================================================
// Material Main Struct (Material Wrapper)
// ============================================================================

#[derive(Debug)]
pub struct Material {
    pub uuid: Uuid,
    pub name: Option<Cow<'static, str>>,
    pub data: MaterialType,
}

impl Material {
    pub fn new(data: MaterialType) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: None,
            data,
        }
    }

    /// Creates a Material from a custom material
    pub fn new_custom<T: RenderableMaterialTrait + 'static>(custom_material: T) -> Self {
        Self::new(MaterialType::Custom(Box::new(custom_material)))
    }

    // Helper constructors
    #[must_use]
    pub fn new_unlit(color: Vec4) -> Self {
        Self::from(UnlitMaterial::new(color))
    }

    #[must_use]
    pub fn new_phong(color: Vec4) -> Self {
        Self::from(PhongMaterial::new(color))
    }

    #[must_use]
    pub fn new_physical(color: Vec4) -> Self {
        Self::from(PhysicalMaterial::new(color))
    }

    /// Exposes the rendering behavior interface
    #[inline]
    pub fn as_renderable(&self) -> &dyn RenderableMaterialTrait {
        &self.data
    }

    /// Gets a reference to the custom material
    pub fn as_custom<T: MaterialTrait + 'static>(&self) -> Option<&T> {
        self.data.as_custom::<T>()
    }

    /// Gets a mutable reference to the custom material
    pub fn as_custom_mut<T: MaterialTrait + 'static>(&mut self) -> Option<&mut T> {
        self.data.as_custom_mut::<T>()
    }

    // Type conversion helper methods
    pub fn as_unlit(&self) -> Option<&UnlitMaterial> {
        match &self.data {
            MaterialType::Unlit(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_unlit_mut(&mut self) -> Option<&mut UnlitMaterial> {
        match &mut self.data {
            MaterialType::Unlit(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_phong(&self) -> Option<&PhongMaterial> {
        match &self.data {
            MaterialType::Phong(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_phong_mut(&mut self) -> Option<&mut PhongMaterial> {
        match &mut self.data {
            MaterialType::Phong(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_physical(&self) -> Option<&PhysicalMaterial> {
        match &self.data {
            MaterialType::Physical(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_physical_mut(&mut self) -> Option<&mut PhysicalMaterial> {
        match &mut self.data {
            MaterialType::Physical(m) => Some(m),
            _ => None,
        }
    }

    pub fn uniforms(&self) -> &dyn Any {
        self.data.as_any()
    }

    pub fn as_any(&self) -> &dyn Any {
        self.data.as_any()
    }

    pub fn as_any_mut(&mut self) -> &mut dyn Any {
        self.data.as_any_mut()
    }

    // Proxy methods
    #[inline]
    pub(crate) fn shader_name(&self) -> &'static str {
        self.data.shader_name()
    }

    #[inline]
    pub(crate) fn shader_defines(&self) -> ShaderDefines {
        self.data.shader_defines()
    }

    #[inline]
    pub(crate) fn settings(&self) -> MaterialSettings {
        self.data.settings()
    }

    // Convenience accessors
    #[inline]
    pub fn alpha_mode(&self) -> AlphaMode {
        self.settings().alpha_mode
    }

    #[inline]
    pub fn depth_write(&self) -> bool {
        self.settings().depth_write
    }

    #[inline]
    pub fn depth_test(&self) -> bool {
        self.settings().depth_test
    }

    #[inline]
    pub fn side(&self) -> Side {
        self.settings().side
    }

    /// Defines GPU resource bindings (delegates to internal data)
    #[inline]
    pub fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }

    #[inline]
    pub fn auto_sync_texture_to_uniforms(&self) -> bool {
        match &self.data {
            MaterialType::Unlit(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Phong(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Physical(m) => m.auto_sync_texture_to_uniforms,
            MaterialType::Custom(_) => false,
        }
    }

    #[inline]
    pub fn visit_textures(&self, visitor: &mut dyn FnMut(&TextureSource)) {
        self.data.visit_textures(visitor);
    }

    #[inline]
    pub fn uniform_buffer(&self) -> BufferRef {
        self.data.uniform_buffer()
    }

    #[inline]
    pub fn with_uniform_bytes(&self, f: &mut dyn FnMut(&[u8])) {
        self.data.with_uniform_bytes(f);
    }

    #[inline]
    pub(crate) fn use_transmission(&self) -> bool {
        match &self.data {
            MaterialType::Physical(m) => m.features.read().contains(PhysicalFeatures::TRANSMISSION),
            _ => false,
        }
    }

    #[inline]
    pub(crate) fn features(&self) -> PhysicalFeatures {
        match &self.data {
            MaterialType::Physical(m) => *m.features.read(),
            _ => PhysicalFeatures::empty(),
        }
    }

    #[inline]
    pub(crate) fn ss_feature_mask(&self) -> u32 {
        let mut mask = 0;
        let features = self.features();
        if features.contains(PhysicalFeatures::SSS) {
            mask |= STENCIL_FEATURE_SSS;
        }
        if features.contains(PhysicalFeatures::SSR) {
            mask |= STENCIL_FEATURE_SSR;
        }
        mask
    }
}

// ============================================================================
// Syntax Sugar: Allows direct conversion from concrete material to generic Material
// ============================================================================

impl From<UnlitMaterial> for Material {
    fn from(data: UnlitMaterial) -> Self {
        Material::new(MaterialType::Unlit(data))
    }
}

impl From<PhongMaterial> for Material {
    fn from(data: PhongMaterial) -> Self {
        Material::new(MaterialType::Phong(data))
    }
}

impl From<PhysicalMaterial> for Material {
    fn from(data: PhysicalMaterial) -> Self {
        Material::new(MaterialType::Physical(data))
    }
}

impl Deref for Material {
    type Target = MaterialType;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
