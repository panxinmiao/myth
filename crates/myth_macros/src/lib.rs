//! Procedural macros for the Myth engine material system.
//!
//! Provides the [`myth_material`] attribute macro that transforms a clean,
//! declarative material definition into a production-ready material type with:
//!
//! - Concurrent-safe GPU buffer management via [`CpuBuffer`] and [`RwLock`]
//! - Automatic version tracking for pipeline cache invalidation
//! - Generated getter/setter methods with fast-path optimizations
//! - Full [`MaterialTrait`] and [`RenderableMaterialTrait`] implementations
//!
//! # Example
//!
//! ```rust,ignore
//! use myth_macros::myth_material;
//!
//! #[myth_material(shader = "templates/unlit", uniforms = UnlitUniforms)]
//! pub struct UnlitMaterial {
//!     /// Base color.
//!     #[uniform]
//!     pub color: Vec4,
//!
//!     /// Opacity value.
//!     #[uniform]
//!     pub opacity: f32,
//!
//!     /// The color map.
//!     #[texture]
//!     pub map: TextureSlot,
//! }
//! ```

use proc_macro::TokenStream;

mod codegen;
mod parse;

/// Transforms a declarative material struct into a complete engine material type.
///
/// # Struct-level Attributes
///
/// | Attribute | Required | Description |
/// |-----------|----------|-------------|
/// | `shader = "path"` | Yes | Shader template path |
/// | `uniforms = Type` | Yes | Uniform buffer struct type |
/// | `crate_path = "path"` | No | Path to `myth_resources` (default: `crate`) |
///
/// # Field Attributes
///
/// | Attribute | Description |
/// |-----------|-------------|
/// | `#[uniform]` | Exposes a uniform struct field as a get/set property |
/// | `#[texture]` | Declares a texture slot with automatic GPU binding |
/// | `#[internal(...)]` | Preserves a field in the generated struct |
///
/// ## `#[internal]` options
///
/// - `default = "expr"` — Default value for construction
/// - `clone_with = "expr"` — Custom clone expression (receives `&Self`)
///
/// # Generated Code
///
/// The macro replaces the annotated struct with:
///
/// 1. **TextureSet struct** — `{Name}TextureSet` containing all texture slots
/// 2. **Material struct** — Rewritten with `CpuBuffer`, `RwLock`, `AtomicU64` internals
/// 3. **Constructor** — `from_uniforms(uniforms) -> Self`
/// 4. **Settings API** — `set_alpha_mode`, `set_side`, `set_depth_test`, `set_depth_write`
/// 5. **Uniform accessors** — Per-field `set_xxx` / `xxx` with double-check locking
/// 6. **Texture accessors** — Per-slot `set_xxx`, `xxx`, `configure_xxx`
/// 7. **Clone impl** — Deep clone with atomic version snapshot
/// 8. **Trait impls** — `MaterialTrait` + `RenderableMaterialTrait`
#[proc_macro_attribute]
pub fn myth_material(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(attr as parse::MaterialAttrs);
    let input = syn::parse_macro_input!(item as syn::DeriveInput);

    match codegen::generate(args, input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
