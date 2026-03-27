//! AST parsing for the `#[myth_material]` attribute macro.
//!
//! Extracts struct-level configuration and classifies each field as
//! `uniform`, `texture`, or `internal`.

use syn::{
    Attribute, DeriveInput, Expr, Field, Fields, Ident, LitStr, Path, Token, Type, Visibility,
    parse::{Parse, ParseStream},
};

// ============================================================================
// Struct-level Attributes
// ============================================================================

/// Configuration parsed from `#[myth_material(shader = "...", ...)]`.
pub struct MaterialAttrs {
    pub shader: String,
    pub crate_path: Path,
}

impl Parse for MaterialAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut shader = None;
        let mut crate_path = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "shader" => {
                    let lit: LitStr = input.parse()?;
                    shader = Some(lit.value());
                }
                "crate_path" => {
                    let lit: LitStr = input.parse()?;
                    crate_path = Some(syn::parse_str(&lit.value())?);
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown attribute `{key}`"),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            shader: shader.ok_or_else(|| input.error("missing required attribute `shader`"))?,
            crate_path: crate_path
                .unwrap_or_else(|| syn::parse_str("myth_resources").expect("valid path")),
        })
    }
}

// ============================================================================
// Parsed Material Definition
// ============================================================================

/// Complete material definition extracted from the annotated struct.
pub struct MaterialDef {
    pub vis: Visibility,
    pub name: Ident,
    pub shader: String,
    pub crate_path: Path,
    pub uniform_fields: Vec<UniformField>,
    pub texture_fields: Vec<TextureField>,
    pub internal_fields: Vec<InternalField>,
}

/// A field marked with `#[uniform]` — maps to a getter/setter on the uniform buffer.
pub struct UniformField {
    pub name: Ident,
    pub ty: Type,
    pub docs: Vec<Attribute>,
    pub default_expr: Option<Expr>,
    /// When `true`, the field is included in the uniform struct but
    /// no public getter/setter is generated.
    pub hidden: bool,
}

/// A field marked with `#[texture]` — generates a texture slot in the texture set.
pub struct TextureField {
    pub name: Ident,
    pub docs: Vec<Attribute>,
}

/// A field marked with `#[internal]` — preserved as-is in the generated struct.
pub struct InternalField {
    pub vis: Visibility,
    pub name: Ident,
    pub ty: Type,
    pub default_expr: Option<Expr>,
    pub clone_expr: Option<Expr>,
    pub docs: Vec<Attribute>,
}

impl MaterialDef {
    /// Builds a `MaterialDef` from the parsed attribute arguments and struct input.
    pub fn from_input(attrs: MaterialAttrs, input: DeriveInput) -> syn::Result<Self> {
        let vis = input.vis;
        let name = input.ident;

        let fields = match input.data {
            syn::Data::Struct(data) => match data.fields {
                Fields::Named(named) => named.named,
                _ => {
                    return Err(syn::Error::new(
                        name.span(),
                        "myth_material only supports structs with named fields",
                    ));
                }
            },
            _ => {
                return Err(syn::Error::new(
                    name.span(),
                    "myth_material can only be applied to structs",
                ));
            }
        };

        let mut uniform_fields = Vec::new();
        let mut texture_fields = Vec::new();
        let mut internal_fields = Vec::new();

        for field in fields {
            let field_name = field
                .ident
                .clone()
                .expect("named struct fields always have idents");
            let kind = classify_field(&field)?;

            let docs: Vec<Attribute> = field
                .attrs
                .iter()
                .filter(|a| a.path().is_ident("doc"))
                .cloned()
                .collect();

            match kind {
                FieldKind::Uniform {
                    default_expr,
                    hidden,
                } => {
                    uniform_fields.push(UniformField {
                        name: field_name,
                        ty: field.ty.clone(),
                        docs,
                        default_expr,
                        hidden,
                    });
                }
                FieldKind::Texture => {
                    texture_fields.push(TextureField {
                        name: field_name,
                        docs,
                    });
                }
                FieldKind::Internal {
                    default_expr,
                    clone_expr,
                } => {
                    internal_fields.push(InternalField {
                        vis: field.vis.clone(),
                        name: field_name,
                        ty: field.ty.clone(),
                        default_expr,
                        clone_expr,
                        docs,
                    });
                }
            }
        }

        Ok(Self {
            vis,
            name,
            shader: attrs.shader,
            crate_path: attrs.crate_path,
            uniform_fields,
            texture_fields,
            internal_fields,
        })
    }

    /// Returns the texture set struct name (e.g., `UnlitMaterial` → `UnlitTextureSet`).
    pub fn texture_set_name(&self) -> Ident {
        let s = self.name.to_string();
        let base = s.strip_suffix("Material").unwrap_or(&s);
        quote::format_ident!("{}TextureSet", base)
    }

    /// Returns the uniform struct name (e.g., `UnlitMaterial` → `UnlitUniforms`).
    pub fn uniform_struct_name(&self) -> Ident {
        let s = self.name.to_string();
        let base = s.strip_suffix("Material").unwrap_or(&s);
        quote::format_ident!("{}Uniforms", base)
    }

    /// Returns whether any uniform field is named `alpha_test`.
    pub fn has_alpha_test(&self) -> bool {
        self.uniform_fields.iter().any(|f| f.name == "alpha_test")
    }
}

// ============================================================================
// Field Classification
// ============================================================================
#[allow(clippy::large_enum_variant)]
enum FieldKind {
    Uniform {
        default_expr: Option<Expr>,
        hidden: bool,
    },
    Texture,
    Internal {
        default_expr: Option<Expr>,
        clone_expr: Option<Expr>,
    },
}

/// Reads field attributes to determine whether it is a uniform, texture, or internal field.
fn classify_field(field: &Field) -> syn::Result<FieldKind> {
    let mut is_uniform = false;
    let mut is_texture = false;
    let mut is_internal = false;
    let mut default_expr = None;
    let mut clone_expr = None;
    let mut hidden = false;

    for attr in &field.attrs {
        if attr.path().is_ident("uniform") {
            is_uniform = true;
            if let syn::Meta::List(_) = &attr.meta {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("default") {
                        let value = meta.value()?;
                        let lit: LitStr = value.parse()?;
                        default_expr = Some(syn::parse_str(&lit.value())?);
                    } else if meta.path.is_ident("hidden") {
                        hidden = true;
                    } else {
                        return Err(meta.error("unknown uniform attribute"));
                    }
                    Ok(())
                })?;
            }
        } else if attr.path().is_ident("texture") {
            is_texture = true;
        } else if attr.path().is_ident("internal") {
            is_internal = true;
            if let syn::Meta::List(_) = &attr.meta {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("default") {
                        let value = meta.value()?;
                        let lit: LitStr = value.parse()?;
                        default_expr = Some(syn::parse_str(&lit.value())?);
                    } else if meta.path.is_ident("clone_with") {
                        let value = meta.value()?;
                        let lit: LitStr = value.parse()?;
                        clone_expr = Some(syn::parse_str(&lit.value())?);
                    } else {
                        return Err(meta.error("unknown internal attribute"));
                    }
                    Ok(())
                })?;
            }
        }
    }

    let field_name = field.ident.as_ref().expect("named field");
    let count = u8::from(is_uniform) + u8::from(is_texture) + u8::from(is_internal);

    if count == 0 {
        return Err(syn::Error::new(
            field_name.span(),
            format!(
                "field `{field_name}` must have one of: #[uniform], #[texture], or #[internal]"
            ),
        ));
    }
    if count > 1 {
        return Err(syn::Error::new(
            field_name.span(),
            format!("field `{field_name}` cannot have multiple kind attributes"),
        ));
    }

    if is_uniform {
        Ok(FieldKind::Uniform {
            default_expr,
            hidden,
        })
    } else if is_texture {
        Ok(FieldKind::Texture)
    } else {
        Ok(FieldKind::Internal {
            default_expr,
            clone_expr,
        })
    }
}
