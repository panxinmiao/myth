//! AST parsing for the `#[gpu_struct]` attribute macro.
//!
//! Extracts struct-level configuration (`dynamic_offset`, `crate_path`) and
//! per-field attributes (`#[default(...)]`) from the annotated struct.

use syn::{
    Attribute, DeriveInput, Expr, Fields, Ident, LitStr, Path, Token, Type, Visibility,
    parse::{Parse, ParseStream},
};

// ============================================================================
// Struct-level Attributes
// ============================================================================

/// Configuration parsed from `#[gpu_struct(...)]`.
pub struct GpuStructAttrs {
    /// When `true`, pad the struct to a 256-byte multiple for dynamic uniform
    /// buffer binding.
    pub dynamic_offset: bool,
    /// Path to the `myth_resources` crate (default: `myth_resources`).
    pub crate_path: Path,
}

impl Parse for GpuStructAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut dynamic_offset = false;
        let mut crate_path = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "dynamic_offset" => {
                    let lit: syn::LitBool = input.parse()?;
                    dynamic_offset = lit.value();
                }
                "crate_path" => {
                    let lit: LitStr = input.parse()?;
                    crate_path = Some(syn::parse_str(&lit.value())?);
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown gpu_struct attribute `{key}`"),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            dynamic_offset,
            crate_path: crate_path
                .unwrap_or_else(|| syn::parse_str("myth_resources").expect("valid path")),
        })
    }
}

// ============================================================================
// Parsed GPU Struct Definition
// ============================================================================

/// Complete GPU struct definition extracted from the annotated struct.
pub struct GpuStructDef {
    pub vis: Visibility,
    pub name: Ident,
    pub dynamic_offset: bool,
    pub crate_path: Path,
    pub fields: Vec<GpuStructField>,
    /// Doc attributes on the struct itself.
    pub struct_docs: Vec<Attribute>,
}

/// A user-declared field in the GPU struct.
pub struct GpuStructField {
    pub vis: Visibility,
    pub name: Ident,
    pub ty: Type,
    pub default_expr: Option<Expr>,
    pub docs: Vec<Attribute>,
}

impl GpuStructDef {
    /// Builds a [`GpuStructDef`] from the parsed attribute arguments and struct input.
    pub fn from_input(attrs: GpuStructAttrs, input: DeriveInput) -> syn::Result<Self> {
        let vis = input.vis;
        let name = input.ident;

        let struct_docs: Vec<Attribute> = input
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("doc"))
            .cloned()
            .collect();

        let raw_fields = match input.data {
            syn::Data::Struct(data) => match data.fields {
                Fields::Named(named) => named.named,
                _ => {
                    return Err(syn::Error::new(
                        name.span(),
                        "gpu_struct only supports structs with named fields",
                    ));
                }
            },
            _ => {
                return Err(syn::Error::new(
                    name.span(),
                    "gpu_struct can only be applied to structs",
                ));
            }
        };

        let mut fields = Vec::new();

        for field in raw_fields {
            let field_name = field
                .ident
                .clone()
                .expect("named struct fields always have idents");

            let docs: Vec<Attribute> = field
                .attrs
                .iter()
                .filter(|a| a.path().is_ident("doc"))
                .cloned()
                .collect();

            let default_expr = extract_default_attr(&field.attrs, &field_name)?;

            fields.push(GpuStructField {
                vis: field.vis.clone(),
                name: field_name,
                ty: field.ty.clone(),
                default_expr,
                docs,
            });
        }

        Ok(Self {
            vis,
            name,
            dynamic_offset: attrs.dynamic_offset,
            crate_path: attrs.crate_path,
            fields,
            struct_docs,
        })
    }
}

/// Extracts the `#[default(...)]` attribute value from a field's attributes.
///
/// Returns `Some(expr)` if found, `None` otherwise.
fn extract_default_attr(attrs: &[Attribute], field_name: &Ident) -> syn::Result<Option<Expr>> {
    for attr in attrs {
        if attr.path().is_ident("default") {
            let expr: Expr = attr.parse_args().map_err(|e| {
                syn::Error::new(
                    field_name.span(),
                    format!("invalid #[default(...)] on `{field_name}`: {e}"),
                )
            })?;
            return Ok(Some(expr));
        }
    }
    Ok(None)
}
