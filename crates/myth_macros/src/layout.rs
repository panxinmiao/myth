//! Std140 memory layout engine for GPU data structures.
//!
//! Computes field offsets and automatic padding insertion following WebGPU's
//! `std140` uniform buffer layout rules. Shared by both [`#[myth_material]`]
//! and [`#[gpu_struct]`] macros.
//!
//! # Std140 Alignment Rules
//!
//! | Type           | Size (bytes) | Alignment (bytes) |
//! |----------------|-------------:|------------------:|
//! | `f32/u32/i32`  |            4 |                 4 |
//! | `Vec2`         |            8 |                 8 |
//! | `Vec3`         |           12 |                16 |
//! | `Vec4/UVec4`   |           16 |                16 |
//! | `Mat3Uniform`  |           48 |                16 |
//! | `Mat4`         |           64 |                16 |
//! | `array<T, N>`  | stride × N   |                16 |
//!
//! Arrays use element stride rounded up to 16-byte multiples.

use proc_macro2::Ident;
use quote::format_ident;
use syn::{GenericArgument, PathArguments, Type};

// ============================================================================
// Type Layout Query
// ============================================================================

/// Returns `(size, alignment)` in bytes for a known GPU type under std140 rules.
///
/// Returns `None` for types not recognized as GPU-compatible. The returned
/// `size` matches the Rust `#[repr(C)]` size of the type so that offset
/// tracking in [`compute_std140_layout`] stays consistent with the actual
/// struct layout emitted by the compiler.
pub fn type_layout(ty: &Type) -> Option<(usize, usize)> {
    let name = type_last_segment(ty)?;
    match name.as_str() {
        "f32" | "u32" | "i32" => Some((4, 4)),
        "Vec2" => Some((8, 8)),
        "Vec3" => Some((12, 16)),
        "Vec4" | "UVec4" => Some((16, 16)),
        "Mat3Uniform" | "Mat3Padded" => Some((48, 16)),
        "Mat4" => Some((64, 16)),
        "UniformArray" => uniform_array_layout(ty),
        _ => None,
    }
}

/// Computes layout for `UniformArray<T, N>`.
///
/// In std140, array element stride is rounded up to a 16-byte multiple. This
/// function only supports element types whose Rust size already equals the
/// std140 stride (i.e., element size ≥ 16). For smaller elements the Rust
/// `#[repr(transparent)]` wrapper would be smaller than the std140 array,
/// causing a layout mismatch.
fn uniform_array_layout(ty: &Type) -> Option<(usize, usize)> {
    let Type::Path(tp) = ty else { return None };
    let seg = tp.path.segments.last()?;
    let PathArguments::AngleBracketed(ref args) = seg.arguments else {
        return None;
    };

    let mut iter = args.args.iter();

    // First generic argument: element type T
    let GenericArgument::Type(elem_ty) = iter.next()? else {
        return None;
    };

    // Second generic argument: array length N (const)
    let count = match iter.next()? {
        GenericArgument::Const(syn::Expr::Lit(lit)) => {
            if let syn::Lit::Int(int_lit) = &lit.lit {
                int_lit.base10_parse::<usize>().ok()?
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let (elem_size, _elem_align) = type_layout(elem_ty)?;

    // Std140 stride: each array element is rounded up to a 16-byte multiple.
    let stride = round_up(elem_size, 16);

    // Reject arrays where Rust size != std140 size (element < 16 bytes).
    if stride != elem_size {
        return None;
    }

    let total_size = elem_size * count;
    Some((total_size, 16))
}

/// Rounds `value` up to the nearest multiple of `alignment`.
fn round_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

/// Extracts the last path segment name (e.g., `glam::Vec4` → `"Vec4"`).
pub fn type_last_segment(ty: &Type) -> Option<String> {
    if let Type::Path(tp) = ty {
        tp.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

// ============================================================================
// Layout Data Structures
// ============================================================================

/// A field in the computed GPU struct layout.
///
/// May represent a user-declared field or an auto-inserted padding field.
pub struct LayoutField {
    pub name: Ident,
    pub ty: Type,
    pub is_padding: bool,
    pub default_expr: Option<syn::Expr>,
}

/// Input field descriptor for layout computation.
pub struct FieldInput {
    pub name: Ident,
    pub ty: Type,
    pub default_expr: Option<syn::Expr>,
}

// ============================================================================
// Std140 Layout Computation
// ============================================================================

/// Computes std140 layout for a list of fields, inserting padding as needed.
///
/// When `dynamic_offset` is `true`, the total struct size is additionally
/// padded to a multiple of 256 bytes to satisfy wgpu's dynamic uniform
/// buffer alignment requirement.
pub fn compute_std140_layout(
    fields: &[FieldInput],
    dynamic_offset: bool,
) -> syn::Result<Vec<LayoutField>> {
    let mut result = Vec::new();
    let mut offset: usize = 0;
    let mut pad_idx: usize = 0;

    for field in fields {
        let (size, align) = type_layout(&field.ty).ok_or_else(|| {
            syn::Error::new(
                field.name.span(),
                format!(
                    "unsupported GPU type `{}` — supported: f32, u32, i32, Vec2, Vec3, Vec4, \
                     UVec4, Mat3Uniform, Mat4, UniformArray<T, N> (T ≥ 16 bytes)",
                    type_last_segment(&field.ty).unwrap_or_default()
                ),
            )
        })?;

        let padding = (align - (offset % align)) % align;
        if padding > 0 {
            push_padding(&mut result, padding, &mut pad_idx);
            offset += padding;
        }

        result.push(LayoutField {
            name: field.name.clone(),
            ty: field.ty.clone(),
            is_padding: false,
            default_expr: field.default_expr.clone(),
        });
        offset += size;
    }

    // Final alignment to 16 bytes (std140 struct requirement)
    let final_padding = (16 - (offset % 16)) % 16;
    if final_padding > 0 {
        push_padding(&mut result, final_padding, &mut pad_idx);
        offset += final_padding;
    }

    // Dynamic uniform buffer alignment (256 bytes)
    if dynamic_offset {
        let dynamic_padding = (256 - (offset % 256)) % 256;
        if dynamic_padding > 0 {
            let count = dynamic_padding / 4;
            let name = format_ident!("__dynamic_pad");
            let ty: Type = syn::parse_str(&format!("[u32; {count}]")).unwrap();
            result.push(LayoutField {
                name,
                ty,
                is_padding: true,
                default_expr: None,
            });
        }
    }

    Ok(result)
}

/// Inserts a padding field into the layout.
fn push_padding(fields: &mut Vec<LayoutField>, pad_bytes: usize, pad_idx: &mut usize) {
    debug_assert!(
        pad_bytes.is_multiple_of(4),
        "padding must be 4-byte aligned"
    );
    let count = pad_bytes / 4;
    let name = format_ident!("__pad_{}", *pad_idx);
    *pad_idx += 1;
    let ty: Type = if count == 1 {
        syn::parse_str("u32").unwrap()
    } else {
        syn::parse_str(&format!("[u32; {count}]")).unwrap()
    };
    fields.push(LayoutField {
        name,
        ty,
        is_padding: true,
        default_expr: None,
    });
}
