//! Code generation for the `#[gpu_struct]` attribute macro.
//!
//! Transforms a parsed [`GpuStructDef`] into a `#[repr(C)]` struct with
//! automatic std140 padding, plus implementations of `Default`, `WgslType`,
//! `WgslStruct`, and `GpuData`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

use crate::gpu_struct_parse::{GpuStructAttrs, GpuStructDef};
use crate::layout::{FieldInput, LayoutField, compute_std140_layout};

/// Main entry point: parses input and produces all generated code.
pub fn generate(attrs: GpuStructAttrs, input: DeriveInput) -> syn::Result<TokenStream> {
    let def = GpuStructDef::from_input(attrs, input)?;
    let layout = build_layout(&def)?;

    let struct_def = gen_struct(&def, &layout);
    let default_impl = gen_default(&def, &layout);
    let wgsl_type_impl = gen_wgsl_type(&def, &layout);
    let wgsl_struct_impl = gen_wgsl_struct(&def, &layout);
    let gpu_data_impl = gen_gpu_data(&def);

    Ok(quote! {
        #struct_def
        #default_impl
        #wgsl_type_impl
        #wgsl_struct_impl
        #gpu_data_impl
    })
}

/// Builds the std140 field layout from the parsed definition.
fn build_layout(def: &GpuStructDef) -> syn::Result<Vec<LayoutField>> {
    let inputs: Vec<FieldInput> = def
        .fields
        .iter()
        .map(|f| FieldInput {
            name: f.name.clone(),
            ty: f.ty.clone(),
            default_expr: f.default_expr.clone(),
        })
        .collect();

    compute_std140_layout(&inputs, def.dynamic_offset)
}

/// Returns `true` if a field should be visible in the generated WGSL struct.
///
/// Padding fields and fields whose names start with `__` are excluded from
/// WGSL output.
fn wgsl_visible(field: &LayoutField) -> bool {
    !field.is_padding && !field.name.to_string().starts_with("__")
}

// ============================================================================
// Struct Definition
// ============================================================================

/// Generates the struct definition with std140 padding fields interleaved.
fn gen_struct(def: &GpuStructDef, layout: &[LayoutField]) -> TokenStream {
    let vis = &def.vis;
    let name = &def.name;
    let struct_docs = &def.struct_docs;

    let fields = layout.iter().map(|f| {
        let fname = &f.name;
        let fty = &f.ty;
        if f.is_padding {
            quote! {
                #[doc(hidden)]
                pub #fname: #fty,
            }
        } else {
            // Find the original field to preserve visibility and docs
            let orig = def.fields.iter().find(|of| of.name == *fname);
            if let Some(orig) = orig {
                let fvis = &orig.vis;
                let docs = &orig.docs;
                quote! {
                    #(#docs)*
                    #fvis #fname: #fty,
                }
            } else {
                quote! { pub #fname: #fty, }
            }
        }
    });

    quote! {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        #(#struct_docs)*
        #vis struct #name {
            #(#fields)*
        }
    }
}

// ============================================================================
// Default Implementation
// ============================================================================

/// Generates the `Default` trait implementation.
///
/// Uses `#[default(...)]` expressions for annotated fields and
/// `Default::default()` for the rest. Padding fields are zero-initialized.
fn gen_default(def: &GpuStructDef, layout: &[LayoutField]) -> TokenStream {
    let name = &def.name;

    let field_defaults = layout.iter().map(|f| {
        let fname = &f.name;
        let fty = &f.ty;
        if f.is_padding {
            // Padding fields are always zeroed
            quote! { #fname: <#fty as Default>::default(), }
        } else if let Some(expr) = &f.default_expr {
            quote! { #fname: #expr, }
        } else {
            quote! { #fname: <#fty as Default>::default(), }
        }
    });

    quote! {
        impl Default for #name {
            fn default() -> Self {
                Self {
                    #(#field_defaults)*
                }
            }
        }
    }
}

// ============================================================================
// WgslType Implementation
// ============================================================================

/// Generates the `WgslType` trait implementation for nested struct support.
fn gen_wgsl_type(def: &GpuStructDef, layout: &[LayoutField]) -> TokenStream {
    let cr = &def.crate_path;
    let name = &def.name;
    let name_str = name.to_string();

    // Collect WGSL definitions from all non-padding user fields
    let collect_fields = layout.iter().filter(|f| wgsl_visible(f)).map(|f| {
        let ty = &f.ty;
        quote! {
            <#ty as #cr::uniforms::WgslType>::collect_wgsl_defs(defs, inserted);
        }
    });

    // Build WGSL struct body (only WGSL-visible fields)
    let body_fields = layout.iter().filter(|f| wgsl_visible(f)).map(|f| {
        let field_name_str = f.name.to_string();
        let ty = &f.ty;
        quote! {
            let _ = std::fmt::Write::write_fmt(
                &mut code,
                format_args!(
                    "    {}: {},\n",
                    #field_name_str,
                    <#ty as #cr::uniforms::WgslType>::wgsl_type_name(),
                ),
            );
        }
    });

    quote! {
        impl #cr::uniforms::WgslType for #name {
            fn wgsl_type_name() -> std::borrow::Cow<'static, str> {
                #name_str.into()
            }

            fn collect_wgsl_defs(
                defs: &mut Vec<String>,
                inserted: &mut std::collections::HashSet<String>,
            ) {
                #(#collect_fields)*

                let my_name = #name_str;
                if !inserted.contains(my_name) {
                    let mut code = format!("struct {} {{\n", my_name);
                    #(#body_fields)*
                    code.push_str("};\n");
                    defs.push(code);
                    inserted.insert(my_name.to_string());
                }
            }
        }
    }
}

// ============================================================================
// WgslStruct Implementation
// ============================================================================

/// Generates the `WgslStruct` trait implementation for top-level binding.
fn gen_wgsl_struct(def: &GpuStructDef, layout: &[LayoutField]) -> TokenStream {
    let cr = &def.crate_path;
    let name = &def.name;

    // Collect nested definitions from non-padding fields
    let collect_deps = layout.iter().filter(|f| wgsl_visible(f)).map(|f| {
        let ty = &f.ty;
        quote! {
            <#ty as #cr::uniforms::WgslType>::collect_wgsl_defs(&mut defs, &mut inserted);
        }
    });

    // Build WGSL struct body (WGSL-visible fields only)
    let body_fields = layout.iter().filter(|f| wgsl_visible(f)).map(|f| {
        let field_name_str = f.name.to_string();
        let ty = &f.ty;
        quote! {
            let _ = std::fmt::Write::write_fmt(
                &mut code,
                format_args!(
                    "    {}: {},\n",
                    #field_name_str,
                    <#ty as #cr::uniforms::WgslType>::wgsl_type_name(),
                ),
            );
        }
    });

    quote! {
        impl #cr::uniforms::WgslStruct for #name {
            fn wgsl_struct_def(struct_name: &str) -> String {
                let mut defs = Vec::new();
                let mut inserted = std::collections::HashSet::new();

                #(#collect_deps)*

                let mut code = format!("struct {} {{\n", struct_name);
                #(#body_fields)*
                code.push_str("};\n");
                defs.push(code);
                defs.join("\n")
            }
        }
    }
}

// ============================================================================
// GpuData Implementation
// ============================================================================

/// Generates the `GpuData` trait implementation for byte-level GPU upload.
fn gen_gpu_data(def: &GpuStructDef) -> TokenStream {
    let cr = &def.crate_path;
    let name = &def.name;

    quote! {
        impl #cr::buffer::GpuData for #name {
            fn as_bytes(&self) -> &[u8] {
                bytemuck::bytes_of(self)
            }

            fn byte_size(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }
    }
}
