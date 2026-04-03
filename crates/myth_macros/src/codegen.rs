//! Code generation for the `#[myth_material]` attribute macro.
//!
//! Transforms a parsed [`MaterialDef`] into the complete set of types and
//! trait implementations required by the Myth engine rendering pipeline.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::DeriveInput;

use crate::layout::{FieldInput, LayoutField, compute_std140_layout};
use crate::parse::{MaterialAttrs, MaterialDef};

/// Main entry point: parses input and produces all generated code.
pub fn generate(attrs: MaterialAttrs, input: DeriveInput) -> syn::Result<TokenStream> {
    let def = MaterialDef::from_input(attrs, input)?;

    let uniform_struct = gen_uniform_struct(&def)?;
    let texture_set = gen_texture_set(&def);
    let material_struct = gen_material_struct(&def);
    let api_impl = gen_api_impl(&def);
    let clone_impl = gen_clone_impl(&def);
    let material_trait = gen_material_trait(&def);
    let renderable_trait = gen_renderable_trait(&def);

    Ok(quote! {
        #uniform_struct
        #texture_set
        #material_struct
        #api_impl
        #clone_impl
        #material_trait
        #renderable_trait
    })
}

// ============================================================================
// Std140 Layout (delegates to shared layout engine)
// ============================================================================

/// Computes the std140 layout for a material's uniform fields, including
/// auto-generated texture transform fields.
fn compute_layout(def: &MaterialDef) -> syn::Result<Vec<LayoutField>> {
    let mut inputs: Vec<FieldInput> = def
        .uniform_fields
        .iter()
        .map(|uf| FieldInput {
            name: uf.name.clone(),
            ty: uf.ty.clone(),
            default_expr: uf.default_expr.clone(),
        })
        .collect();

    // Append texture transform fields (Mat3Uniform for each texture)
    let mat3_uniform_ty: syn::Type = syn::parse_str("Mat3Uniform").unwrap();
    for tf in &def.texture_fields {
        inputs.push(FieldInput {
            name: format_ident!("{}_transform", tf.name),
            ty: mat3_uniform_ty.clone(),
            default_expr: Some(syn::parse_str("Mat3Uniform::IDENTITY").unwrap()),
        });
    }

    compute_std140_layout(&inputs, false)
}

// ============================================================================
// Uniform Struct Generation
// ============================================================================

/// Generates the uniform struct with std140 padding, plus Default, WgslType,
/// WgslStruct, and GpuData trait implementations.
fn gen_uniform_struct(def: &MaterialDef) -> syn::Result<TokenStream> {
    let cr = &def.crate_path;
    let uniforms_name = def.uniform_struct_name();
    let uniforms_name_str = uniforms_name.to_string();
    let layout = compute_layout(def)?;

    // --- Struct fields ---
    let struct_fields = layout.iter().map(|f| {
        let name = &f.name;
        let ty = &f.ty;
        if f.is_padding {
            quote! {
                #[doc(hidden)]
                pub #name: #ty,
            }
        } else {
            quote! { pub #name: #ty, }
        }
    });

    // --- Default impl ---
    let default_fields = layout.iter().map(|f| {
        let name = &f.name;
        let ty = &f.ty;
        if let Some(expr) = &f.default_expr {
            quote! { #name: #expr, }
        } else {
            quote! { #name: <#ty as Default>::default(), }
        }
    });

    // --- WgslType: collect_wgsl_defs entries (skip padding) ---
    let wgsl_collect_fields = layout.iter().filter(|f| !f.is_padding).map(|f| {
        let ty = &f.ty;
        quote! {
            <#ty as #cr::uniforms::WgslType>::collect_wgsl_defs(defs, inserted);
        }
    });

    // --- WgslType: struct body lines (skip padding) ---
    let wgsl_body_fields = layout.iter().filter(|f| !f.is_padding).map(|f| {
        let name_str = f.name.to_string();
        let ty = &f.ty;
        quote! {
            let _ = std::fmt::Write::write_fmt(
                &mut code,
                format_args!("    {}: {},\n", #name_str, <#ty as #cr::uniforms::WgslType>::wgsl_type_name()),
            );
        }
    });

    // --- WgslStruct: same collect + body but with struct_name parameter ---
    let wgsl_struct_collect = layout.iter().filter(|f| !f.is_padding).map(|f| {
        let ty = &f.ty;
        quote! {
            <#ty as #cr::uniforms::WgslType>::collect_wgsl_defs(&mut defs, &mut inserted);
        }
    });
    let wgsl_struct_body = layout.iter().filter(|f| !f.is_padding).map(|f| {
        let name_str = f.name.to_string();
        let ty = &f.ty;
        quote! {
            let _ = std::fmt::Write::write_fmt(
                &mut code,
                format_args!("    {}: {},\n", #name_str, <#ty as #cr::uniforms::WgslType>::wgsl_type_name()),
            );
        }
    });

    Ok(quote! {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct #uniforms_name {
            #(#struct_fields)*
        }

        impl Default for #uniforms_name {
            fn default() -> Self {
                Self {
                    #(#default_fields)*
                }
            }
        }

        impl #cr::uniforms::WgslType for #uniforms_name {
            fn wgsl_type_name() -> std::borrow::Cow<'static, str> {
                #uniforms_name_str.into()
            }

            fn collect_wgsl_defs(defs: &mut Vec<String>, inserted: &mut std::collections::HashSet<String>) {
                #(#wgsl_collect_fields)*

                let my_name = #uniforms_name_str;
                if !inserted.contains(my_name) {
                    let mut code = format!("struct {} {{\n", my_name);
                    #(#wgsl_body_fields)*
                    code.push_str("};\n");
                    defs.push(code);
                    inserted.insert(my_name.to_string());
                }
            }
        }

        impl #cr::uniforms::WgslStruct for #uniforms_name {
            fn wgsl_struct_def(struct_name: &str) -> String {
                let mut defs = Vec::new();
                let mut inserted = std::collections::HashSet::new();

                #(#wgsl_struct_collect)*

                let mut code = format!("struct {} {{\n", struct_name);
                #(#wgsl_struct_body)*
                code.push_str("};\n");
                defs.push(code);
                defs.join("\n")
            }
        }

        impl #cr::buffer::GpuData for #uniforms_name {
            fn as_bytes(&self) -> &[u8] {
                bytemuck::bytes_of(self)
            }

            fn byte_size(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }
    })
}

// ============================================================================
// TextureSet Struct
// ============================================================================

/// Generates the `{Name}TextureSet` struct containing all texture slot fields.
fn gen_texture_set(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let texture_set_name = def.texture_set_name();

    let fields = def.texture_fields.iter().map(|f| {
        let name = &f.name;
        let docs = &f.docs;
        quote! {
            #(#docs)*
            pub #name: #cr::material::TextureSlot,
        }
    });

    quote! {
        #[derive(Clone, Default, Debug)]
        pub struct #texture_set_name {
            #(#fields)*
        }
    }
}

// ============================================================================
// Material Struct (Rewritten)
// ============================================================================

/// Generates the material struct with engine-internal fields.
fn gen_material_struct(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let vis = &def.vis;
    let name = &def.name;
    let uniforms_type = def.uniform_struct_name();
    let texture_set_name = def.texture_set_name();

    let internal_fields = def.internal_fields.iter().map(|f| {
        let fvis = &f.vis;
        let fname = &f.name;
        let fty = &f.ty;
        let docs = &f.docs;
        quote! {
            #(#docs)*
            #fvis #fname: #fty,
        }
    });

    quote! {
        #[derive(Debug)]
        #vis struct #name {
            #[doc(hidden)]
            pub uniforms: #cr::buffer::CpuBuffer<#uniforms_type>,
            #[doc(hidden)]
            pub settings: parking_lot::RwLock<#cr::material::MaterialSettings>,
            #[doc(hidden)]
            pub textures: parking_lot::RwLock<#texture_set_name>,
            pub(crate) version: std::sync::atomic::AtomicU64,
            /// When `true`, texture transforms are automatically flushed to uniforms.
            pub auto_sync_texture_to_uniforms: bool,
            #(#internal_fields)*
        }
    }
}

// ============================================================================
// API Implementation Block
// ============================================================================

/// Generates the full `impl` block with constructor, settings, accessors, and utilities.
fn gen_api_impl(def: &MaterialDef) -> TokenStream {
    let name = &def.name;

    let constructor = gen_constructor(def);
    let settings_api = gen_settings_api(def);
    let uniform_accessors = gen_uniform_accessors(def);
    let texture_accessors = gen_texture_accessors(def);
    let flush_transforms = gen_flush_transforms(def);
    let utility_methods = gen_utility_methods(def);

    quote! {
        impl #name {
            #constructor
            #settings_api
            #uniform_accessors
            #texture_accessors
            #flush_transforms
            #utility_methods
        }
    }
}

// ============================================================================
// Constructor
// ============================================================================

/// Generates `from_uniforms(uniforms) -> Self`.
fn gen_constructor(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let uniforms_type = def.uniform_struct_name();
    let texture_set_name = def.texture_set_name();

    let uniforms_label = uniforms_type.to_string();

    let internal_inits = def.internal_fields.iter().map(|f| {
        let fname = &f.name;
        if let Some(default) = &f.default_expr {
            quote! { #fname: #default, }
        } else {
            quote! { #fname: Default::default(), }
        }
    });

    quote! {
        /// Creates a new material instance from uniform data with default settings.
        pub fn from_uniforms(uniforms: #uniforms_type) -> Self {
            Self {
                uniforms: #cr::buffer::CpuBuffer::new(
                    uniforms,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    Some(#uniforms_label),
                ),
                settings: parking_lot::RwLock::new(
                    #cr::material::MaterialSettings::default(),
                ),
                version: std::sync::atomic::AtomicU64::new(0),
                textures: parking_lot::RwLock::new(#texture_set_name::default()),
                auto_sync_texture_to_uniforms: false,
                #(#internal_inits)*
            }
        }
    }
}

// ============================================================================
// Settings API
// ============================================================================

/// Generates settings accessor methods (alpha_mode, side, depth_test, depth_write).
fn gen_settings_api(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;

    quote! {
        /// Returns a RAII guard for batch-modifying material settings.
        ///
        /// The guard automatically increments the material version on drop
        /// if any settings changed, triggering pipeline cache invalidation.
        pub fn settings_mut(&self) -> #cr::material::SettingsGuard<'_> {
            #cr::material::SettingsGuard::new(
                self.settings.write(),
                &self.version,
            )
        }

        /// Returns a snapshot of the current material settings.
        pub fn settings(&self) -> #cr::material::MaterialSettings {
            *self.settings.read()
        }

        /// Sets the alpha blending mode.
        pub fn set_alpha_mode(&self, mode: #cr::material::AlphaMode) {
            self.settings_mut().alpha_mode = mode;
        }

        /// Returns the current alpha blending mode.
        pub fn alpha_mode(&self) -> #cr::material::AlphaMode {
            self.settings.read().alpha_mode
        }

        /// Sets the face culling mode (Front/Back/Double).
        pub fn set_side(&self, side: #cr::material::Side) {
            self.settings_mut().side = side;
        }

        /// Returns the current face culling mode.
        pub fn side(&self) -> #cr::material::Side {
            self.settings.read().side
        }

        /// Enables or disables depth testing.
        pub fn set_depth_test(&self, depth_test: bool) {
            self.settings_mut().depth_test = depth_test;
        }

        /// Returns whether depth testing is enabled.
        pub fn depth_test(&self) -> bool {
            self.settings.read().depth_test
        }

        /// Enables or disables depth buffer writing.
        ///
        /// For transparent objects, it is usually recommended to disable this.
        pub fn set_depth_write(&self, depth_write: bool) {
            self.settings_mut().depth_write = depth_write;
        }

        /// Returns whether depth buffer writing is enabled.
        pub fn depth_write(&self) -> bool {
            self.settings.read().depth_write
        }
    }
}

// ============================================================================
// Uniform Accessors
// ============================================================================

/// Generates per-field getters and setters with double-check locking.
fn gen_uniform_accessors(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let uniforms_type = def.uniform_struct_name();

    let accessors = def.uniform_fields.iter().filter(|f| !f.hidden).map(|f| {
        let name = &f.name;
        let ty = &f.ty;
        let setter_name = format_ident!("set_{}", name);
        let docs = &f.docs;

        quote! {
            #(#docs)*
            #[allow(clippy::float_cmp)]
            pub fn #setter_name(&self, value: #ty) {
                // Fast path: shared read lock — minimal contention
                if self.uniforms.read().#name == value {
                    return;
                }
                // Slow path: exclusive write lock only when modification is needed
                let mut guard = self.uniforms.write();
                // Double-check under write lock to prevent concurrent races
                if guard.#name != value {
                    guard.#name = value;
                } else {
                    guard.skip_sync();
                }
            }

            #(#docs)*
            pub fn #name(&self) -> #ty {
                self.uniforms.read().#name
            }
        }
    });

    quote! {
        /// Returns a write guard for batch-modifying uniform parameters.
        ///
        /// The guard automatically marks data as dirty on drop,
        /// triggering GPU buffer synchronization.
        pub fn uniforms_mut(&self) -> #cr::buffer::CheckedBufferGuard<'_, #uniforms_type> {
            self.uniforms.write_checked()
        }

        /// Returns a read guard for accessing uniform parameters.
        pub fn uniforms(&self) -> #cr::buffer::BufferReadGuard<'_, #uniforms_type> {
            self.uniforms.read()
        }

        #(#accessors)*
    }
}

// ============================================================================
// Texture Accessors
// ============================================================================

/// Generates per-slot texture getters, setters, and configure methods.
fn gen_texture_accessors(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;

    let accessors = def.texture_fields.iter().map(|f| {
        let name = &f.name;
        let setter_name = format_ident!("set_{}", name);
        let transform_setter = format_ident!("set_{}_transform", name);
        let configure_name = format_ident!("configure_{}", name);
        let docs = &f.docs;

        quote! {
            #(#docs)*
            pub fn #setter_name(&self, value: Option<#cr::TextureHandle>) {
                let mut tex_data = self.textures.write();
                if tex_data.#name.texture != value {
                    tex_data.#name.texture = value;
                    self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            #(#docs)*
            pub fn #name(&self) -> Option<#cr::TextureHandle> {
                self.textures.read().#name.texture.clone()
            }

            /// Sets the UV transform for this texture slot.
            pub fn #transform_setter(&self, transform: #cr::material::TextureTransform) {
                self.textures.write().#name.transform = transform;
            }

            /// Configures this texture slot via a closure for batch modifications.
            ///
            /// The closure receives a mutable reference to the [`TextureSlot`].
            /// Version is automatically bumped if the texture or UV channel changes.
            pub fn #configure_name<F>(&self, f: F)
            where
                F: FnOnce(&mut #cr::material::TextureSlot),
            {
                let mut tex_data = self.textures.write();
                let slot = &mut tex_data.#name;
                let old_texture = slot.texture.clone();
                let old_channel = slot.channel;
                f(slot);
                if slot.texture != old_texture || slot.channel != old_channel {
                    self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    });

    quote! { #(#accessors)* }
}

// ============================================================================
// Flush Texture Transforms
// ============================================================================

/// Generates `flush_texture_transforms` that syncs UV matrices to the uniform buffer.
fn gen_flush_transforms(def: &MaterialDef) -> TokenStream {
    if def.texture_fields.is_empty() {
        return quote! {
            /// Flushes texture transform matrices to the uniform buffer.
            pub fn flush_texture_transforms(&self) -> bool {
                false
            }
        };
    }

    let transform_checks = def.texture_fields.iter().map(|f| {
        let name = &f.name;
        let transform_field = format_ident!("{}_transform", name);
        quote! {
            {
                let new_matrix = tex_data.#name.compute_matrix();
                if uniforms.#transform_field != new_matrix {
                    uniforms.#transform_field = new_matrix;
                    changed = true;
                }
            }
        }
    });

    quote! {
        /// Flushes texture transform matrices to the uniform buffer.
        ///
        /// Only writes when values actually change, avoiding unnecessary
        /// version bumps. Returns whether any data was updated.
        pub fn flush_texture_transforms(&self) -> bool {
            let mut changed = false;
            let tex_data = self.textures.read();
            let mut uniforms = self.uniforms.write();
            #(#transform_checks)*
            if !changed {
                uniforms.skip_sync();
            }
            changed
        }
    }
}

// ============================================================================
// Utility Methods
// ============================================================================

/// Generates `configure` and `notify_pipeline_dirty` methods.
fn gen_utility_methods(def: &MaterialDef) -> TokenStream {
    let uniforms_type = def.uniform_struct_name();

    quote! {
        /// Provides access to uniform data through a closure (under write lock).
        pub fn configure<F>(&self, f: F)
        where
            F: FnOnce(&#uniforms_type),
        {
            let guard = self.uniforms.write();
            f(&*guard);
        }

        /// Manually marks the material pipeline as dirty, forcing a rebuild.
        ///
        /// In most cases this is not needed, as the standard API automatically
        /// tracks version changes. Use this only when:
        /// - After directly modifying internal fields (e.g., in loader code)
        /// - When material configuration has changed but version wasn't updated
        #[inline]
        pub fn notify_pipeline_dirty(&self) {
            self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Clone Implementation
// ============================================================================

/// Generates a manual `Clone` impl with atomic version snapshot and custom field cloning.
fn gen_clone_impl(def: &MaterialDef) -> TokenStream {
    let name = &def.name;

    let internal_clones = def.internal_fields.iter().map(|f| {
        let fname = &f.name;
        if let Some(clone_expr) = &f.clone_expr {
            quote! { #fname: (#clone_expr)(self), }
        } else {
            quote! { #fname: self.#fname.clone(), }
        }
    });

    quote! {
        impl Clone for #name {
            fn clone(&self) -> Self {
                use std::sync::atomic::Ordering;
                Self {
                    uniforms: self.uniforms.clone(),
                    settings: parking_lot::RwLock::new(self.settings.read().clone()),
                    textures: parking_lot::RwLock::new(self.textures.read().clone()),
                    version: std::sync::atomic::AtomicU64::new(
                        self.version.load(Ordering::Relaxed),
                    ),
                    auto_sync_texture_to_uniforms: self.auto_sync_texture_to_uniforms,
                    #(#internal_clones)*
                }
            }
        }
    }
}

// ============================================================================
// MaterialTrait Implementation
// ============================================================================

/// Generates the `MaterialTrait` implementation (type identity and downcasting).
fn gen_material_trait(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let name = &def.name;

    quote! {
        impl #cr::material::MaterialTrait for #name {
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
    }
}

// ============================================================================
// RenderableMaterialTrait Implementation
// ============================================================================

/// Generates the full `RenderableMaterialTrait` implementation including
/// shader defines, texture visitation, and GPU resource binding.
fn gen_renderable_trait(def: &MaterialDef) -> TokenStream {
    let cr = &def.crate_path;
    let name = &def.name;
    let uniforms_type = def.uniform_struct_name();
    let shader = &def.shader;

    let has_textures = !def.texture_fields.is_empty();

    // --- shader_defines: texture-based macro generation ---
    let shader_defines_texture_code = if has_textures {
        let defines = def.texture_fields.iter().map(|f| {
            let fname = &f.name;
            let upper = fname.to_string().to_uppercase();
            let has_key = format!("HAS_{upper}");
            let uv_key = format!("{upper}_UV");
            quote! {
                if tex_data.#fname.texture.is_some() {
                    defines.set(#has_key, "1");
                    if tex_data.#fname.channel > 0 {
                        defines.set(#uv_key, &tex_data.#fname.channel.to_string());
                    }
                }
            }
        });

        quote! {
            let tex_data = self.textures.read();
            #(#defines)*
        }
    } else {
        quote! {}
    };

    // --- visit_textures ---
    let visit_textures_code = if has_textures {
        let visits = def.texture_fields.iter().map(|f| {
            let fname = &f.name;
            quote! {
                if let Some(handle) = &tex_data.#fname.texture {
                    visitor(&#cr::texture::TextureSource::Asset(*handle));
                }
            }
        });
        quote! {
            let tex_data = self.textures.read();
            #(#visits)*
        }
    } else {
        quote! {}
    };

    // --- define_bindings: GPU resource declarations ---
    let define_bindings_texture_code = if has_textures {
        let bindings = def.texture_fields.iter().map(|f| {
            let fname = &f.name;
            let binding_name = fname.to_string();
            quote! {
                if let Some(handle) = &tex_data.#fname.texture {
                    builder.add_texture(
                        #binding_name,
                        Some(#cr::texture::TextureSource::Asset(*handle)),
                        wgpu::TextureSampleType::Float { filterable: true },
                        wgpu::TextureViewDimension::D2,
                        wgpu::ShaderStages::FRAGMENT,
                    );
                }
            }
        });
        quote! {
            let tex_data = self.textures.read();
            #(#bindings)*
        }
    } else {
        quote! {}
    };

    quote! {
        impl #cr::material::RenderableMaterialTrait for #name {
            fn shader_name(&self) -> &'static str {
                #shader
            }

            fn version(&self) -> u64 {
                self.version.load(std::sync::atomic::Ordering::Relaxed)
            }

            fn settings(&self) -> #cr::material::MaterialSettings {
                *self.settings.read()
            }

            fn uniform_buffer(&self) -> #cr::buffer::BufferRef {
                self.uniforms.handle()
            }

            fn with_uniform_bytes(&self, visitor: &mut dyn FnMut(&[u8])) {
                use #cr::buffer::GpuData;
                let guard = self.uniforms.read();
                visitor(guard.as_bytes());
            }

            fn shader_defines(&self) -> #cr::shader_defines::ShaderDefines {
                let mut defines = #cr::shader_defines::ShaderDefines::new();
                #shader_defines_texture_code
                self.settings.read().generate_shader_defines(&mut defines);
                self.extra_defines(&mut defines);
                defines
            }

            fn visit_textures(&self, visitor: &mut dyn FnMut(&#cr::texture::TextureSource)) {
                #visit_textures_code
            }

            fn define_bindings<'a>(
                &'a self,
                builder: &mut #cr::builder::ResourceBuilder<'a>,
            ) {
                builder.add_uniform::<#uniforms_type>(
                    "material",
                    &self.uniforms,
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                );
                #define_bindings_texture_code
            }
        }
    }
}
