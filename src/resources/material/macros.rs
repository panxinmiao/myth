/// [Macro 1] API Generator
/// Automatically generates Setters, Getters, and Configure methods.
/// Hides underlying uniforms and bindings fields, exposing only a clean Public API.
///
/// Texture slots use a Guard pattern to automatically track texture presence/absence
/// changes and trigger version updates.
#[macro_export]
macro_rules! impl_material_api {
    (
        $struct_name:ident,
        $uniform_struct:ty,
        // Uniforms: (field_name, type, doc)
        uniforms: [ $(($u_field:ident, $u_type:ty, $u_doc:expr)),* $(,)? ],
        // Textures: (field_name, doc)
        textures: [ $(($t_field:ident, $t_doc:expr)),* $(,)? ],
        manual_clone_fields: { $($m_field:ident : $m_expr:expr),* $(,)? }
    ) => {
        impl $struct_name {

            // ==========================================
            // 1. Common Settings API
            // ==========================================

            pub fn settings_mut(&self) -> $crate::resources::material::SettingsGuard<'_> {
                $crate::resources::material::SettingsGuard::new(
                    self.settings.write(),
                    &self.version
                )
            }

            pub fn settings(&self) -> $crate::resources::material::MaterialSettings {
                *self.settings.read()
            }


            pub fn set_alpha_mode(&self, mode: $crate::resources::material::AlphaMode) {
                self.settings_mut().alpha_mode = mode;
                // Handle associated Uniform logic
                if let $crate::resources::material::AlphaMode::Mask(cutoff, _a2c) = mode {
                     self.uniforms.write().alpha_test = cutoff;
                }
            }

            pub fn alpha_mode(&self) -> $crate::resources::material::AlphaMode {
                self.settings.read().alpha_mode
            }

            /// Sets the face culling mode (Front/Back/Double).
            pub fn set_side(&self, side: $crate::resources::material::Side) {
                self.settings_mut().side = side;
            }
            pub fn side(&self) -> $crate::resources::material::Side {
                self.settings.read().side
            }

            /// Enables or disables depth testing.
            pub fn set_depth_test(&self, depth_test: bool) {
                self.settings_mut().depth_test = depth_test;
            }
            pub fn depth_test(&self) -> bool {
                self.settings.read().depth_test
            }

            /// Enables or disables depth writing.
            /// For transparent objects, it's usually recommended to disable this.
            pub fn set_depth_write(&self, depth_write: bool) {
                self.settings_mut().depth_write = depth_write;
            }
            pub fn depth_write(&self) -> bool {
                self.settings.read().depth_write
            }


            // --- Uniform Accessors ---

            /// Gets a mutable accessor for all Uniform parameters (for batch modifications).
            /// The Guard will automatically mark data as Dirty when it goes out of scope.
            pub fn uniforms_mut(&self) -> $crate::resources::buffer::BufferGuard<'_, $uniform_struct> {
                self.uniforms.write()
            }

            /// Gets a read-only accessor for all Uniform parameters.
            pub fn uniforms(&self) -> $crate::resources::buffer::BufferReadGuard<'_, $uniform_struct> {
                self.uniforms.read()
            }

            $(
                paste::paste! {
                    #[doc = $u_doc]
                    #[allow(clippy::float_cmp)]
                    pub fn [<set_ $u_field>](&self, value: $u_type) {
                        // Fast path: acquire read lock (Shared Lock), minimal overhead
                        if self.uniforms.read().$u_field == value {
                            return;
                        }

                        // 2. Slow path: only acquire write lock when modification is actually needed
                        let mut guard = self.uniforms.write();
                        // Double-check to prevent concurrent modification override
                        if guard.$u_field != value {
                            guard.$u_field = value;
                            // Guard will auto-increment version on Drop
                        } else {
                            // Value unchanged, skip sync
                            guard.skip_sync();
                        }
                    }
                }

                pub fn $u_field(&self) -> $u_type {
                    self.uniforms.read().$u_field
                }
            )*

            // === Texture Accessors ===
            $(
                paste::paste! {

                    #[doc = $t_doc]
                    pub fn [<set_ $t_field>](&self, value: Option<$crate::assets::TextureHandle>) {
                        // Acquire textures write lock
                        let mut tex_data = self.textures.write();
                        if tex_data.$t_field.texture != value {
                            tex_data.$t_field.texture = value;
                            // Texture changed, BindGroup needs rebuild, manually bump version
                            self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                    pub fn $t_field(&self) -> Option<$crate::assets::TextureHandle> {
                        // Acquire textures read lock
                        self.textures.read().$t_field.texture.clone()
                    }

                    // Allow modifying texture transform (UV Transform)
                    pub fn [<set_ $t_field _transform>](&self, transform: $crate::resources::material::TextureTransform) {
                         let mut tex_data = self.textures.write();
                         tex_data.$t_field.transform = transform;
                         // flush_texture_transforms() or handle automatically here
                    }

                    // e.g., modify texture, channel, transform simultaneously
                    pub fn [<configure_ $t_field>]<F>(&self, f: F)
                    where F: FnOnce(&mut $crate::resources::material::TextureSlot)
                    {
                        let mut tex_data = self.textures.write();
                        let slot = &mut tex_data.$t_field;

                        // Record pre-modification state (key fields) to determine if version bump is needed
                        // Assuming TextureSlot implements PartialEq, or we only care about texture and channel
                        let old_texture = slot.texture.clone();
                        let old_channel = slot.channel;

                        // Execute user's closure
                        f(slot);

                        // Check if version needs update
                        // 1. Texture changed -> BindGroup changes -> Version++
                        // 2. Channel changed -> Shader macro changes (HAS_MAP_UV 1) -> Version++
                        if slot.texture != old_texture || slot.channel != old_channel {
                            self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                }
            )*

            /// Flushes texture transform matrices to Uniforms.
            /// Only writes when values actually change, avoiding unnecessary version bumps.
            /// Returns whether any data was updated.
            pub fn flush_texture_transforms(&self) -> bool {
                let mut changed = false;

                let tex_data = self.textures.read();
                let mut uniforms = self.uniforms.write();

                $(
                    paste::paste! {
                        // Calculate matrix
                        let new_matrix = tex_data.$t_field.compute_matrix();

                        // Auto-derive field name: map -> map_transform
                        if uniforms.[<$t_field _transform>] != new_matrix {
                            uniforms.[<$t_field _transform>] = new_matrix;
                            changed = true;
                        }
                    }
                )*
                changed
            }

            // --- Batch Config ---
            pub fn configure<F>(&self, f: F)
            where
                F: FnOnce(&$uniform_struct)
            {
                let mut guard = self.uniforms.write();
                f(&mut *guard);
            }

            /// Manually notifies that the material pipeline needs to be rebuilt.
            ///
            /// **Note**: In most cases, you don't need to call this method manually, as
            /// modifying texture slots through the standard API will automatically track
            /// version changes. This method is only for the following special scenarios:
            ///
            /// - After directly modifying `pub(crate)` fields (e.g., in loader internal code)
            /// - When you're certain the material configuration has changed but the version hasn't been updated
            #[inline]
            pub fn notify_pipeline_dirty(&self) {
                self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        impl Clone for $struct_name {
            fn clone(&self) -> Self {
                use std::sync::atomic::Ordering;
                Self {
                    // 1. Uniforms: CpuBuffer
                    uniforms: self.uniforms.clone(),

                    // 2. Settings: read lock -> copy data -> new lock
                    settings: parking_lot::RwLock::new(self.settings.read().clone()),

                    // 3. Textures: read lock -> copy data -> new lock
                    textures: parking_lot::RwLock::new(self.textures.read().clone()),

                    // 4. Version: atomic read -> new atomic variable
                    version: std::sync::atomic::AtomicU64::new(
                        self.version.load(Ordering::Relaxed)
                    ),

                    auto_sync_texture_to_uniforms: self.auto_sync_texture_to_uniforms,

                    // Insert custom field clone logic
                    $(
                        $m_field: ($m_expr)(self),
                    )*
                }
            }
        }
    };

    //
    // Overload without manual_clone_fields
    (
        $struct_name:ident,
        $uniform_struct:ty,
        uniforms: [ $($u_args:tt)* ],
        textures: [ $($t_args:tt)* ]
    ) => {
        $crate::impl_material_api!(
            $struct_name,
            $uniform_struct,
            uniforms: [ $($u_args)* ],
            textures: [ $($t_args)* ],
            manual_clone_fields: {}
        );
    };
}

/// [Macro 2] Trait Implementer
/// Automatically implements `MaterialTrait` and `RenderableMaterialTrait`.
/// Handles all binding logic, shader macro generation, and other tedious work.
#[macro_export]
macro_rules! impl_material_trait {
    (
        $struct_name:ident,
        $shader_name:expr,
        $uniform_struct:ty,
        textures: [ $($field:ident),* $(,)? ]
    ) => {
        // 1. Implement common interface
        impl $crate::resources::material::MaterialTrait for $struct_name {
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }

        // 2. Implement rendering interface
        impl $crate::resources::material::RenderableMaterialTrait for $struct_name {
            fn shader_name(&self) -> &'static str { $shader_name }
            fn version(&self) -> u64 { self.version.load(std::sync::atomic::Ordering::Relaxed) }
            fn settings(&self) -> $crate::resources::material::MaterialSettings { *self.settings.read() }
            fn uniform_buffer(&self) -> $crate::resources::buffer::BufferRef { self.uniforms.handle() }

            fn with_uniform_bytes(&self, visitor: &mut dyn FnMut(&[u8])) {
                use $crate::resources::buffer::GpuData;

                // 1. Acquire read lock (Guard)
                let guard = self.uniforms.read();

                // 2. Pass internal data slice to callback function
                visitor(guard.as_bytes());

                // 3. Closure ends, Guard destroyed, lock released
            }

            fn shader_defines(&self) -> $crate::resources::shader_defines::ShaderDefines {
                let mut defines = $crate::resources::shader_defines::ShaderDefines::new();

                let tex_data = self.textures.read();

                $(
                    if tex_data.$field.texture.is_some() {
                        let field_upper = stringify!($field).to_uppercase();
                        let has_define_key = format!("HAS_{}", field_upper);
                        defines.set(&has_define_key, "1");
                        if tex_data.$field.channel > 0 {
                            let uv_define_key = format!("{}_UV", field_upper);
                            let uv_define_val = tex_data.$field.channel.to_string();
                            defines.set(&uv_define_key, &uv_define_val);
                        }
                    }
                )*

                // Acquire Settings read lock
                self.settings.read().generate_shader_defines(&mut defines);

                // Accumulate extra defines from material features
                self.extra_defines(&mut defines);
                defines
            }

            fn visit_textures(&self, visitor: &mut dyn FnMut(&$crate::resources::texture::TextureSource)) {
                // [Refactor] Acquire texture read lock
                let tex_data = self.textures.read();
                $(
                    if let Some(handle) = &tex_data.$field.texture {
                        visitor(&$crate::resources::texture::TextureSource::Asset(*handle));
                    }
                )*
            }

            fn define_bindings<'a>(&'a self, builder: &mut $crate::renderer::core::builder::ResourceBuilder<'a>) {
                // Uniform
                builder.add_uniform::<$uniform_struct>(
                    "material",
                    &self.uniforms,
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );

                let tex_data = self.textures.read();

                // Textures
                $(
                    if let Some(handle) = &tex_data.$field.texture {
                        let binding_name = stringify!($field);
                        let tex_source = $crate::resources::texture::TextureSource::Asset(*handle);

                        builder.add_texture(
                            binding_name,
                            Some(tex_source),
                            wgpu::TextureSampleType::Float { filterable: true },
                            wgpu::TextureViewDimension::D2,
                            wgpu::ShaderStages::FRAGMENT
                        );

                        paste::paste! {
                            let sampler_source = tex_data.[<$field _sampler>]
                                .or_else(|| Some($crate::resources::texture::SamplerSource::FromTexture(*handle)));

                            builder.add_sampler(
                                binding_name,
                                sampler_source,
                                wgpu::SamplerBindingType::Filtering,
                                wgpu::ShaderStages::FRAGMENT
                            );
                        }
                    }
                )*
            }
        }
    };
}
