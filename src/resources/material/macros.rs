// src/resources/material/macros.rs

/// [宏 1] API 生成器
/// 自动生成 Setters, Getters 和 Configure 方法。
/// 隐藏了底层的 uniforms 和 bindings 字段，只暴露干净的 Public API。
/// 
/// 纹理槽位采用 Guard 模式，自动追踪纹理有/无状态的变化以触发版本更新。
#[macro_export]
macro_rules! impl_material_api {
    (
        $struct_name:ident, 
        $uniform_struct:ty,
        // Uniforms: (字段名, 类型, 文档)
        uniforms: [ $(($u_field:ident, $u_type:ty, $u_doc:expr)),* $(,)? ],
        // Textures: (字段名, 文档)
        textures: [ $(($t_field:ident, $t_doc:expr)),* $(,)? ]
    ) => {
        impl $struct_name {

            // ==========================================
            // 1. 通用 Settings API
            // ==========================================
            
            /// 设置是否开启透明混合。
            /// 注意：切换此选项可能会触发布局重建。
            // pub fn set_transparent(&mut self, transparent: bool) {
            //     let transparent_mode = if transparent {
            //         $crate::resources::material::AlphaMode::Blend
            //     } else {
            //         $crate::resources::material::AlphaMode::Opaque
            //     };

            //     if self.settings.alpha_mode != transparent_mode {
            //         self.settings.alpha_mode = transparent_mode;
            //         self.version = self.version.wrapping_add(1);
            //     }
            // }
            // pub fn transparent(&self) -> bool {
            //     match self.settings.alpha_mode {
            //         $crate::resources::material::AlphaMode::Blend => true,
            //         _ => false,
            //     }
            // }

            pub fn set_alpha_mode(&mut self, mode: $crate::resources::material::AlphaMode) {
                if self.settings.alpha_mode != mode {
                    match mode {
                        $crate::resources::material::AlphaMode::Mask(cutoff) => {
                            let mut uniforms = self.uniforms.write();
                            uniforms.alpha_test = cutoff;
                        }
                        _ => {}
                    }
                    self.settings.alpha_mode = mode;
                    self.version = self.version.wrapping_add(1);
                }
            }

            pub fn alpha_mode(&self) -> $crate::resources::material::AlphaMode {
                self.settings.alpha_mode
            }

            /// 设置渲染面剔除模式 (Front/Back/Double)。
            pub fn set_side(&mut self, side: $crate::resources::material::Side) {
                if self.settings.side != side {
                    self.settings.side = side;
                    self.version = self.version.wrapping_add(1);
                }
            }
            pub fn side(&self) -> $crate::resources::material::Side {
                self.settings.side
            }

            /// 开启或关闭深度测试。
            pub fn set_depth_test(&mut self, depth_test: bool) {
                if self.settings.depth_test != depth_test {
                    self.settings.depth_test = depth_test;
                    self.version = self.version.wrapping_add(1);
                }
            }
            pub fn depth_test(&self) -> bool {
                self.settings.depth_test
            }

            /// 开启或关闭深度写入。
            /// 对于透明物体，通常建议关闭此选项。
            pub fn set_depth_write(&mut self, depth_write: bool) {
                if self.settings.depth_write != depth_write {
                    self.settings.depth_write = depth_write;
                    self.version = self.version.wrapping_add(1);
                }
            }
            pub fn depth_write(&self) -> bool {
                self.settings.depth_write
            }


            // --- Uniform Accessors ---
            $(
                paste::paste! {
                    #[doc = $u_doc]
                    pub fn [<set_ $u_field>](&mut self, value: $u_type) {
                        self.uniforms.write().$u_field = value;
                    }
                }

                pub fn $u_field(&self) -> $u_type {
                    self.uniforms.read().$u_field
                }
            )*

            // ==========================================
            // 2. 纹理槽位 API (Guard 模式)
            // ==========================================
            $(
                paste::paste! {
                    #[doc = $t_doc]
                    #[doc = "\n\n获取纹理槽位的只读引用。"]
                    #[inline]
                    pub fn $t_field(&self) -> &$crate::resources::material::TextureSlot {
                        &self.$t_field
                    }

                    #[doc = $t_doc]
                    #[doc = "\n\n获取纹理槽位的可变引用守卫。"]
                    #[doc = "当纹理的有/无状态变化时，会自动触发版本更新。"]
                    #[inline]
                    pub fn [<$t_field _mut>](&mut self) -> $crate::resources::material::TextureSlotGuard<'_> {
                        $crate::resources::material::TextureSlotGuard::new(
                            &mut self.$t_field,
                            &mut self.version
                        )
                    }

                    #[doc = $t_doc]
                    #[doc = "\n\n快捷方法：直接设置纹理句柄。"]
                    #[doc = "会自动处理版本更新。"]
                    #[inline]
                    pub fn [<set_ $t_field>](&mut self, texture: Option<impl Into<$crate::resources::material::TextureSlot>>) {
                        let new_slot = texture.map(|t| t.into()).unwrap_or_default();
                        let was_some = self.$t_field.texture.is_some();
                        let is_some = new_slot.texture.is_some();
                        self.$t_field = new_slot;
                        if was_some != is_some {
                            self.version = self.version.wrapping_add(1);
                        }
                    }
                }
            )*

            /// 刷新纹理变换矩阵到 Uniform
            /// 仅在数值实际改变时写入，避免触发不必要的 Version Bump
            /// 返回值表示是否有数据更新
            pub fn flush_texture_transforms(&mut self) -> bool {
                let mut changed = false;
                let mut uniforms = self.uniforms.write();

                $(
                    paste::paste! {
                        // 计算矩阵
                        let new_matrix = self.$t_field.compute_matrix();
                        
                        // 自动推导字段名: map -> map_transform
                        if uniforms.[<$t_field _transform>] != new_matrix {
                            uniforms.[<$t_field _transform>] = new_matrix;
                            changed = true;
                        }
                    }
                )*
                changed
            }

            // --- 批量配置 (Batch Config) ---
            pub fn configure<F>(&mut self, f: F)
            where
                F: FnOnce(&mut $uniform_struct)
            {
                let mut guard = self.uniforms.write();
                f(&mut *guard);
            }

            /// 手动通知材质管线需要重建。
            /// 
            /// **注意**: 大多数情况下不需要手动调用此方法，因为通过标准 API 修改纹理槽位
            /// 时会自动追踪版本变化。此方法仅用于以下特殊场景：
            /// 
            /// - 直接修改 `pub(crate)` 字段后（如加载器内部代码）
            /// - 确信材质配置已改变但版本未更新时
            #[inline]
            pub fn notify_pipeline_dirty(&mut self) {
                self.version = self.version.wrapping_add(1);
            }
        }
    };
}

/// [宏 2] Trait 实现器
/// 自动实现 MaterialTrait 和 RenderableMaterialTrait。
/// 负责处理所有的绑定逻辑、Shader 宏生成等繁琐工作。
#[macro_export]
macro_rules! impl_material_trait {
    (
        $struct_name:ident,
        $shader_name:expr,
        $uniform_struct:ty,
        default_defines: [ $(($def_key:expr, $def_val:expr)),* $(,)? ],
        // Textures: (字段名)
        textures: [ $($field:ident),* $(,)? ]
    ) => {
        // 1. 实现通用接口
        impl $crate::resources::material::MaterialTrait for $struct_name {
            fn as_any(&self) -> &dyn std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }

        // 2. 实现渲染接口
        impl $crate::resources::material::RenderableMaterialTrait for $struct_name {
            fn shader_name(&self) -> &'static str { $shader_name }
            fn version(&self) -> u64 { self.version }
            fn settings(&self) -> &$crate::resources::material::MaterialSettings { &self.settings }
            fn uniform_buffer(&self) -> &$crate::resources::buffer::BufferRef { self.uniforms.handle() }
            fn uniform_bytes(&self) -> &[u8] { self.uniforms.as_bytes() }

            fn shader_defines(&self) -> $crate::resources::shader_defines::ShaderDefines {
                let mut defines = $crate::resources::shader_defines::ShaderDefines::new();
                // 默认宏定义
                $(
                    defines.set($def_key, $def_val);
                )*

                // 纹理宏定义
                $(
                    if self.$field.is_some() {
                        // 2.1 自动生成开关宏：map -> HAS_MAP
                        // stringify!(map) -> "map" -> to_uppercase() -> "MAP"
                        let field_upper = stringify!($field).to_uppercase();
                        let has_define_key = format!("HAS_{}", field_upper);
                        defines.set(&has_define_key, "1");

                        // 2.2 自动生成 UV 通道宏：map -> MAP_UV
                        // 值为 self.map.channel (例如 "0", "1")
                        let uv_define_key = format!("{}_UV", field_upper);
                        let uv_define_val = self.$field.channel.to_string();
                        defines.set(&uv_define_key, &uv_define_val);
                    }

                )*
                // settings 相关宏定义
                self.settings.generate_shader_defines(&mut defines);

                defines
            }

            fn visit_textures(&self, visitor: &mut dyn FnMut(&$crate::resources::texture::TextureSource)) {
                $(
                    if let Some(handle) = &self.$field.texture {
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

                // Textures
                $(
                    if let Some(handle) = &self.$field.texture {
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
                            let sampler_source = self.[<$field _sampler>]
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