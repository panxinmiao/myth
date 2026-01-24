// src/resources/material/macros.rs

/// [宏 1] API 生成器
/// 自动生成 Setters, Getters 和 Configure 方法。
/// 隐藏了底层的 uniforms 和 bindings 字段，只暴露干净的 Public API。
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
            pub fn set_transparent(&mut self, transparent: bool) {
                if self.settings.transparent != transparent {
                    self.settings.transparent = transparent;
                    self.version += 1; // 标记脏状态，触发 Pipeline 重建
                }
            }
            pub fn transparent(&self) -> bool {
                self.settings.transparent
            }

            /// 设置渲染面剔除模式 (Front/Back/Double)。
            pub fn set_side(&mut self, side: $crate::resources::material::Side) {
                if self.settings.side != side {
                    self.settings.side = side;
                    self.version += 1;
                }
            }
            pub fn side(&self) -> $crate::resources::material::Side {
                self.settings.side
            }

            /// 开启或关闭深度测试。
            pub fn set_depth_test(&mut self, depth_test: bool) {
                if self.settings.depth_test != depth_test {
                    self.settings.depth_test = depth_test;
                    self.version += 1;
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
                    self.version += 1;
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

            // --- Texture Accessors ---
            $(
                paste::paste! {
                    #[doc = $t_doc]
                    pub fn [<set_ $t_field>](&mut self, texture: impl Into<Option<$crate::resources::texture::TextureSource>>) {
                        self.bindings.$t_field = texture.into();
                    }

                    pub fn [<set_ $t_field _sampler>](&mut self, sampler: impl Into<Option<$crate::resources::texture::SamplerSource>>) {
                        self.bindings.[<$t_field _sampler>] = sampler.into();
                    }
                }

                pub fn $t_field(&self) -> Option<&$crate::resources::texture::TextureSource> {
                    self.bindings.$t_field.as_ref()
                }
            )*

            // --- 批量配置 (Batch Config) ---
            pub fn configure<F>(&mut self, f: F)
            where
                F: FnOnce(&mut $uniform_struct)
            {
                let mut guard = self.uniforms.write();
                f(&mut *guard);
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
        // Textures: (字段名, 宏名称)
        textures: [ $(($field:ident, $macro_name:expr)),* $(,)? ]
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
            fn bindings(&self) -> &$crate::resources::material::MaterialBindings { &self.bindings }
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
                    if self.bindings.$field.is_some() {
                        defines.set($macro_name, "1");
                    }
                )*
                defines
            }

            fn visit_textures(&self, visitor: &mut dyn FnMut(&$crate::resources::texture::TextureSource)) {
                $(
                    if let Some(tex) = &self.bindings.$field {
                        visitor(tex);
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
                    if let Some(tex) = &self.bindings.$field {
                        let binding_name = stringify!($field);

                        builder.add_texture(
                            binding_name,
                            Some(*tex),
                            wgpu::TextureSampleType::Float { filterable: true },
                            wgpu::TextureViewDimension::D2,
                            wgpu::ShaderStages::FRAGMENT
                        );

                        paste::paste! {
                            let sampler_source = self.bindings.[<$field _sampler>]
                                .or_else(|| match tex {
                                    $crate::resources::texture::TextureSource::Asset(handle) => 
                                        Some($crate::resources::texture::SamplerSource::FromTexture(*handle)),
                                    _ => None,
                                });
                            
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