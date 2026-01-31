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

            pub fn settings_mut(&self) -> $crate::resources::material::SettingsGuard<'_> {
                $crate::resources::material::SettingsGuard::new(
                    self.settings.write(), 
                    &self.version 
                )
            }

            pub fn settings(&self) -> $crate::resources::material::MaterialSettings {
                *self.settings.read()
            }


            pub fn set_alpha_mode(&mut self, mode: $crate::resources::material::AlphaMode) {
                self.settings_mut().alpha_mode = mode;
                // 处理与之关联的 Uniform 逻辑
                if let $crate::resources::material::AlphaMode::Mask(cutoff) = mode {
                     self.uniforms.write().alpha_test = cutoff;
                }
            }

            pub fn alpha_mode(&self) -> $crate::resources::material::AlphaMode {
                self.settings.read().alpha_mode
            }

            /// 设置渲染面剔除模式 (Front/Back/Double)。
            pub fn set_side(&mut self, side: $crate::resources::material::Side) {
                self.settings_mut().side = side;
            }
            pub fn side(&self) -> $crate::resources::material::Side {
                self.settings.read().side
            }

            /// 开启或关闭深度测试。
            pub fn set_depth_test(&mut self, depth_test: bool) {
                self.settings_mut().depth_test = depth_test;
            }
            pub fn depth_test(&self) -> bool {
                self.settings.read().depth_test
            }

            /// 开启或关闭深度写入。
            /// 对于透明物体，通常建议关闭此选项。
            pub fn set_depth_write(&mut self, depth_write: bool) {
                self.settings_mut().depth_write = depth_write;
            }
            pub fn depth_write(&self) -> bool {
                self.settings.read().depth_write
            }


            // --- Uniform Accessors ---

            /// 获取所有 Uniform 参数的可变访问器 (批量修改用)
            /// 返回的 Guard 离开作用域时会自动标记数据为 Dirty
            pub fn uniforms_mut(&self) -> $crate::resources::buffer::BufferGuard<'_, $uniform_struct> {
                self.uniforms.write()
            }

            /// 获取所有 Uniform 参数的只读访问器
            pub fn uniforms(&self) -> $crate::resources::buffer::BufferReadGuard<'_, $uniform_struct> {
                self.uniforms.read()
            }

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

            // === Texture Accessors ===
            $(
                paste::paste! {

                    #[doc = $t_doc]
                    pub fn [<set_ $t_field>](&self, value: Option<$crate::assets::TextureHandle>) {
                        // 获取 textures 的写锁
                        let mut tex_data = self.textures.write();
                        if tex_data.$t_field.texture != value {
                            tex_data.$t_field.texture = value;
                            // 纹理改变，BindGroup 需要重建，手动 bump version
                            self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                    pub fn $t_field(&self) -> Option<$crate::assets::TextureHandle> {
                        // 获取 textures 的读锁
                        self.textures.read().$t_field.texture.clone()
                    }

                    // 允许修改纹理变换 (UV Transform)
                    pub fn [<set_ $t_field _transform>](&self, transform: $crate::resources::material::TextureTransform) {
                         let mut tex_data = self.textures.write();
                         tex_data.$t_field.transform = transform;
                         //  flush_texture_transforms() 或者在这里自动处理
                    }

                    // 比如同时修改 texture, channel, transform
                    pub fn [<configure_ $t_field>]<F>(&self, f: F) 
                    where F: FnOnce(&mut $crate::resources::material::TextureSlot) 
                    {
                        let mut tex_data = self.textures.write();
                        let slot = &mut tex_data.$t_field;
                        
                        // 记录修改前的状态 (关键字段) 用于判断是否需要 bump version
                        // 假设 TextureSlot 实现了 PartialEq，或者我们只关心 texture 和 channel
                        let old_texture = slot.texture.clone();
                        let old_channel = slot.channel;

                        // 执行用户的闭包
                        f(slot);

                        // 检查是否需要更新版本号
                        // 1. 纹理变了 -> BindGroup 变 -> Version++
                        // 2. Channel 变了 -> Shader 宏变 (HAS_MAP_UV 1) -> Version++
                        if slot.texture != old_texture || slot.channel != old_channel {
                            self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                }
            )*

            /// 刷新纹理变换矩阵到 Uniform
            /// 仅在数值实际改变时写入，避免触发不必要的 Version Bump
            /// 返回值表示是否有数据更新
            pub fn flush_texture_transforms(&mut self) -> bool {
                let mut changed = false;

                let tex_data = self.textures.read();
                let mut uniforms = self.uniforms.write();

                $(
                    paste::paste! {
                        // 计算矩阵
                        let new_matrix = tex_data.$t_field.compute_matrix();
                        
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
                self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            fn version(&self) -> u64 { self.version.load(std::sync::atomic::Ordering::Relaxed) }
            fn settings(&self) -> $crate::resources::material::MaterialSettings { *self.settings.read() }
            fn uniform_buffer(&self) -> $crate::resources::buffer::BufferRef { self.uniforms.handle() }
            // fn uniform_bytes(&self) -> &[u8] { self.uniforms.as_bytes() }

            fn with_uniform_bytes(&self, visitor: &mut dyn FnMut(&[u8])) {
                use $crate::resources::buffer::GpuData;
                
                // 1. 获取读锁 (Guard)
                let guard = self.uniforms.read();
                
                // 2. 将内部数据的切片传给回调函数
                visitor(guard.as_bytes());
                
                // 3. 闭包结束，Guard 销毁，锁释放
            }

            fn shader_defines(&self) -> $crate::resources::shader_defines::ShaderDefines {
                let mut defines = $crate::resources::shader_defines::ShaderDefines::new();
                // 默认宏定义
                $(
                    defines.set($def_key, $def_val);
                )*

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
                
                // Settings 读锁
                self.settings.read().generate_shader_defines(&mut defines);
                defines
            }

            fn visit_textures(&self, visitor: &mut dyn FnMut(&$crate::resources::texture::TextureSource)) {
                // [重构] 获取纹理读锁
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