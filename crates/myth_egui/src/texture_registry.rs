use myth_assets::TextureHandle;
use myth_render::core::ResourceManager;
use rustc_hash::FxHashMap;

use crate::{Renderer, egui};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExternalTextureKey {
    view_id: u64,
    sampler_id: usize,
}

#[derive(Clone, Copy, Debug)]
struct RegisteredTexture {
    id: egui::TextureId,
    key: Option<ExternalTextureKey>,
    warned_unsupported_view: bool,
}

#[derive(Default)]
pub(crate) struct TextureRegistry {
    textures: FxHashMap<TextureHandle, RegisteredTexture>,
}

impl TextureRegistry {
    pub(crate) fn texture_id(
        &mut self,
        handle: TextureHandle,
        renderer: &mut Renderer,
    ) -> egui::TextureId {
        if let Some(texture) = self.textures.get(&handle) {
            return texture.id;
        }

        let id = renderer.reserve_external_texture();
        self.textures.insert(
            handle,
            RegisteredTexture {
                id,
                key: None,
                warned_unsupported_view: false,
            },
        );
        id
    }

    pub(crate) fn free(&mut self, handle: TextureHandle, renderer: &mut Renderer) {
        if let Some(texture) = self.textures.remove(&handle) {
            renderer.free_texture(&texture.id);
        }
    }

    pub(crate) fn free_by_id(&mut self, id: egui::TextureId) {
        self.textures.retain(|_, texture| texture.id != id);
    }

    pub(crate) fn resolve(
        &mut self,
        device: &wgpu::Device,
        resource_manager: &ResourceManager,
        renderer: &mut Renderer,
    ) {
        for (handle, texture) in &mut self.textures {
            let Some(binding) = resource_manager.get_texture_binding(*handle) else {
                renderer.use_placeholder_texture(texture.id);
                texture.key = None;
                continue;
            };

            let Some(gpu_image) = resource_manager.get_image(binding.image_handle) else {
                renderer.use_placeholder_texture(texture.id);
                texture.key = None;
                continue;
            };

            if gpu_image.default_view_dimension != wgpu::TextureViewDimension::D2 {
                renderer.use_placeholder_texture(texture.id);
                texture.key = None;
                if !texture.warned_unsupported_view {
                    log::warn!(
                        "myth_egui only supports 2D texture views, got {:?}",
                        gpu_image.default_view_dimension
                    );
                    texture.warned_unsupported_view = true;
                }
                continue;
            }

            let Some(sampler) = resource_manager.get_sampler_by_index(binding.sampler_id) else {
                renderer.use_placeholder_texture(texture.id);
                texture.key = None;
                log::warn!("Missing Myth sampler {}", binding.sampler_id);
                continue;
            };

            let key = ExternalTextureKey {
                view_id: gpu_image.id,
                sampler_id: binding.sampler_id,
            };

            if texture.key != Some(key) {
                renderer.update_external_texture(
                    device,
                    texture.id,
                    &gpu_image.default_view,
                    sampler,
                );
                texture.key = Some(key);
            }
        }
    }
}
