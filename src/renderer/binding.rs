use uuid::Uuid;
use wgpu::{ShaderStages};
use crate::core::buffer::BufferRef; 
use crate::core::material::{MeshBasicMaterial, MeshStandardMaterial, Material};
use crate::core::geometry::Geometry;
use crate::core::world::WorldEnvironment;
use crate::core::uniforms::*;
use crate::renderer::resource_builder::ResourceBuilder;

/// 实际的绑定资源数据 (用于生成 BindGroup)
/// 层只持有 ID 或 数据引用，不持有 GPU 句柄
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingResource {

    /// 持有 CPU Buffer 的引用 (统一了 Vertex/Index/Uniform/Storage)
    Buffer {
        buffer: BufferRef,
        offset: u64,        // 偏移量 (默认为 0)
        size: Option<u64>,  // 绑定窗口大小 (None 表示整个 Buffer)
    },
    
    /// 外部 Buffer ID (用于高级场景，暂保留)
    // BufferId(Uuid),

    /// 纹理 ID (可能为空，意味着需要使用缺省纹理)
    Texture(Option<Uuid>),
    
    /// 采样器 ID (通常跟随纹理，但也可以独立)
    Sampler(Option<Uuid>),

}

pub fn define_texture_binding(builder: &mut ResourceBuilder, name: &'static str, texture_id: Option<Uuid>) {
    builder.add_texture(
        name,
        texture_id,
        wgpu::TextureSampleType::Float { filterable: true },
        wgpu::TextureViewDimension::D2,
        ShaderStages::FRAGMENT,
    );
    builder.add_sampler(
        name,
        texture_id,
        wgpu::SamplerBindingType::Filtering,
        ShaderStages::FRAGMENT,
    );
}

pub trait Bindings {
    fn define_bindings(&self, builder: &mut ResourceBuilder);
}

impl Bindings for MeshBasicMaterial {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // 1. Uniform Buffer
        builder.add_uniform::<MeshBasicUniforms>(
            "material", 
            &self.uniform_buffer, 
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );

        // 2. Texture + Sampler
        if let Some(id) = self.map {
            define_texture_binding(builder, "map", Some(id));
        }
    }
}

impl Bindings for MeshStandardMaterial {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // 1. Uniform Buffer
        builder.add_uniform::<MeshStandardUniforms>(
            "material", 
            &self.uniform_buffer, 
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );

        // 2. Maps
        // 辅助闭包：减少重复代码
        let mut add_tex = |name: &str, id: Option<Uuid>| {
            if let Some(tex_id) = id {
                builder.add_texture(name, Some(tex_id), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, ShaderStages::FRAGMENT);
                builder.add_sampler(name, Some(tex_id), wgpu::SamplerBindingType::Filtering, ShaderStages::FRAGMENT);
            }
        };

        add_tex("map", self.map);
        add_tex("normal_map", self.normal_map);
        add_tex("roughness_map", self.roughness_map);
        add_tex("metalness_map", self.metalness_map);
        add_tex("emissive_map", self.emissive_map);
        add_tex("ao_map", self.ao_map);
    }
}

impl Bindings for Material {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        match self.data.as_any().downcast_ref::<MeshBasicMaterial>() {
            Some(basic) => basic.define_bindings(builder),
            None => {},
        }

        match self.data.as_any().downcast_ref::<MeshStandardMaterial>() {
            Some(standard) => standard.define_bindings(builder),
            None => {},
        }
    }
}

impl Bindings for Geometry {
    fn define_bindings(&self, _builder: &mut ResourceBuilder) {
        // 需要根据 morph_attributes 来自动生成 morph texture 资源（或 storage buffer）
        // 然后绑定到 Group 2 中
       
    }
}

impl Bindings for WorldEnvironment {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // Binding 0: Frame Uniforms (Vertex + Fragment)
        builder.add_uniform::<GlobalFrameUniforms>(
            "global", 
            &self.frame_uniforms, // 传入 BufferRef
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );

        // Binding 1: Light Uniforms (Fragment only)
        builder.add_uniform::<GlobalLightUniforms>(
            "lights",
            &self.light_uniforms,
            wgpu::ShaderStages::FRAGMENT
        );
    }

    // 未来可以在这里添加：
    // builder.add_texture("env_map", self.env_map_id, ...)
}

