use std::collections::HashMap;
use wgpu::ShaderStages;
use crate::core::material::{Material, MaterialValue};
use super::uniforms::{MeshBasicUniforms, MeshStandardUniforms};

#[derive(Debug, Clone)]
pub struct MaterialShaderInfo {
    pub source_template: String, // e.g. "pbr.wgsl"
    pub shader_hash: String,     // e.g. "MeshStandard_USE_MAP"
    pub layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub wgsl_code: String,       // Struct definitions & Binding declarations
    pub defines: Vec<String>,    // Macros
    pub texture_bindings: HashMap<String, u32>,
    pub sampler_bindings: HashMap<String, u32>,
}

pub fn generate_material_layout(material: &Material) -> MaterialShaderInfo {
    match material.type_name.as_str() {
        "MeshStandard" => generate_standard(material),
        _ => generate_basic(material),
    }
}

// === Generator Logic ===

fn generate_standard(material: &Material) -> MaterialShaderInfo {
    let mut ctx = GeneratorContext::new("pbr.wgsl", "MeshStandard");
    
    // 1. Uniforms (Auto-generated code from Rust struct)
    let uniform_code = MeshStandardUniforms::wgsl_struct_code("MaterialUniforms");
    ctx.add_uniform_binding(&uniform_code);

    // 2. Textures
    // 辅助闭包：检查属性并添加
    let mut check_add = |key: &str, define: &str| {
        if let Some(MaterialValue::Texture(tex_ref)) = material.properties.get(key) {
            let texture = tex_ref.read().unwrap();
            // 将 Texture 传给 add_texture
            ctx.add_texture(key, define, &texture); 
        }
    };

    check_add("map", "USE_MAP");
    check_add("normalMap", "USE_NORMAL_MAP");
    check_add("roughnessMap", "USE_ROUGHNESS_MAP");
    check_add("metalnessMap", "USE_METALNESS_MAP");
    check_add("emissiveMap", "USE_EMISSIVE_MAP");
    check_add("occlusionMap", "USE_AOMAP");
    
    // 如果有 envMap (环境贴图)，这会自动识别为 Cube Texture！
    check_add("envMap", "USE_ENV_MAP"); 

    ctx.build()
}

fn generate_basic(material: &Material) -> MaterialShaderInfo {
    let mut ctx = GeneratorContext::new("basic.wgsl", material.type_name.as_str());
    
    let uniform_code = MeshBasicUniforms::wgsl_struct_code("MaterialUniforms");
    ctx.add_uniform_binding(&uniform_code);

    // 辅助闭包：检查属性并添加
    let mut check_add = |key: &str, define: &str| {
        if let Some(MaterialValue::Texture(tex_ref)) = material.properties.get(key) {
            let texture = tex_ref.read().unwrap();
            // 将 Texture 传给 add_texture
            ctx.add_texture(key, define, &texture); 
        }
    };

    check_add("map", "USE_MAP");

    ctx.build()
}

// 根据 view_dimension 获取 WGSL 类型
fn get_wgsl_texture_type(dim: wgpu::TextureViewDimension) -> &'static str {
    match dim {
        wgpu::TextureViewDimension::D1 => "texture_1d<f32>",
        wgpu::TextureViewDimension::D2 => "texture_2d<f32>",
        wgpu::TextureViewDimension::D2Array => "texture_2d_array<f32>",
        wgpu::TextureViewDimension::Cube => "texture_cube<f32>",
        wgpu::TextureViewDimension::CubeArray => "texture_cube_array<f32>",
        wgpu::TextureViewDimension::D3 => "texture_3d<f32>",
    }
}

// === Helper Context ===

struct GeneratorContext {
    template: String,
    hash_base: String,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    wgsl_lines: Vec<String>,
    defines: Vec<String>,
    tex_bindings: HashMap<String, u32>,
    samp_bindings: HashMap<String, u32>,
    current_binding: u32,
}

impl GeneratorContext {
    fn new(template: &str, hash_base: &str) -> Self {
        Self {
            template: template.into(),
            hash_base: hash_base.into(),
            entries: Vec::new(),
            wgsl_lines: Vec::new(),
            defines: Vec::new(),
            tex_bindings: HashMap::new(),
            samp_bindings: HashMap::new(),
            current_binding: 0,
        }
    }

    fn add_uniform_binding(&mut self, struct_code: &str) {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        self.wgsl_lines.push(struct_code.to_string());
        self.wgsl_lines.push("@group(1) @binding(0) var<uniform> material: MaterialUniforms;".to_string());
        self.current_binding = 1;
    }

    fn add_texture(&mut self, name: &str, define: &str, texture: &crate::core::texture::Texture) {
        let tex_bind = self.current_binding;
        let samp_bind = self.current_binding + 1;
        self.current_binding += 2;

        self.defines.push(define.into());
        
        // Texture Entry
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: tex_bind,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: texture.view_dimension,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        });

        // Sampler Entry
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: samp_bind,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        // WGSL
        let wgsl_type = get_wgsl_texture_type(texture.view_dimension);
        self.wgsl_lines.push(format!("@group(1) @binding({}) var t_{}: {};", tex_bind, name, wgsl_type));
        self.wgsl_lines.push(format!("@group(1) @binding({}) var s_{}: sampler;", samp_bind, name));
        // Maps
        self.tex_bindings.insert(name.into(), tex_bind);
        self.samp_bindings.insert(name.into(), samp_bind);
    }

    fn build(self) -> MaterialShaderInfo {
        let mut defines = self.defines;
        defines.sort();
        let hash = format!("{}_{}", self.hash_base, defines.join("|"));
        
        MaterialShaderInfo {
            source_template: self.template,
            shader_hash: hash,
            layout_entries: self.entries,
            wgsl_code: self.wgsl_lines.join("\n"),
            defines,
            texture_bindings: self.tex_bindings,
            sampler_bindings: self.samp_bindings,
        }
    }
}