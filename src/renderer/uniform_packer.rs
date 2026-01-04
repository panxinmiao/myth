use glam::{Vec2, Vec3, Vec4};
use crate::core::material::{Material, MaterialValue};
use super::uniforms::{MeshBasicUniforms, MeshStandardUniforms, Mat3A};

pub fn pack_material_uniforms(material: &Material) -> Vec<u8> {
    match material.type_name.as_str() {
        "MeshStandard" => {
            let mut u = MeshStandardUniforms::default();
            
            // Helper closure to extract values
            if let Some(v) = get_vec4(material, "color") { u.color = v; }
            if let Some(v) = get_vec3(material, "emissive") { u.emissive = v; }
            if let Some(v) = get_float(material, "roughness") { u.roughness = v; }
            if let Some(v) = get_float(material, "metalness") { u.metalness = v; }
            if let Some(v) = get_vec2(material, "normalScale") { u.normal_scale = v; }
            if let Some(v) = get_float(material, "occlusionStrength") { u.occlusion_strength = v; }

            // 获取 UV 变换 (通常从 material.properties 里取，由 Scene 系统更新)
            u.map_transform = get_tex_matrix(material, "map");
            u.normal_map_transform = get_tex_matrix(material, "normalMap");
            u.roughness_map_transform = get_tex_matrix(material, "roughnessMap");
            u.metalness_map_transform = get_tex_matrix(material, "metalnessMap");
            u.emissive_map_transform = get_tex_matrix(material, "emissiveMap");
            u.occlusion_map_transform = get_tex_matrix(material, "occlusionMap");
            
            bytemuck::bytes_of(&u).to_vec()
        }
        "MeshBasic" | "LineBasic" => {
            let mut u = MeshBasicUniforms::default();
            if let Some(v) = get_vec4(material, "color") { u.color = v; }
            // opacity 通常来自 color.a 或者单独的属性，这里假设单独存
            if let Some(v) = get_float(material, "opacity") { u.opacity = v; } else { u.opacity = u.color.w; }

            u.map_transform = get_tex_matrix(material, "map");
            
            bytemuck::bytes_of(&u).to_vec()
        }
        _ => Vec::new(), // 未知材质，返回空，可能会导致 crash，生产环境应返回默认 Basic
    }
}

// === Helpers ===

fn get_float(mat: &Material, key: &str) -> Option<f32> {
    match mat.properties.get(key) {
        Some(MaterialValue::Float(v)) => Some(*v),
        _ => None,
    }
}

fn get_vec2(mat: &Material, key: &str) -> Option<Vec2> {
    match mat.properties.get(key) {
        Some(MaterialValue::Vec2(v)) => Some(*v),
        _ => None,
    }
}

fn get_vec3(mat: &Material, key: &str) -> Option<Vec3> {
    match mat.properties.get(key) {
        Some(MaterialValue::Vec3(v)) => Some(*v),
        _ => None,
    }
}

fn get_vec4(mat: &Material, key: &str) -> Option<Vec4> {
    match mat.properties.get(key) {
        Some(MaterialValue::Vec4(v)) => Some(*v),
        _ => None,
    }
}

// 修改 helper，返回 Mat3A
fn get_tex_matrix(mat: &Material, key: &str) -> Mat3A {
    if let Some(MaterialValue::Texture(tex_ref)) = mat.properties.get(key) {
        let tex = tex_ref.read().unwrap();
        // Texture::transform.get_matrix() 返回 glam::Mat3
        // 调用我们写的转换函数
        Mat3A::from_mat3(tex.transform.get_matrix())
    } else {
        Mat3A::IDENTITY
    }
}