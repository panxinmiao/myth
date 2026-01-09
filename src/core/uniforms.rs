use glam::{Vec2, Vec3, Vec4, Mat3, Mat4};
use bytemuck::{Pod, Zeroable};

/// 专门用于 Uniform Buffer 的 3x3 矩阵
/// 对应 WGSL 中的 `mat3x3<f32>` (std140 布局)
/// 内存布局: [Col0(4 floats), Col1(4 floats), Col2(4 floats)]
/// 第 4 个 float 是 padding，GPU 会忽略它
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Mat3A {
    pub cols: [Vec4; 3], 
}

impl Mat3A {
    pub const IDENTITY: Self = Self {
        cols: [
            Vec4::new(1.0, 0.0, 0.0, 0.0), // Col 0
            Vec4::new(0.0, 1.0, 0.0, 0.0), // Col 1
            Vec4::new(0.0, 0.0, 1.0, 0.0), // Col 2
        ],
    };

    /// 从 glam::Mat3 转换 (自动处理 Padding)
    pub fn from_mat3(m: Mat3) -> Self {
        Self {
            cols: [
                Vec4::new(m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0),
                Vec4::new(m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0),
                Vec4::new(m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0),
            ],
        }
    }

    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            cols: [
                Vec4::new(m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0),
                Vec4::new(m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0),
                Vec4::new(m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0),
            ],
        }
    }
}

// ============================================================================
// 1. 类型映射 Trait (Rust Type -> WGSL Type String)
// ============================================================================

pub trait WgslType {
    fn wgsl_type_name() -> &'static str;
}

impl WgslType for f32 { fn wgsl_type_name() -> &'static str { "f32" } }
impl WgslType for i32 { fn wgsl_type_name() -> &'static str { "i32" } }
impl WgslType for u32 { fn wgsl_type_name() -> &'static str { "u32" } }
impl WgslType for Vec2 { fn wgsl_type_name() -> &'static str { "vec2<f32>" } }
impl WgslType for Vec3 { fn wgsl_type_name() -> &'static str { "vec3<f32>" } }
impl WgslType for Vec4 { fn wgsl_type_name() -> &'static str { "vec4<f32>" } }
impl WgslType for Mat4 { fn wgsl_type_name() -> &'static str { "mat4x4<f32>" } }
impl WgslType for Mat3A { fn wgsl_type_name() -> &'static str { "mat3x3<f32>" } }

// 专用 Padding 类型 (在 WGSL 中通常用 private 变量或手动对齐，这里简化映射为 f32/vec3)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Padf32(pub f32);
impl WgslType for Padf32 { fn wgsl_type_name() -> &'static str { "f32" } }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PadVec3(pub Vec3);
impl WgslType for PadVec3 { fn wgsl_type_name() -> &'static str { "vec3<f32>" } }


// #[repr(C)]
// #[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
// pub struct GlobalUniforms {
//     pub view_projection: Mat4,
//     pub view_projection_inverse: Mat4,
//     pub view_matrix: Mat4,
// }


#[repr(C, align(256))] // 强制每个实例占用 256 字节
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DynamicModelUniforms {
    pub model_matrix: Mat4,       //64
    pub model_matrix_inverse: Mat4,  //64
    pub normal_matrix: Mat3A,   //48

    pub _padding: [f32; 20], // 填充至 256 bytes
}

impl DynamicModelUniforms {
    pub fn wgsl_struct_def(struct_name: &str) -> String {
        let mut code = format!("struct {} {{\n", struct_name);
        code.push_str("    model_matrix: mat4x4<f32>,\n");
        code.push_str("    model_matrix_inverse: mat4x4<f32>,\n");
        code.push_str("    normal_matrix: mat3x3<f32>,\n");
        code.push_str("};");
        code
    }
}


// ============================================================================
// 2. 宏定义 (Single Source of Truth)
// ============================================================================


macro_rules! define_uniform_struct {
    (
        $(#[$meta:meta])* struct $name:ident {
            $(
                $field_name:ident : $field_type:ty
            ),* $(,)?
        }
    ) => {
        // 1. 生成 Rust 结构体
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        $(#[$meta])*
        pub struct $name {
            $(
                pub $field_name : $field_type,
            )*
        }

        // 2. 生成获取 WGSL 代码的方法
        impl $name {
            pub fn wgsl_struct_def(struct_name: &str) -> String {
                let mut code = format!("struct {} {{\n", struct_name);
                $(
                    code.push_str(&format!(
                        "    {}: {},\n", 
                        stringify!($field_name), 
                        <$field_type as WgslType>::wgsl_type_name()
                    ));
                )*
                code.push_str("};");
                code
            }
        }
    };
}

// ============================================================================
// 3. Uniform 定义 (在此处修改，两端自动同步)
// ============================================================================
define_uniform_struct!(
    /// 全局 Uniforms (每个 Frame 更新)
    struct GlobalFrameUniforms {
        view_projection: Mat4,
        view_projection_inverse: Mat4,
        view_matrix: Mat4,
    }
);

impl Default for GlobalFrameUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY,
            view_projection_inverse: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
        }
    }
}


// Standard PBR Material
// 必须严格遵守 std140 对齐规则
define_uniform_struct!(
    struct MeshStandardUniforms {
        color: Vec4,           // 16
        emissive: Vec3,        // 12
        occlusion_strength: f32, // 4 (12+4=16)
        normal_scale: Vec2,    // 8
        roughness: f32,        // 4  
        metalness: f32,        // 4

        // 使用优化后的 Mat3A (48 bytes)
        map_transform: Mat3A,         
        normal_map_transform: Mat3A,   
        roughness_map_transform: Mat3A,
        metalness_map_transform: Mat3A,
        emissive_map_transform: Mat3A, 
        occlusion_map_transform: Mat3A,
    }
);

impl Default for MeshStandardUniforms {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            emissive: Vec3::ZERO,
            occlusion_strength: 1.0,
            normal_scale: Vec2::ONE,
            roughness: 1.0,
            metalness: 0.0,
            map_transform: Mat3A::IDENTITY,
            normal_map_transform: Mat3A::IDENTITY,
            roughness_map_transform: Mat3A::IDENTITY,
            metalness_map_transform: Mat3A::IDENTITY,
            emissive_map_transform: Mat3A::IDENTITY,
            occlusion_map_transform: Mat3A::IDENTITY,
        }
    }
}

// Basic Material
define_uniform_struct!(
    struct MeshBasicUniforms {
        color: Vec4,           // 16
        opacity: f32,          // 4
        _padding: PadVec3,      // 12 (4+12=16)
        map_transform: Mat3A,
    }
);

impl Default for MeshBasicUniforms {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            opacity: 1.0,
            _padding: PadVec3(Vec3::ZERO),
            map_transform: Mat3A::IDENTITY,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::mem;

//     #[test]
//     fn test_alignment() {
//         assert_eq!(mem::size_of::<MeshStandardUniforms>() % 16, 0, "Standard Uniforms not aligned to 16 bytes");
//         assert_eq!(mem::size_of::<MeshBasicUniforms>() % 16, 0, "Basic Uniforms not aligned to 16 bytes");
//     }
// }