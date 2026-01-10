use glam::{Vec2, Vec3, Vec4, Mat3A, Mat4};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};


// ============================================================================
// 1. 类型映射 Trait (Rust Type -> WGSL Type String)
// ============================================================================

pub trait WgslType {
    fn wgsl_type_name() -> Cow<'static, str>;
}

impl WgslType for f32 { fn wgsl_type_name() -> Cow<'static, str> { "f32".into() } }
impl WgslType for i32 { fn wgsl_type_name() -> Cow<'static, str> { "i32".into() } }
impl WgslType for u32 { fn wgsl_type_name() -> Cow<'static, str> { "u32".into() } }
impl WgslType for Vec2 { fn wgsl_type_name() -> Cow<'static, str> { "vec2<f32>".into() } }
impl WgslType for Vec3 { fn wgsl_type_name() -> Cow<'static, str> { "vec3<f32>".into() } }
impl WgslType for Vec4 { fn wgsl_type_name() -> Cow<'static, str> { "vec4<f32>".into() } }
impl WgslType for Mat4 { fn wgsl_type_name() -> Cow<'static, str> { "mat4x4<f32>".into() } }
impl WgslType for Mat3A { fn wgsl_type_name() -> Cow<'static, str> { "mat3x3<f32>".into() } }


// ----------------------------------------------------------------------------
// 2. 核心：自定义数组类型 UniformArray
// ----------------------------------------------------------------------------
/// 专门用于 Uniform Buffer 的数组包装器
/// 自动处理 WGSL 类型映射和 Default 实现
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformArray<T: Pod, const N: usize>(pub [T; N]);

// 1. 手动实现 Zeroable (安全：只要内部数组是 Zeroable，且布局透明)
unsafe impl<T: Pod, const N: usize> Zeroable for UniformArray<T, N> {}

// 2. 手动实现 Pod (安全：只要内部数组是 Pod，且布局透明)
unsafe impl<T: Pod, const N: usize> Pod for UniformArray<T, N> {}

// 1. 实现 WgslType：自动生成 array<T, N>
impl<T: WgslType + Pod, const N: usize> WgslType for UniformArray<T, N> {
    fn wgsl_type_name() -> Cow<'static, str> {
        // 动态生成包含长度的 WGSL 类型字符串
        format!("array<{}, {}>", T::wgsl_type_name(), N).into()
    }
}

// 2. 实现 Default：自动初始化数组
impl<T: Default + Pod + Copy, const N: usize> Default for UniformArray<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

// 3. 实现 Deref：让它用起来像普通数组
impl<T: Pod, const N: usize> Deref for UniformArray<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Pod, const N: usize> DerefMut for UniformArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

// 4. 便捷构造函数
impl<T: Pod, const N: usize> UniformArray<T, N> {
    pub fn new(arr: [T; N]) -> Self {
        Self(arr)
    }
}

impl<T: Pod, const N: usize> From<[T; N]> for UniformArray<T, N> {
    fn from(arr: [T; N]) -> Self {
        Self(arr)
    }
}

pub trait UniformBlock: Pod + Zeroable {
    fn wgsl_struct_def(struct_name: &str) -> String;
}




// 单个光源数据 (必须 16 字节对齐)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuLightData {
    // color(rgb) + intensity(a)
    pub color: [f32; 4], 
    // position(xyz) + range(w)
    pub position: [f32; 4], 
    // direction(xyz) + spot_angle(w)
    pub direction: [f32; 4],
    // extra params
    pub info: [f32; 4], 
}

pub const MAX_DIR_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 16;
pub const MAX_SPOT_LIGHTS: usize = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GlobalLightUniforms {
    pub ambient: [f32; 4], // 环境光
    
    // 计数器 (注意对齐)
    pub num_dir_lights: u32,
    pub num_point_lights: u32,
    pub num_spot_lights: u32,
    pub __pad: u32, 

    // 灯光数组
    pub dir_lights: [GpuLightData; MAX_DIR_LIGHTS],
    pub point_lights: [GpuLightData; MAX_POINT_LIGHTS],
    pub spot_lights: [GpuLightData; MAX_SPOT_LIGHTS],
}

impl Default for GlobalLightUniforms {
    fn default() -> Self {
        Self {
            ambient: [0.05, 0.05, 0.05, 1.0],
            num_dir_lights: 0,
            num_point_lights: 0,
            num_spot_lights: 0,
            __pad: 0,
            dir_lights: [GpuLightData::default(); MAX_DIR_LIGHTS],
            point_lights: [GpuLightData::default(); MAX_POINT_LIGHTS],
            spot_lights: [GpuLightData::default(); MAX_SPOT_LIGHTS],
        }
    }
}

impl UniformBlock for GlobalLightUniforms {
    fn wgsl_struct_def(struct_name: &str) -> String {
        let mut code = format!("struct {} {{\n", struct_name);
        code.push_str("    ambient: vec4<f32>,\n");
        code.push_str("    num_dir_lights: u32,\n");
        code.push_str("    num_point_lights: u32,\n");
        code.push_str("    num_spot_lights: u32,\n");
        // code.push_str("    __pad: u32,\n");
        code.push_str(&format!("    dir_lights: array<LightData, {}>,\n", MAX_DIR_LIGHTS));
        code.push_str(&format!("    point_lights: array<LightData, {}>,\n", MAX_POINT_LIGHTS));
        code.push_str(&format!("    spot_lights: array<LightData, {}>,\n", MAX_SPOT_LIGHTS));
        code.push_str("};\n");

        // LightData 结构体定义
        code.push_str("struct LightData {\n");
        code.push_str("    color: vec4<f32>,\n");
        code.push_str("    position: vec4<f32>,\n");
        code.push_str("    direction: vec4<f32>,\n");
        code.push_str("    info: vec4<f32>,\n");
        code.push_str("};\n");

        code
    }
}


// ============================================================================
// 2. 宏定义 (Single Source of Truth)
// ============================================================================


macro_rules! define_uniform_struct {
    // --------------------------------------------------------
    // 入口模式：匹配 struct 定义
    // --------------------------------------------------------
    (
        $(#[$meta:meta])* struct $name:ident {
            $(
                $field_name:ident : $field_type:ty $(= $default_val:expr)?
            ),* $(,)?
        }
    ) => {
        // 1. 调用内部规则生成 struct 定义 (忽略默认值)
        define_uniform_struct!(@def_struct 
            $(#[$meta])* struct $name { 
                $( $field_name : $field_type ),* }
        );

        // 2. 调用内部规则生成 Default 实现 (处理默认值)
        define_uniform_struct!(@impl_default 
            $name { 
                $( $field_name : $field_type $(= $default_val)? ),* }
        );

        // 3. 调用内部规则生成 WGSL 代码 (忽略默认值)
        define_uniform_struct!(@impl_wgsl 
            $name { 
                $( $field_name : $field_type ),* }
        );
    };

    // --------------------------------------------------------
    // 内部规则 1: 生成 Rust Struct
    // --------------------------------------------------------
    (@def_struct $(#[$meta:meta])* struct $name:ident { $( $field_name:ident : $field_type:ty ),* }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        $(#[$meta])*
        pub struct $name {
            $(
                pub $field_name : $field_type,
            )*
        }
    };

    // --------------------------------------------------------
    // 内部规则 2: 生成 Default 实现
    // --------------------------------------------------------
    (@impl_default $name:ident { $( $field_name:ident : $field_type:ty $(= $default_val:expr)? ),* }) => {
        impl Default for $name {
            fn default() -> Self {
                Self {
                    $(
                        $field_name: define_uniform_struct!(@val_or_default $field_type $(, $default_val)?),
                    )*
                }
            }
        }
    };

    // 辅助规则：如果有显式值，则使用显式值；否则使用 Type::default()
    (@val_or_default $type:ty, $val:expr) => { $val };
    (@val_or_default $type:ty) => { <$type as Default>::default() };

    // --------------------------------------------------------
    // 内部规则 3: 生成 WGSL struct string
    // --------------------------------------------------------
    (@impl_wgsl $name:ident { $( $field_name:ident : $field_type:ty ),* }) => {
        impl crate::core::uniforms::UniformBlock for $name {
            fn wgsl_struct_def(struct_name: &str) -> String {
                let mut code = format!("struct {} {{\n", struct_name);
                $(
                    // 忽略以 __ 开头的 padding 字段
                    if !stringify!($field_name).starts_with("__") { 
                        code.push_str(&format!(
                            "    {}: {},\n", 
                            stringify!($field_name), 
                            <$field_type as WgslType>::wgsl_type_name()
                        ));
                    }
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
    /// 动态模型 Uniforms (每个对象更新)
    struct DynamicModelUniforms {
        model_matrix: Mat4,       //64
        model_matrix_inverse: Mat4,  //64
        normal_matrix: Mat3A,   //48

        __padding_20: UniformArray<f32, 20>, // 填充至 256 bytes
    }
);


define_uniform_struct!(
    /// 全局 Uniforms (每个 Frame 更新)
    struct GlobalFrameUniforms {
        view_projection: Mat4 = Mat4::IDENTITY,
        view_projection_inverse: Mat4 = Mat4::IDENTITY,
        view_matrix: Mat4 = Mat4::IDENTITY,
    }
);


// define_uniform_struct!(
//     /// 光源 Uniforms (每个 Frame 更新)
//     struct GlobalLightUniforms {
//         directional_light_direction: Vec3,
//         __padding1: f32, // Padding
//         directional_light_color: Vec3,
//         __padding2: f32, // Padding
//     }
// );

// Standard PBR Material
// 必须严格遵守 std140 对齐规则
define_uniform_struct!(
    struct MeshStandardUniforms {
        color: Vec4 = Vec4::ONE,           // 16
        emissive: Vec3 = Vec3::ZERO,        // 12
        occlusion_strength: f32 = 1.0,     // 4 (12+4=16)
        normal_scale: Vec2 = Vec2::ONE,    // 8
        roughness: f32 = 1.0,            // 4  
        metalness: f32 = 0.0,           // 4
        // 使用优化后的 Mat3A (48 bytes)
        map_transform: Mat3A = Mat3A::IDENTITY,         
        normal_map_transform: Mat3A = Mat3A::IDENTITY,   
        roughness_map_transform: Mat3A = Mat3A::IDENTITY,
        metalness_map_transform: Mat3A = Mat3A::IDENTITY,
        emissive_map_transform: Mat3A = Mat3A::IDENTITY, 
        occlusion_map_transform: Mat3A = Mat3A::IDENTITY,
    }
);


// Basic Material
define_uniform_struct!(
    struct MeshBasicUniforms {
        color: Vec4 = Vec4::ONE,           // 16
        opacity: f32 = 1.0,          // 4
        __padding: UniformArray<f32, 3>,      // 12 (4+12=16)
        map_transform: Mat3A = Mat3A::IDENTITY,
    }
);


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_alignment() {
        assert_eq!(mem::size_of::<MeshStandardUniforms>() % 16, 0, "Standard Uniforms not aligned to 16 bytes");
        assert_eq!(mem::size_of::<MeshBasicUniforms>() % 16, 0, "Basic Uniforms not aligned to 16 bytes");
    }

    #[test]
    fn test_wgsl_generation() {
        let standard_wgsl = MeshStandardUniforms::wgsl_struct_def("MeshStandardUniforms");
        let standard_default = MeshStandardUniforms::default();
        println!("WGSL for MeshStandardUniforms:\n{}", standard_wgsl);
        println!("Default Standard Uniforms: {:?}", standard_default);

        let basic_wgsl = DynamicModelUniforms::wgsl_struct_def("DynamicModelUniforms");
        let basic_default = DynamicModelUniforms::default();
        println!("WGSL for DynamicModelUniforms:\n{}", basic_wgsl);
        println!("Default DynamicModelUniforms: {:?}", basic_default);
    }
}