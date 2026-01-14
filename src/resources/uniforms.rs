use glam::{Vec2, Vec3, Vec4, Mat3A, Mat4};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};
use std::collections::HashSet;


// ============================================================================
// 1. 类型映射 Trait (Rust Type -> WGSL Type String)
// ============================================================================
pub trait WgslType {
    fn wgsl_type_name() -> Cow<'static, str>;

    fn collect_wgsl_defs(_defs: &mut Vec<String>, _inserted: &mut HashSet<String>) {
        // 默认实现为空（针对 f32, vec3 等基础类型）
    }
}

impl WgslType for f32 { fn wgsl_type_name() -> Cow<'static, str> { "f32".into() } }
impl WgslType for i32 { fn wgsl_type_name() -> Cow<'static, str> { "i32".into() } }
impl WgslType for u32 { fn wgsl_type_name() -> Cow<'static, str> { "u32".into() } }
impl WgslType for Vec2 { fn wgsl_type_name() -> Cow<'static, str> { "vec2<f32>".into() } }
impl WgslType for Vec3 { fn wgsl_type_name() -> Cow<'static, str> { "vec3<f32>".into() } }
impl WgslType for Vec4 { fn wgsl_type_name() -> Cow<'static, str> { "vec4<f32>".into() } }
impl WgslType for Mat4 { fn wgsl_type_name() -> Cow<'static, str> { "mat4x4<f32>".into() } }
impl WgslType for Mat3A { fn wgsl_type_name() -> Cow<'static, str> { "mat3x3<f32>".into() } }


/// 专门用于 Uniform Buffer 的数组包装器
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformArray<T: Pod, const N: usize>(pub [T; N]);

unsafe impl<T: Pod, const N: usize> Zeroable for UniformArray<T, N> {}
unsafe impl<T: Pod, const N: usize> Pod for UniformArray<T, N> {}

// 1. 实现 WgslType：自动生成 array<T, N>
impl<T: WgslType + Pod, const N: usize> WgslType for UniformArray<T, N> {
    fn wgsl_type_name() -> Cow<'static, str> {
        // 动态生成包含长度的 WGSL 类型字符串
        format!("array<{}, {}>", T::wgsl_type_name(), N).into()
    }

    fn collect_wgsl_defs(defs: &mut Vec<String>, inserted: &mut HashSet<String>) {
        T::collect_wgsl_defs(defs, inserted);
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

pub trait WgslStruct: Pod + Zeroable {
    fn wgsl_struct_def(struct_name: &str) -> String;
}

// ============================================================================
// 2. 宏定义 (Single Source of Truth)
// ============================================================================

macro_rules! define_uniform_struct {
    // --------------------------------------------------------
    // 入口模式
    // --------------------------------------------------------
    (
        $(#[$meta:meta])* struct $name:ident {
            $(
                $vis:vis $field_name:ident : $field_type:ty $(= $default_val:expr)?
            ),* $(,)?
        }
    ) => {
        // 1. 生成 Rust Struct
        define_uniform_struct!(@def_struct 
            $(#[$meta])* struct $name { 
                $( $vis $field_name : $field_type ),* }
        );

        // 2. 生成 Default 实现
        define_uniform_struct!(@impl_default 
            $name { 
                $( $field_name : $field_type $(= $default_val)? ),* }
        );

        // 3. 生成 WgslType 实现 (支持作为嵌套字段)
        define_uniform_struct!(@impl_wgsl_type 
            $name { 
                $( $field_name : $field_type ),* }
        );

        // 4. 生成 UniformBlock 实现 (顶层入口)
        define_uniform_struct!(@impl_uniform_block 
            $name { 
                $( $field_name : $field_type ),* }
        );
    };

    (@def_struct $(#[$meta:meta])* struct $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        $(#[$meta])*
        pub struct $name {
            $( $vis $field_name : $field_type, )*
        }
    };

    (@impl_default $name:ident { $( $vis:vis $field_name:ident : $field_type:ty $(= $default_val:expr)? ),* }) => {
        impl Default for $name {
            fn default() -> Self {
                Self {
                    $( $field_name: define_uniform_struct!(@val_or_default $field_type $(, $default_val)?), )*
                }
            }
        }
    };
    (@val_or_default $type:ty, $val:expr) => { $val };
    (@val_or_default $type:ty) => { <$type as Default>::default() };


    // --------------------------------------------------------
    // 新增规则: 生成 WGSL 结构体定义逻辑 (供内部和外部使用)
    // --------------------------------------------------------
    (@gen_body $name_str:expr, { $( $vis:vis$field_name:ident : $field_type:ty ),* }) => {{
        let mut code = format!("struct {} {{\n", $name_str);
        $(
            if !stringify!($field_name).starts_with("__") { 
                code.push_str(&format!(
                    "    {}: {},\n", 
                    stringify!($field_name), 
                    <$field_type as WgslType>::wgsl_type_name()
                ));
            }
        )*
        code.push_str("};\n");
        code
    }};

    // --------------------------------------------------------
    // 内部规则 3: 实现 WgslType (让该结构体可以被嵌套)
    // --------------------------------------------------------
    (@impl_wgsl_type $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        impl WgslType for $name {
            fn wgsl_type_name() -> std::borrow::Cow<'static, str> {
                stringify!($name).into()
            }

            fn collect_wgsl_defs(defs: &mut Vec<String>, inserted: &mut std::collections::HashSet<String>) {
                // 1. 递归收集所有字段的定义 (依赖优先)
                $(
                    <$field_type as WgslType>::collect_wgsl_defs(defs, inserted);
                )*

                // 2. 生成自身的定义
                let my_name = stringify!($name);
                if !inserted.contains(my_name) {
                    let my_def = define_uniform_struct!(@gen_body my_name, { $( $field_name : $field_type ),* });
                    defs.push(my_def);
                    inserted.insert(my_name.to_string());
                }
            }
        }
    };

    // --------------------------------------------------------
    // 内部规则 4: 实现 UniformBlock (顶层调用)
    // --------------------------------------------------------
    (@impl_uniform_block $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        impl crate::resources::uniforms::WgslStruct for $name {
            fn wgsl_struct_def(struct_name: &str) -> String {
                let mut defs = Vec::new();
                let mut inserted = std::collections::HashSet::new();

                // 1. 收集所有字段的依赖 (例如 LightData)
                $(
                    <$field_type as WgslType>::collect_wgsl_defs(&mut defs, &mut inserted);
                )*

                // 2. 生成顶层结构体 (使用传入的 struct_name，可能被重命名)
                let top_def = define_uniform_struct!(@gen_body struct_name, { $( $vis $field_name : $field_type ),* });
                
                // 3. 拼接所有内容
                defs.push(top_def);
                defs.join("\n")
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
        pub world_matrix: Mat4,       //64
        pub world_matrix_inverse: Mat4,  //64
        pub normal_matrix: Mat3A,   //48

        pub(crate) __padding_20: UniformArray<f32, 20> = UniformArray::new([0.0; 20]),
    }
);


define_uniform_struct!(
    /// 全局 Uniforms (每个 Frame 更新)
    struct RenderStateUniforms {
        pub view_projection: Mat4 = Mat4::IDENTITY,
        pub view_projection_inverse: Mat4 = Mat4::IDENTITY,
        pub view_matrix: Mat4 = Mat4::IDENTITY,
        pub camera_position: Vec3 = Vec3::ZERO,
        pub time: f32 = 0.0,
    }
);


define_uniform_struct!(
    /// 全局 Uniforms (每个 Frame 更新)
    struct EnvironmentUniforms {
        pub ambient_light: Vec3 = Vec3::ZERO,
        pub num_lights: u32 = 0,
    }
);


// Basic Material
define_uniform_struct!(
    struct MeshBasicUniforms {
        pub color: Vec4 = Vec4::ONE,           // 16
        pub opacity: f32 = 1.0,          // 4
        pub(crate) __padding: UniformArray<f32, 3>,      // 12 (4+12=16)
        pub map_transform: Mat3A = Mat3A::IDENTITY,
    }
);

// Phong Material
define_uniform_struct!(
    struct MeshPhongUniforms {
        pub color: Vec4 = Vec4::ONE,

        pub specular: Vec3 = Vec3::ONE,
        pub opacity: f32 = 1.0,

        pub emissive: Vec3 = Vec3::ZERO,
        pub emissive_intensity: f32 = 1.0,

        pub normal_scale: Vec2 = Vec2::ONE,
        pub shininess: f32 = 30.0,
        pub(crate) __padding: f32, // 4 (12+4=16)

        pub map_transform: Mat3A = Mat3A::IDENTITY,
        pub normal_map_transform: Mat3A = Mat3A::IDENTITY,
        pub specular_map_transform: Mat3A = Mat3A::IDENTITY,
        pub emissive_map_transform: Mat3A = Mat3A::IDENTITY,
        pub light_map_transform: Mat3A = Mat3A::IDENTITY,
    }
);

// Standard PBR Material
define_uniform_struct!(
    struct MeshStandardUniforms {
        pub color: Vec4 = Vec4::ONE,           // 16
        pub emissive: Vec3 = Vec3::ZERO,        // 12
        pub occlusion_strength: f32 = 1.0,     // 4 (12+4=16)
        pub normal_scale: Vec2 = Vec2::ONE,    // 8
        pub roughness: f32 = 1.0,            // 4  
        pub metalness: f32 = 0.0,           // 4
        // 使用优化后的 Mat3A (48 bytes)
        pub map_transform: Mat3A = Mat3A::IDENTITY,         
        pub normal_map_transform: Mat3A = Mat3A::IDENTITY,   
        pub roughness_map_transform: Mat3A = Mat3A::IDENTITY,
        pub metalness_map_transform: Mat3A = Mat3A::IDENTITY,
        pub emissive_map_transform: Mat3A = Mat3A::IDENTITY, 
        pub occlusion_map_transform: Mat3A = Mat3A::IDENTITY,
    }
);

define_uniform_struct!(
    struct GpuLightStorage {
        // 16 bytes chunk 0
        pub color: Vec3,
        pub intensity: f32,

        // 16 bytes chunk 1
        pub position: Vec3,
        pub range: f32,

        // 16 bytes chunk 2
        pub direction: Vec3,
        pub decay: f32,

        pub inner_cone_cos: f32,
        pub outer_cone_cos: f32,

        pub light_type: u32,
        pub(crate) _padding1: f32,
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

    #[test]
    fn test_nested_wgsl() {
        let wgsl = GpuLightStorage::wgsl_struct_def("GpuLightStorage");
        println!("{}", wgsl);
    }
}