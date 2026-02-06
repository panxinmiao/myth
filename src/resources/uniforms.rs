use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, UVec4, Vec2, Vec3, Vec4};
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

// ============================================================================
// Mat3Padded: A mat3x3<f32> with correct GPU alignment (48 bytes)
// ============================================================================
//
// In WGSL/WebGPU, mat3x3<f32> has the following layout:
// - Each column is a vec3<f32>, but aligned to 16 bytes
// - Total size: 3 columns × 16 bytes = 48 bytes
//
// glam::Mat3A provides this on native (via SIMD), but it's not Pod on WASM.
// glam::Mat3 is only 36 bytes (3 columns × 12 bytes).
//
// So we create our own type that's always 48 bytes on all platforms.
// ============================================================================

/// A mat3x3<f32> representation with correct GPU alignment (48 bytes total).
/// Each column is stored as a Vec4 (only xyz used, w is padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3Padded {
    /// Column 0 (x-axis), w is padding
    pub col0: Vec4,
    /// Column 1 (y-axis), w is padding
    pub col1: Vec4,
    /// Column 2 (z-axis), w is padding
    pub col2: Vec4,
}

unsafe impl Zeroable for Mat3Padded {}
unsafe impl Pod for Mat3Padded {}

impl Mat3Padded {
    pub const IDENTITY: Self = Self {
        col0: Vec4::new(1.0, 0.0, 0.0, 0.0),
        col1: Vec4::new(0.0, 1.0, 0.0, 0.0),
        col2: Vec4::new(0.0, 0.0, 1.0, 0.0),
    };

    pub const ZERO: Self = Self {
        col0: Vec4::ZERO,
        col1: Vec4::ZERO,
        col2: Vec4::ZERO,
    };

    #[must_use]
    pub fn new(col0: Vec3, col1: Vec3, col2: Vec3) -> Self {
        Self {
            col0: col0.extend(0.0),
            col1: col1.extend(0.0),
            col2: col2.extend(0.0),
        }
    }

    #[must_use]
    pub fn from_cols(col0: Vec3, col1: Vec3, col2: Vec3) -> Self {
        Self::new(col0, col1, col2)
    }

    /// Create from a column-major array (9 floats)
    /// Array layout: [col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z]
    #[must_use]
    pub fn from_cols_array(arr: &[f32; 9]) -> Self {
        Self {
            col0: Vec4::new(arr[0], arr[1], arr[2], 0.0),
            col1: Vec4::new(arr[3], arr[4], arr[5], 0.0),
            col2: Vec4::new(arr[6], arr[7], arr[8], 0.0),
        }
    }

    /// Create from Mat4 (extracts upper-left 3x3)
    #[must_use]
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            col0: Vec4::new(m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0),
            col1: Vec4::new(m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0),
            col2: Vec4::new(m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0),
        }
    }
}

impl From<Mat3> for Mat3Padded {
    fn from(m: Mat3) -> Self {
        Self {
            col0: m.x_axis.extend(0.0),
            col1: m.y_axis.extend(0.0),
            col2: m.z_axis.extend(0.0),
        }
    }
}

impl From<Mat4> for Mat3Padded {
    fn from(m: Mat4) -> Self {
        Self {
            col0: Vec4::new(m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0),
            col1: Vec4::new(m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0),
            col2: Vec4::new(m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<glam::Mat3A> for Mat3Padded {
    fn from(m: glam::Mat3A) -> Self {
        Self {
            col0: Vec4::new(m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0),
            col1: Vec4::new(m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0),
            col2: Vec4::new(m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0),
        }
    }
}

// Type alias for backward compatibility
pub type Mat3Uniform = Mat3Padded;

// ============================================================================
// 1. Type Mapping Trait (Rust Type -> WGSL Type String)
// ============================================================================
pub trait WgslType {
    fn wgsl_type_name() -> Cow<'static, str>;

    fn collect_wgsl_defs(_defs: &mut Vec<String>, _inserted: &mut HashSet<String>) {
        // Default implementation is empty (for primitive types like f32, vec3, etc.)
    }
}
impl WgslType for f32 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "f32".into()
    }
}
impl WgslType for i16 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "i16".into()
    }
}
impl WgslType for i32 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "i32".into()
    }
}
impl WgslType for u8 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "u8".into()
    }
}
impl WgslType for u16 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "u16".into()
    }
}
impl WgslType for u32 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "u32".into()
    }
}
impl WgslType for Vec2 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "vec2<f32>".into()
    }
}
impl WgslType for Vec3 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "vec3<f32>".into()
    }
}
impl WgslType for Vec4 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "vec4<f32>".into()
    }
}
impl WgslType for Mat4 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "mat4x4<f32>".into()
    }
}
impl WgslType for Mat3Uniform {
    fn wgsl_type_name() -> Cow<'static, str> {
        "mat3x3<f32>".into()
    }
}
impl WgslType for UVec4 {
    fn wgsl_type_name() -> Cow<'static, str> {
        "vec4<u32>".into()
    }
}

/// Array wrapper specifically for Uniform Buffer
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformArray<T: Pod, const N: usize>(pub [T; N]);

unsafe impl<T: Pod, const N: usize> Zeroable for UniformArray<T, N> {}
unsafe impl<T: Pod, const N: usize> Pod for UniformArray<T, N> {}

// 1. Implement WgslType: auto-generate array<T, N>
impl<T: WgslType + Pod, const N: usize> WgslType for UniformArray<T, N> {
    fn wgsl_type_name() -> Cow<'static, str> {
        // Dynamically generate WGSL type string with length
        format!("array<{}, {}>", T::wgsl_type_name(), N).into()
    }

    fn collect_wgsl_defs(defs: &mut Vec<String>, inserted: &mut HashSet<String>) {
        T::collect_wgsl_defs(defs, inserted);
    }
}

// 2. Implement Default: auto-initialize array
impl<T: Default + Pod + Copy, const N: usize> Default for UniformArray<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

// 3. Implement Deref: make it behave like a regular array
impl<T: Pod, const N: usize> Deref for UniformArray<T, N> {
    type Target = [T; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Pod, const N: usize> DerefMut for UniformArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// 4. Convenience constructors
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
// 2. Macro Definition (Single Source of Truth)
// ============================================================================

macro_rules! define_gpu_data_struct {
    // --------------------------------------------------------
    // Entry pattern
    // --------------------------------------------------------
    (
        $(#[$meta:meta])* struct $name:ident {
            $(
                $vis:vis $field_name:ident : $field_type:ty $(= $default_val:expr)?
            ),* $(,)?
        }
    ) => {
        // 1. Generate Rust Struct
        define_gpu_data_struct!(@def_struct
            $(#[$meta])* struct $name {
                $( $vis $field_name : $field_type ),* }
        );

        // 2. Generate Default implementation
        define_gpu_data_struct!(@impl_default
            $name {
                $( $field_name : $field_type $(= $default_val)? ),* }
        );

        // 3. Generate WgslType implementation (supports nested fields)
        define_gpu_data_struct!(@impl_wgsl_type
            $name {
                $( $field_name : $field_type ),* }
        );

        // 4. Generate UniformBlock implementation (top-level entry)
        define_gpu_data_struct!(@impl_uniform_block
            $name {
                $( $field_name : $field_type ),* }
        );

        // 5. Generate GpuData implementation
        define_gpu_data_struct!(@impl_gpu_data
            $name {
                $( $field_name : $field_type ),* }
        );
    };

    (@def_struct $(#[$meta:meta])* struct $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        $(#[$meta])*
        pub struct $name {
            $( $vis $field_name : $field_type, )*
        }
    };

    (@impl_default $name:ident { $( $vis:vis $field_name:ident : $field_type:ty $(= $default_val:expr)? ),* }) => {
        impl Default for $name {
            fn default() -> Self {
                Self {
                    $( $field_name: define_gpu_data_struct!(@val_or_default $field_type $(, $default_val)?), )*
                }
            }
        }
    };
    (@val_or_default $type:ty, $val:expr) => { $val };
    (@val_or_default $type:ty) => { <$type as Default>::default() };


    // --------------------------------------------------------
    // Rule: Generate WGSL struct definition logic (for internal and external use)
    // --------------------------------------------------------
    (@gen_body $name_str:expr, { $( $vis:vis$field_name:ident : $field_type:ty ),* }) => {{
        use std::fmt::Write;
        let mut code = format!("struct {} {{\n", $name_str);
        $(
            if !stringify!($field_name).starts_with("__") {
                let _ = writeln!(
                    code,
                    "    {}: {},",
                    stringify!($field_name),
                    <$field_type as WgslType>::wgsl_type_name()
                );
            }
        )*
        code.push_str("};\n");
        code
    }};

    // --------------------------------------------------------
    // Internal rule 3: Implement WgslType (allows struct to be nested)
    // --------------------------------------------------------
    (@impl_wgsl_type $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        impl WgslType for $name {
            fn wgsl_type_name() -> std::borrow::Cow<'static, str> {
                stringify!($name).into()
            }

            fn collect_wgsl_defs(defs: &mut Vec<String>, inserted: &mut std::collections::HashSet<String>) {
                // 1. Recursively collect all field definitions (dependencies first)
                $(
                    <$field_type as WgslType>::collect_wgsl_defs(defs, inserted);
                )*

                // 2. Generate own definition
                let my_name = stringify!($name);
                if !inserted.contains(my_name) {
                    let my_def = define_gpu_data_struct!(@gen_body my_name, { $( $field_name : $field_type ),* });
                    defs.push(my_def);
                    inserted.insert(my_name.to_string());
                }
            }
        }
    };

    // --------------------------------------------------------
    // Internal rule 4: Implement UniformBlock (top-level call)
    // --------------------------------------------------------
    (@impl_uniform_block $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        impl crate::resources::uniforms::WgslStruct for $name {
            fn wgsl_struct_def(struct_name: &str) -> String {
                let mut defs = Vec::new();
                let mut inserted = std::collections::HashSet::new();

                // 1. Collect all field dependencies (e.g. LightData)
                $(
                    <$field_type as WgslType>::collect_wgsl_defs(&mut defs, &mut inserted);
                )*

                // 2. Generate top-level struct (using passed struct_name, may be renamed)
                let top_def = define_gpu_data_struct!(@gen_body struct_name, { $( $vis $field_name : $field_type ),* });

                // 3. Concatenate all content
                defs.push(top_def);
                defs.join("\n")
            }
        }
    };

    // impl_gpu_data_for_pod
    (@impl_gpu_data $name:ident { $( $vis:vis $field_name:ident : $field_type:ty ),* }) => {
        impl $crate::resources::buffer::GpuData for $name {
            fn as_bytes(&self) -> &[u8] {
                bytemuck::bytes_of(self)
            }

            fn byte_size(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }
    };




}
// ============================================================================
// 3. GPU Data Struct Definitions (std140) (Modify here, both ends sync automatically)
// ============================================================================

define_gpu_data_struct!(
    /// Dynamic Model Uniforms (updated per object)
    struct DynamicModelUniforms {
        pub world_matrix: Mat4,       //64
        pub world_matrix_inverse: Mat4,  //64
        pub normal_matrix: Mat3Uniform,   //48

        pub(crate) __padding_20: UniformArray<f32, 20> = UniformArray::new([0.0; 20]),
    }
);

define_gpu_data_struct!(
    /// Global Uniforms (updated per frame)
    struct RenderStateUniforms {
        pub view_projection: Mat4 = Mat4::IDENTITY,
        pub view_projection_inverse: Mat4 = Mat4::IDENTITY,
        pub view_matrix: Mat4 = Mat4::IDENTITY,
        pub camera_position: Vec3 = Vec3::ZERO,
        pub time: f32 = 0.0,
    }
);

define_gpu_data_struct!(
    /// Global Uniforms (updated per frame)
    struct EnvironmentUniforms {
        pub ambient_light: Vec3 = Vec3::ZERO,
        pub num_lights: u32 = 0,

        pub env_map_intensity: f32 = 1.0,
        pub env_map_max_mip_level: f32 = 0.0,
        pub(crate) __padding: UniformArray<f32, 2>,
    }
);

// Basic Material
define_gpu_data_struct!(
    struct MeshBasicUniforms {
        pub color: Vec4 = Vec4::ONE,           // 16

        pub opacity: f32 = 1.0,          // 4
        pub alpha_test: f32 = 0.0,       // 4
        pub(crate) __padding: UniformArray<f32, 2>,      // 8 (4+4+8=16)

        pub map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
    }
);

// Phong Material
define_gpu_data_struct!(
    struct MeshPhongUniforms {
        pub color: Vec4 = Vec4::ONE,

        pub specular: Vec3 = Vec3::splat(0.06667),  // 0x111111. Represents non-metallic (dielectric) materials like plastic, rock, or water
        pub opacity: f32 = 1.0,

        pub emissive: Vec3 = Vec3::ZERO,
        pub emissive_intensity: f32 = 1.0,

        pub normal_scale: Vec2 = Vec2::ONE,
        pub shininess: f32 = 30.0,
        pub alpha_test: f32 = 0.0,

        pub map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub normal_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub specular_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub emissive_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub light_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
    }
);

// Standard PBR Material
define_gpu_data_struct!(
    struct MeshStandardUniforms {
        pub color: Vec4 = Vec4::ONE,           // 16

        pub emissive: Vec3 = Vec3::ZERO,        // 12
        pub emissive_intensity: f32 = 1.0,    // 4

        pub roughness: f32 = 1.0,            // 4  
        pub metalness: f32 = 0.0,           // 4
        pub opacity: f32 = 1.0,            // 4
        pub alpha_test: f32 = 0.0,              // 4 (8+4+4=16)

        pub normal_scale: Vec2 = Vec2::ONE,    // 8
        pub ao_map_intensity: f32 = 1.0,     // 4
        pub(crate) __padding: f32,   // 4 (8+4+4=16)

        pub specular: Vec3 = Vec3::ONE,               // 12
        pub specular_intensity: f32 = 1.0,    // 4

        // Using optimized Mat3Uniform (48 bytes)
        pub map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub normal_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub roughness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub metalness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub emissive_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub ao_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub specular_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub specular_intensity_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
    }
);

define_gpu_data_struct!(
    struct MeshPhysicalUniforms {
        pub color: Vec4 = Vec4::ONE,          // 16

        pub emissive: Vec3 = Vec3::ZERO,      // 12
        pub emissive_intensity: f32 = 1.0,    // 4

        pub roughness: f32 = 1.0,            // 4  
        pub metalness: f32 = 0.0,            // 4
        pub opacity: f32 = 1.0,              // 4
        pub alpha_test: f32 = 0.0,           // 4 

        pub normal_scale: Vec2 = Vec2::ONE,   // 8
        pub ao_map_intensity: f32 = 1.0,      // 4
        pub ior: f32 = 1.5,                   // 4

        pub specular_color: Vec3 = Vec3::ONE,  // 12
        pub specular_intensity: f32 = 1.0,     // 4

        pub clearcoat: f32 = 0.0,                       // 4
        pub clearcoat_roughness: f32 = 0.0,             // 4
        pub clearcoat_normal_scale: Vec2 = Vec2::ONE,   // 8

        pub sheen_color: Vec3 = Vec3::ZERO,             // 12
        pub sheen_roughness: f32 = 1.0,                 // 4

        pub iridescence: f32 = 0.0,                    // 4
        pub iridescence_ior: f32 = 1.3,                // 4
        pub iridescence_thickness_min: f32 = 100.0,    // 4
        pub iridescence_thickness_max: f32 = 400.0,    // 4

        pub anisotropy_vector: Vec2 = Vec2::ZERO,      // 8
        pub transmission: f32 = 0.0,               // 4
        pub thickness: f32 = 0.0,                  // 4

        pub attenuation_color: Vec3 = Vec3::ONE,   // 12
        pub attenuation_distance: f32 = -1.0,     // 4    -1 (infinite)

        pub dispersion: f32 = 0.0,                 // 4
        pub(crate) __padding1: f32,          // 4
        pub(crate) __padding2: f32,          // 4
        pub(crate) __padding3: f32,          // 4



        // Using optimized Mat3Uniform (48 bytes)
        pub map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub normal_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub roughness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub metalness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub emissive_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub ao_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub specular_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub specular_intensity_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub clearcoat_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub clearcoat_normal_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub clearcoat_roughness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub sheen_color_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub sheen_roughness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub iridescence_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub iridescence_thickness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub anisotropy_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub transmission_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
        pub thickness_map_transform: Mat3Uniform = Mat3Uniform::IDENTITY,
    }
);

define_gpu_data_struct!(
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
        pub(crate) __padding1: f32,
    }
);

define_gpu_data_struct!(
    struct MorphUniforms {
        pub count: u32,
        pub vertex_count: u32,
        pub flags: u32,
        pub(crate) __padding: u32,

        // 32 morph target weights and indices, packed into Vec4 to satisfy Uniform buffer 16-byte alignment requirement
        // weights[0] = Vec4(w0, w1, w2, w3), weights[1] = Vec4(w4, w5, w6, w7), ...
        pub weights: UniformArray<Vec4, 8>,
        pub indices: UniformArray<UVec4, 8>,
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_alignment() {
        assert_eq!(
            mem::size_of::<MeshStandardUniforms>() % 16,
            0,
            "Standard Uniforms not aligned to 16 bytes"
        );
        assert_eq!(
            mem::size_of::<MeshBasicUniforms>() % 16,
            0,
            "Basic Uniforms not aligned to 16 bytes"
        );
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
