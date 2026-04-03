use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, UVec4, Vec2, Vec3, Vec4};
use myth_macros::gpu_struct;
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

/// A `mat3x3<f32>` representation with correct GPU alignment (48 bytes total).
/// Each column is stored as a `Vec4` (only xyz used, w is padding).
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

/// SAFETY: Mat3Padded is a plain data structure with no padding bytes used for actual data (the w components are just padding).
/// It is safe to treat it as a byte array for GPU upload.
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

/// SAFETY: UniformArray is a transparent wrapper around [T; N], and T is Pod,
/// so it is safe to treat UniformArray<T, N> as a byte array for GPU upload.
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
// GPU Data Struct Definitions (std140, auto-padded by #[gpu_struct])
// ============================================================================

// Material uniform structs (UnlitUniforms, PhongUniforms, PhysicalUniforms)
// are auto-generated by the #[myth_material] proc macro.
pub use crate::material::PhongUniforms;
pub use crate::material::PhysicalUniforms;
pub use crate::material::UnlitUniforms;

/// Per-object dynamic uniforms uploaded to the GPU each draw call.
///
/// Uses `dynamic_offset = true` to enforce 256-byte alignment, as required
/// by wgpu's dynamic uniform buffer binding.
#[gpu_struct(dynamic_offset = true, crate_path = "crate")]
pub struct DynamicModelUniforms {
    pub world_matrix: Mat4,
    pub world_matrix_inverse: Mat4,
    pub normal_matrix: Mat3Uniform,
    pub previous_world_matrix: Mat4,
    pub instance_tint: Vec4,
}

/// Global render state uniforms updated once per frame.
///
/// Contains camera matrices, jitter parameters, and timing data used by
/// all rendering passes.
#[gpu_struct(crate_path = "crate")]
pub struct RenderStateUniforms {
    #[default(Mat4::IDENTITY)]
    pub view_projection: Mat4,
    #[default(Mat4::IDENTITY)]
    pub view_projection_inverse: Mat4,

    #[default(Mat4::IDENTITY)]
    pub projection_matrix: Mat4,
    #[default(Mat4::IDENTITY)]
    pub projection_inverse: Mat4,

    #[default(Mat4::IDENTITY)]
    pub view_matrix: Mat4,

    #[default(Mat4::IDENTITY)]
    pub prev_view_projection: Mat4,

    #[default(Mat4::IDENTITY)]
    pub unjittered_view_projection: Mat4,
    #[default(Mat4::IDENTITY)]
    pub prev_unjittered_view_projection: Mat4,

    pub camera_position: Vec3,
    #[default(0.1)]
    pub camera_near: f32,

    pub jitter: Vec2,
    pub prev_jitter: Vec2,

    #[default(1000.0)]
    pub camera_far: f32,
    pub time: f32,
    pub time_cycle_2pi: f32,
    pub delta_time: f32,
}

/// Environment lighting uniforms updated once per frame.
#[gpu_struct(crate_path = "crate")]
pub struct EnvironmentUniforms {
    pub ambient_light: Vec3,
    pub num_lights: u32,

    #[default(1.0)]
    pub env_map_intensity: f32,
    pub env_map_rotation: f32,
    pub env_map_max_mip_level: f32,
}

/// Per-light GPU data including shadow cascade parameters.
#[gpu_struct(crate_path = "crate")]
pub struct GpuLightStorage {
    pub color: Vec3,
    pub intensity: f32,

    pub position: Vec3,
    pub range: f32,

    pub direction: Vec3,
    pub decay: f32,

    pub inner_cone_cos: f32,
    pub outer_cone_cos: f32,

    pub light_type: u32,
    /// Base layer index into the 2D shadow array (−1 if no 2D shadow).
    pub shadow_layer_index: i32,

    pub shadow_bias: f32,
    pub shadow_normal_bias: f32,
    pub cascade_count: u32,
    /// Base cube index into the cube array shadow map (−1 if no point shadow).
    pub point_shadow_index: i32,

    /// Cascade split distances (view-space depth thresholds).
    pub cascade_splits: Vec4,

    /// Shadow VP matrices: up to 4 cascades for directional, 1 for spot.
    pub shadow_matrices: UniformArray<Mat4, 4>,
}

/// Morph target animation uniforms.
///
/// Weights and indices are packed into Vec4/UVec4 to satisfy the uniform
/// buffer 16-byte alignment requirement.
#[gpu_struct(crate_path = "crate")]
pub struct MorphUniforms {
    pub count: u32,
    pub vertex_count: u32,
    pub flags: u32,

    pub weights: UniformArray<Vec4, 32>,
    pub prev_weights: UniformArray<Vec4, 32>,
    pub indices: UniformArray<UVec4, 32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_alignment() {
        assert_eq!(
            mem::size_of::<PhysicalUniforms>() % 16,
            0,
            "Physical Uniforms not aligned to 16 bytes"
        );
        assert_eq!(
            mem::size_of::<UnlitUniforms>() % 16,
            0,
            "Unlit Uniforms not aligned to 16 bytes"
        );
        assert_eq!(
            mem::size_of::<PhongUniforms>() % 16,
            0,
            "Phong Uniforms not aligned to 16 bytes"
        );
    }

    #[test]
    fn test_wgsl_generation() {
        let physical_wgsl = PhysicalUniforms::wgsl_struct_def("PhysicalUniforms");
        let physical_default = PhysicalUniforms::default();
        println!("WGSL for PhysicalUniforms:\n{physical_wgsl}");
        println!("Default Physical Uniforms: {physical_default:?}");

        let basic_wgsl = DynamicModelUniforms::wgsl_struct_def("DynamicModelUniforms");
        let basic_default = DynamicModelUniforms::default();
        println!("WGSL for DynamicModelUniforms:\n{basic_wgsl}");
        println!("Default DynamicModelUniforms: {basic_default:?}");
    }

    #[test]
    fn test_nested_wgsl() {
        let wgsl = GpuLightStorage::wgsl_struct_def("GpuLightStorage");
        println!("{wgsl}");
    }
}
