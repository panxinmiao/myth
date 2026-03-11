use crate::resources::mesh::MAX_MORPH_TARGETS;
use glam::{Quat, Vec3, Vec4};
use smallvec::{SmallVec, smallvec};

/// Trait for types that support keyframe interpolation.
///
/// Implementations must provide linear and cubic Hermite spline interpolation.
/// The trait requires `Clone` for value copying during keyframe sampling.
pub trait Interpolatable: Clone + Sized {
    fn interpolate_linear(start: &Self, end: &Self, t: f32) -> Self;

    fn interpolate_cubic(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
    ) -> Self;

    fn interpolate_linear_into(start: &Self, end: &Self, t: f32, out: &mut Self) {
        *out = Self::interpolate_linear(start, end, t);
    }

    fn interpolate_cubic_into(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
        out: &mut Self,
    ) {
        *out = Self::interpolate_cubic(v0, out_tangent0, in_tangent1, v1, t, dt);
    }
}

/// Container for morph target blend weights.
///
/// Uses `SmallVec` to avoid heap allocation for common morph target counts
/// (up to [`MAX_MORPH_TARGETS`] weights stored inline on the stack).
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct MorphWeightData {
    pub weights: SmallVec<[f32; MAX_MORPH_TARGETS]>,
}

impl MorphWeightData {
    /// Creates a zero-initialized weight vector of the given length.
    #[must_use]
    pub fn allocate(n: usize) -> Self {
        Self {
            weights: smallvec![0.0; n],
        }
    }

    /// Performs zero-allocation linear interpolation, writing results directly
    /// into a pre-allocated output buffer.
    ///
    /// The number of weights written is `min(start.len, end.len, out_buffer.len)`.
    pub fn interpolate_linear_into(start: &Self, end: &Self, t: f32, out_buffer: &mut [f32]) {
        let len = start
            .weights
            .len()
            .min(end.weights.len())
            .min(out_buffer.len());
        // for i in 0..len {
        for (i, item) in out_buffer.iter_mut().enumerate().take(len) {
            *item = start.weights[i] + (end.weights[i] - start.weights[i]) * t;
        }
    }

    /// Performs zero-allocation cubic Hermite interpolation, writing results
    /// directly into a pre-allocated output buffer.
    pub fn interpolate_cubic_into(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
        out_buffer: &mut [f32],
    ) {
        let t2 = t * t;
        let t3 = t2 * t;
        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let len = v0.weights.len().min(v1.weights.len()).min(out_buffer.len());
        for (i, item) in out_buffer.iter_mut().enumerate().take(len) {
            let m0 = out_tangent0.weights[i] * dt;
            let m1 = in_tangent1.weights[i] * dt;
            *item = s0 * v0.weights[i] + s1 * m0 + s2 * v1.weights[i] + s3 * m1;
        }
    }
}

impl Interpolatable for MorphWeightData {
    fn interpolate_linear(start: &Self, end: &Self, t: f32) -> Self {
        let len = start.weights.len().max(end.weights.len());
        let mut result = MorphWeightData::allocate(len);
        for i in 0..len {
            let s = if i < start.weights.len() {
                start.weights[i]
            } else {
                0.0
            };
            let e = if i < end.weights.len() {
                end.weights[i]
            } else {
                0.0
            };
            result.weights[i] = s + (e - s) * t;
        }
        result
    }

    fn interpolate_cubic(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
    ) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;
        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let len = v0.weights.len().max(v1.weights.len());
        let mut result = MorphWeightData::allocate(len);

        for i in 0..len {
            let m0 = out_tangent0.weights[i] * dt;
            let m1 = in_tangent1.weights[i] * dt;
            result.weights[i] = s0 * v0.weights[i] + s1 * m0 + s2 * v1.weights[i] + s3 * m1;
        }
        result
    }

    fn interpolate_linear_into(start: &Self, end: &Self, t: f32, out: &mut Self) {
        let len = start.weights.len().min(end.weights.len());
        if out.weights.len() < len {
            out.weights.resize(len, 0.0);
        }
        for i in 0..len {
            out.weights[i] = start.weights[i] + (end.weights[i] - start.weights[i]) * t;
        }
    }

    fn interpolate_cubic_into(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
        out: &mut Self,
    ) {
        let len = v0.weights.len().min(v1.weights.len());
        if out.weights.len() < len {
            out.weights.resize(len, 0.0);
        }

        let t2 = t * t;
        let t3 = t2 * t;
        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        for i in 0..len {
            let m0 = out_tangent0.weights[i] * dt;
            let m1 = in_tangent1.weights[i] * dt;
            out.weights[i] = s0 * v0.weights[i] + s1 * m0 + s2 * v1.weights[i] + s3 * m1;
        }
    }
}

impl Interpolatable for f32 {
    fn interpolate_linear(start: &Self, end: &Self, t: f32) -> Self {
        start + (end - start) * t
    }

    fn interpolate_cubic(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
    ) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;

        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let m0 = out_tangent0 * dt;
        let m1 = in_tangent1 * dt;

        s0 * v0 + s1 * m0 + s2 * v1 + s3 * m1
    }
}

impl Interpolatable for Vec3 {
    fn interpolate_linear(start: &Self, end: &Self, t: f32) -> Self {
        start.lerp(*end, t)
    }

    fn interpolate_cubic(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
    ) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;

        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let m0 = out_tangent0 * dt;
        let m1 = in_tangent1 * dt;

        v0 * s0 + m0 * s1 + v1 * s2 + m1 * s3
    }
}

impl Interpolatable for Quat {
    fn interpolate_linear(start: &Self, end: &Self, t: f32) -> Self {
        start.slerp(*end, t)
    }

    fn interpolate_cubic(
        v0: &Self,
        out_tangent0: &Self,
        in_tangent1: &Self,
        v1: &Self,
        t: f32,
        dt: f32,
    ) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;

        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let v0_v = Vec4::from(*v0);
        let v1_v = Vec4::from(*v1);
        let m0_v = Vec4::from(*out_tangent0) * dt;
        let m1_v = Vec4::from(*in_tangent1) * dt;

        let result = v0_v * s0 + m0_v * s1 + v1_v * s2 + m1_v * s3;

        Quat::from_vec4(result).normalize()
    }
}
