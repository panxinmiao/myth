use glam::{Vec3, Quat, Vec4};
use crate::resources::mesh::MAX_MORPH_TARGETS;

pub trait Interpolatable: Copy + Clone + Sized {
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self;
    
    fn interpolate_cubic(
        v0: Self, 
        out_tangent0: Self, 
        in_tangent1: Self, 
        v1: Self, 
        t: f32, 
        dt: f32
    ) -> Self;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MorphWeightData {
    pub weights: [f32; MAX_MORPH_TARGETS],
}

impl Interpolatable for MorphWeightData {
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self {
        let mut result = MorphWeightData::default();
        for i in 0..MAX_MORPH_TARGETS {
            result.weights[i] = start.weights[i] + (end.weights[i] - start.weights[i]) * t;
        }
        result
    }

    fn interpolate_cubic(v0: Self, out_tangent0: Self, in_tangent1: Self, v1: Self, t: f32, dt: f32) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;
        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;
        
        let mut result = MorphWeightData::default();
        for i in 0..MAX_MORPH_TARGETS {
            let m0 = out_tangent0.weights[i] * dt;
            let m1 = in_tangent1.weights[i] * dt;
            result.weights[i] = s0 * v0.weights[i] + s1 * m0 + s2 * v1.weights[i] + s3 * m1;
        }
        result
    }
}

impl Interpolatable for f32 {
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self {
        start + (end - start) * t
    }

    fn interpolate_cubic(v0: Self, out_tangent0: Self, in_tangent1: Self, v1: Self, t: f32, dt: f32) -> Self {
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
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self {
        start.lerp(end, t)
    }

    fn interpolate_cubic(v0: Self, out_tangent0: Self, in_tangent1: Self, v1: Self, t: f32, dt: f32) -> Self {
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
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self {
        start.slerp(end, t)
    }

    fn interpolate_cubic(v0: Self, out_tangent0: Self, in_tangent1: Self, v1: Self, t: f32, dt: f32) -> Self {
        let t2 = t * t;
        let t3 = t2 * t;

        let s2 = -2.0 * t3 + 3.0 * t2;
        let s3 = t3 - t2;
        let s0 = 1.0 - s2;
        let s1 = s3 - t2 + t;

        let v0_v = Vec4::from(v0);
        let v1_v = Vec4::from(v1);
        let m0_v = Vec4::from(out_tangent0) * dt;
        let m1_v = Vec4::from(in_tangent1) * dt;

        let result = v0_v * s0 + m0_v * s1 + v1_v * s2 + m1_v * s3;
        
        Quat::from_vec4(result).normalize()
    }
}