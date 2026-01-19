// src/animation/values.rs
use glam::{Vec3, Quat, Vec4};

pub trait Interpolatable: Copy + Clone + Sized {
    fn interpolate_linear(start: Self, end: Self, t: f32) -> Self;
    
    /// 三次样条插值
    /// 遵循 glTF 规范
    /// dt: 关键帧之间的时间差 (t1 - t0)，用于缩放切线
    fn interpolate_cubic(
        v0: Self, 
        out_tangent0: Self, 
        in_tangent1: Self, 
        v1: Self, 
        t: f32, 
        dt: f32
    ) -> Self;
}

// === f32 实现 ===
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

// === Vec3 实现 ===
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

        // glam::Vec3 支持直接 * f32 和 + 运算
        v0 * s0 + m0 * s1 + v1 * s2 + m1 * s3
    }
}

// === Quat 实现 (关键点：转为 Vec4 计算) ===
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

        // Quat 不能直接进行非归一化的线性运算，所以转为 Vec4
        let v0_v = Vec4::from(v0);
        let v1_v = Vec4::from(v1);
        let m0_v = Vec4::from(out_tangent0) * dt;
        let m1_v = Vec4::from(in_tangent1) * dt;

        let result = v0_v * s0 + m0_v * s1 + v1_v * s2 + m1_v * s3;
        
        // 计算完成后必须归一化
        Quat::from_vec4(result).normalize()
    }
}