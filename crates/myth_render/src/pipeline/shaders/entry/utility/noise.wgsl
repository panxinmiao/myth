fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3D(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash13(i);
    let b = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let f1 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash13(i + vec3<f32>(1.0, 1.0, 1.0));

    let x1 = mix(a, b, u.x);
    let x2 = mix(c, d, u.x);
    let x3 = mix(e, f1, u.x);
    let x4 = mix(g, h, u.x);

    let y1 = mix(x1, x2, u.y);
    let y2 = mix(x3, x4, u.y);

    return mix(y1, y2, u.z);
}


fn fbm3D(p: vec3<f32>) -> f32 {
    var f = 0.0;
    var w = 0.5;
    var current_p = p;

    for (var i = 0; i < 4; i++) {
        f += w * noise3D(current_p);
        current_p *= 2.0;
        w *= 0.5;
    }
    return f;
}