// ============================================================================
// Sky-to-Cubemap Bake Compute Shader
// ============================================================================
//
// Samples the Sky-View LUT to render each face of a cubemap. The resulting
// cubemap is then fed into the existing IBL PMREM prefiltering pipeline.
//
// Output: 2D array storage texture (NxN, 6 layers, Rgba16Float)
//
// Dispatch: (N/8, N/8, 6) workgroups

const PI: f32 = 3.14159265358979323846;

struct BakeParams {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_disk_size: f32,
    exposure: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var sky_view_lut: texture_2d<f32>;

@group(0) @binding(1)
var lut_sampler: sampler;

@group(0) @binding(2)
var<uniform> params: BakeParams;

@group(0) @binding(3)
var dest: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(4)
var transmittance_lut: texture_2d<f32>;

/// Compute cubemap face direction from texel coordinates.
fn get_cube_direction(face: u32, uv_: vec2<f32>) -> vec3<f32> {
    let uv = 2.0 * uv_ - 1.0;
    switch (face) {
        case 0u: { return vec3<f32>( 1.0, -uv.y, -uv.x); } // +X
        case 1u: { return vec3<f32>(-1.0, -uv.y,  uv.x); } // -X
        case 2u: { return vec3<f32>( uv.x,  1.0,  uv.y); } // +Y
        case 3u: { return vec3<f32>( uv.x, -1.0, -uv.y); } // -Y
        case 4u: { return vec3<f32>( uv.x, -uv.y,  1.0); } // +Z
        case 5u: { return vec3<f32>(-uv.x, -uv.y, -1.0); } // -Z
        default: { return vec3<f32>(0.0); }
    }
}

/// Inverse of the sky-view LUT non-linear mapping.
/// Converts a world direction to UV coordinates in the sky-view LUT.
fn direction_to_sky_view_uv(dir: vec3<f32>) -> vec2<f32> {
    // Latitude (elevation angle)
    let theta = asin(clamp(dir.y, -1.0, 1.0));

    // Non-linear latitude mapping (matches sky_view_lut.wgsl)
    var v: f32;
    if theta < 0.0 {
        let coord = sqrt(-theta / (PI * 0.5));
        v = 0.5 - 0.5 * coord;
    } else {
        let coord = sqrt(theta / (PI * 0.5));
        v = 0.5 + 0.5 * coord;
    }

    // Longitude (azimuth angle)
    let phi = atan2(dir.x, dir.z);
    let u = (phi + PI) / (2.0 * PI);

    return vec2<f32>(u, v);
}

fn ray_sphere_intersect(o: vec3<f32>, d: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(d, d);
    let b = 2.0 * dot(d, o);
    let c = dot(o, o) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let sq = sqrt(discriminant);
    return vec2<f32>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
}

// 【新增】透射率 UV 映射（Hillaire 模型标准）
fn transmittance_lut_uv(altitude: f32, cos_zenith: f32) -> vec2<f32> {
    let planet_radius = 6360000.0;
    let atmosphere_radius = 6460000.0;
    let H = sqrt(max(0.0, atmosphere_radius * atmosphere_radius - planet_radius * planet_radius));
    let rho = sqrt(max(0.0, (planet_radius + altitude) * (planet_radius + altitude) - planet_radius * planet_radius));
    let d = ray_sphere_intersect(
        vec3<f32>(0.0, planet_radius + altitude, 0.0),
        vec3<f32>(0.0, cos_zenith, sqrt(max(0.0, 1.0 - cos_zenith * cos_zenith))),
        atmosphere_radius
    ).y;
    let d_min = atmosphere_radius - planet_radius - altitude;
    let d_max = rho + H;
    let x_mu = (d - d_min) / (d_max - d_min);
    let x_r = rho / H;
    return vec2<f32>(x_mu, x_r);
}

/// Approximate sun disk rendering.
fn sun_disk(dir: vec3<f32>, sun_dir: vec3<f32>, sun_size_deg: f32) -> vec3<f32> {
    let cos_angle = dot(dir, sun_dir);
    // 注意：输入是直径，必须除以 2 变成半径
    let sun_angular_radius = (sun_size_deg * 0.5) * PI / 180.0;
    let sun_cos = cos(sun_angular_radius);
    
    // 修复 smoothstep 越界问题：平滑范围向内收缩，避免超过 1.0
    let smoothing = 0.00002; 
    let t = smoothstep(sun_cos - smoothing, sun_cos + smoothing, cos_angle);
    
    if t > 0.0 {
        // 计算阳光穿过大气的透射率 (Transmittance)
        // 假设我们在地表上方 1.0 米处观察
        let planet_radius = 6360000.0;
        let altitude = 1.0; 
        let d_planet = ray_sphere_intersect(
            vec3<f32>(0.0, planet_radius + altitude, 0.0),
            sun_dir,
            planet_radius
        );
        // 如果阳光被地球遮挡了，那么就不显示太阳 (透射率为 0)
        if d_planet.y > 0.0 {
            return vec3<f32>(0.0);
        }
        let sun_cos_zenith = sun_dir.y;
        
        let trans_uv = transmittance_lut_uv(altitude, sun_cos_zenith);
        let transmittance = textureSampleLevel(transmittance_lut, lut_sampler, trans_uv, 0.0).rgb;
        
        // 只有未被地球遮挡时才显示太阳 (利用 transmittance 自然归零)
        return transmittance * (t * params.sun_intensity);
    }
    
    return vec3<f32>(0.0);
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let face = id.z;
    let dims = textureDimensions(dest);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let dir = normalize(get_cube_direction(face, uv));

    // Sample sky-view LUT
    let sky_uv = direction_to_sky_view_uv(dir);
    var color = textureSampleLevel(sky_view_lut, lut_sampler, sky_uv, 0.0).rgb;

    // Add sun disk contribution
    color += sun_disk(dir, params.sun_direction, params.sun_disk_size);

    // Apply exposure
    color *= params.exposure;

    let safe_color = clamp(color, vec3<f32>(0.0), vec3<f32>(65000.0));
    textureStore(dest, vec2<u32>(id.xy), face, vec4<f32>(safe_color, 1.0));
}
