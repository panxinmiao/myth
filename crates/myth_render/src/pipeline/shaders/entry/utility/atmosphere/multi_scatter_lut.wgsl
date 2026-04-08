// ============================================================================
// Multi-Scattering LUT Compute Shader
// ============================================================================
//
// Computes the isotropic multi-scattering contribution using the transmittance
// LUT. Accounts for energy that bounces multiple times in the atmosphere.
//
// Output: 2D storage texture (32×32, Rgba16Float)
//   UV.x → sun zenith cos angle
//   UV.y → altitude (linear mapping)
//   RGB  → multi-scattering luminance factor
//
// NOTE: This file is prepended with atmosphere_common.wgsl at compile time.
//
// Dispatch: (32/8, 32/8, 1) = (4, 4, 1) workgroups

{$ include "entry/utility/atmosphere/atmosphere_common" $}

@group(0) @binding(1)
var transmittance_tex: texture_2d<f32>;

@group(0) @binding(2)
var lut_sampler: sampler;

@group(0) @binding(3)
var dest: texture_storage_2d<rgba16float, write>;

const MS_SAMPLE_COUNT: u32 = 20u;
const SPHERE_SAMPLES: u32 = 64u;

fn sample_transmittance_lut(altitude: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_lut_uv(altitude, cos_zenith);
    return textureSampleLevel(transmittance_tex, lut_sampler, uv, 0.0).rgb;
}

fn integrate_scattered_luminance(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
) -> vec3<f32> {
    let atmo_isect = ray_sphere_intersect(ray_origin, ray_dir, atmo.atmosphere_radius);
    var t_max = atmo_isect.y;
    if t_max < 0.0 {
        return vec3<f32>(0.0);
    }

    let ground_isect = ray_sphere_intersect(ray_origin, ray_dir, atmo.planet_radius);
    if ground_isect.x > 0.0 {
        t_max = ground_isect.x;
    }

    let dt = t_max / f32(MS_SAMPLE_COUNT);
    var luminance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    let uniform_phase = 1.0 / (4.0 * PI);

    for (var i = 0u; i < MS_SAMPLE_COUNT; i++) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin + ray_dir * t;
        let h = length(pos) - atmo.planet_radius;

        let medium = sample_medium(h);
        let sample_optical = exp(-medium.extinction * dt);

        let sun_cos = dot(normalize(pos), sun_dir);
        let sun_transmittance = sample_transmittance_lut(h, sun_cos);
        let scattering = medium.rayleigh_scattering + vec3<f32>(medium.mie_scattering);
        let scatter_luminance = sun_transmittance * scattering * uniform_phase;
        let scatter_integral = (scatter_luminance - scatter_luminance * sample_optical) / max(medium.extinction, vec3<f32>(1e-8));
        luminance += scatter_integral * throughput;
        throughput *= sample_optical;
    }

    return luminance;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dest);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);

    let sun_cos_theta = uv.x * 2.0 - 1.0;
    let sun_dir = vec3<f32>(0.0, sun_cos_theta, sqrt(max(0.0, 1.0 - sun_cos_theta * sun_cos_theta)));

    let altitude = uv.y * (atmo.atmosphere_radius - atmo.planet_radius);
    let r = atmo.planet_radius + altitude;
    let ray_origin = vec3<f32>(0.0, r, 0.0);

    var luminance_sum = vec3<f32>(0.0);

    for (var i = 0u; i < SPHERE_SAMPLES; i++) {
        let fi = f32(i);
        let golden_ratio = (1.0 + sqrt(5.0)) / 2.0;

        let cos_theta = 1.0 - 2.0 * (fi + 0.5) / f32(SPHERE_SAMPLES);
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        let phi = 2.0 * PI * fract(fi / golden_ratio);

        let ray_dir = vec3<f32>(
            sin_theta * cos(phi),
            cos_theta,
            sin_theta * sin(phi)
        );

        luminance_sum += integrate_scattered_luminance(ray_origin, ray_dir, sun_dir);
    }

    // Geometric series: L_ms = L_2nd / (1 - Psi)
    let ms = luminance_sum / f32(SPHERE_SAMPLES);
    let result = ms / max(vec3<f32>(1.0) - ms, vec3<f32>(1e-8));

    textureStore(dest, vec2<u32>(id.xy), vec4<f32>(result, 1.0));
}
