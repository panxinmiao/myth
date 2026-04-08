// ============================================================================
// Sky-View LUT Compute Shader
// ============================================================================
//
// Renders the full sky hemisphere into a compact 2D lookup table using
// non-linear longitude/latitude mapping. This LUT is the primary output
// consumed by the skybox pass and the dynamic IBL bake pipeline.
//
// Output: 2D storage texture (192×108, Rgba16Float)
//   UV.x → longitude (azimuth)
//   UV.y → latitude (elevation), non-linear near horizon
//   RGB  → sky luminance (linear HDR)
//
// NOTE: This file is prepended with atmosphere_common.wgsl at compile time.
//
// Dispatch: (192/8, 108/8, 1) = (24, 14, 1) workgroups

{$ include "entry/utility/atmosphere/atmosphere_common" $}

@group(0) @binding(1)
var transmittance_tex: texture_2d<f32>;

@group(0) @binding(2)
var multi_scatter_tex: texture_2d<f32>;

@group(0) @binding(3)
var lut_sampler: sampler;

@group(0) @binding(4)
var dest: texture_storage_2d<rgba16float, write>;

const SKY_VIEW_STEPS: u32 = 30u;

fn sample_transmittance_lut(altitude: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_lut_uv(altitude, cos_zenith);
    return textureSampleLevel(transmittance_tex, lut_sampler, uv, 0.0).rgb;
}

fn sample_multi_scatter(altitude: f32, sun_cos_zenith: f32) -> vec3<f32> {
    let u = sun_cos_zenith * 0.5 + 0.5;
    let v = altitude / (atmo.atmosphere_radius - atmo.planet_radius);
    return textureSampleLevel(multi_scatter_tex, lut_sampler, vec2<f32>(u, v), 0.0).rgb;
}

fn compute_sky_luminance(
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

    let cos_theta = dot(ray_dir, sun_dir);
    let rp = rayleigh_phase(cos_theta);
    let mp = hg_phase(cos_theta, atmo.mie_anisotropy);

    let dt = t_max / f32(SKY_VIEW_STEPS);
    var luminance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);

    for (var i = 0u; i < SKY_VIEW_STEPS; i++) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin + ray_dir * t;
        let h = length(pos) - atmo.planet_radius;
        let up_dir = normalize(pos);

        let medium = sample_medium(h);
        let sample_optical = exp(-medium.extinction * dt);

        let sun_cos = dot(up_dir, sun_dir);
        let sun_transmittance = sample_transmittance_lut(h, sun_cos);

        // Planet shadow check
        let planet_shadow = ray_sphere_intersect(pos, sun_dir, atmo.planet_radius);
        var visible_sun = sun_transmittance;
        if planet_shadow.x > 0.0 {
            visible_sun = vec3<f32>(0.0);
        }

        let rayleigh_scatter = medium.rayleigh_scattering * rp;
        let mie_scatter = vec3<f32>(medium.mie_scattering) * mp;
        let single_scatter = (rayleigh_scatter + mie_scatter) * visible_sun * atmo.sun_intensity;

        let total_scattering = medium.rayleigh_scattering + vec3<f32>(medium.mie_scattering);
        let ms = sample_multi_scatter(h, sun_cos) * total_scattering;

        let scatter_luminance = single_scatter + ms;
        let scatter_integral = (scatter_luminance - scatter_luminance * sample_optical) / max(medium.extinction, vec3<f32>(1e-8));
        luminance += scatter_integral * throughput;
        throughput *= sample_optical;
    }

    return luminance;
}

fn sky_view_uv_to_direction(uv: vec2<f32>) -> vec3<f32> {
    let phi = uv.x * 2.0 * PI - PI;

    let v = uv.y;
    var theta: f32;
    if v < 0.5 {
        let coord = 1.0 - 2.0 * v;
        theta = -(PI * 0.5) * coord * coord;
    } else {
        let coord = 2.0 * v - 1.0;
        theta = (PI * 0.5) * coord * coord;
    }

    let cos_theta = cos(theta);
    let sin_theta = sin(theta);

    return vec3<f32>(
        cos_theta * sin(phi),
        sin_theta,
        cos_theta * cos(phi)
    );
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dest);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let ray_origin = vec3<f32>(0.0, atmo.planet_radius + 1.0, 0.0);
    let ray_dir = sky_view_uv_to_direction(uv);

    let luminance = compute_sky_luminance(ray_origin, ray_dir, atmo.sun_direction);
    let safe_luminance = clamp(luminance, vec3<f32>(0.0), vec3<f32>(65000.0));
    textureStore(dest, vec2<u32>(id.xy), vec4<f32>(safe_luminance, 1.0));
}
