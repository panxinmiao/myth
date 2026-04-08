// ============================================================================
// Hillaire 2020 Atmosphere — Shared Functions
// ============================================================================
//
// This file is concatenated with each atmosphere compute shader at compile time
// via Rust's include_str!. It is NOT a standalone shader module.
//
// Reference: S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere
//            Rendering Technique", EGSR 2020.

const PI: f32 = 3.14159265358979323846;

struct AtmosphereParams {
    rayleigh_scattering: vec3<f32>,
    rayleigh_scale_height: f32,

    mie_scattering: f32,
    mie_absorption: f32,
    mie_scale_height: f32,
    mie_anisotropy: f32,

    ozone_absorption: vec3<f32>,
    _pad0: f32,

    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    sun_cos_angle: f32,

    sun_direction: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> atmo: AtmosphereParams;

fn ray_sphere_intersect(o: vec3<f32>, d: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(d, d);
    let b = 2.0 * dot(d, o);
    let c = dot(o, o) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let sq = sqrt(discriminant);
    return vec2<f32>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
}

struct ScatteringCoefficients {
    rayleigh_scattering: vec3<f32>,
    mie_scattering: f32,
    extinction: vec3<f32>,
};

fn sample_medium(altitude: f32) -> ScatteringCoefficients {
    let mie_density = exp(-altitude / atmo.mie_scale_height);
    let rayleigh_density = exp(-altitude / atmo.rayleigh_scale_height);

    let mie_scattering_val = atmo.mie_scattering * mie_density;
    let mie_absorption_val = atmo.mie_absorption * mie_density;
    let rayleigh_scattering_val = atmo.rayleigh_scattering * rayleigh_density;

    let ozone_center = 25000.0;
    let ozone_width = 15000.0;
    let ozone_density = max(0.0, 1.0 - abs(altitude - ozone_center) / ozone_width);
    let ozone_absorption_val = atmo.ozone_absorption * ozone_density;

    var result: ScatteringCoefficients;
    result.rayleigh_scattering = rayleigh_scattering_val;
    result.mie_scattering = mie_scattering_val;
    result.extinction = rayleigh_scattering_val
        + vec3<f32>(mie_scattering_val + mie_absorption_val)
        + ozone_absorption_val;
    return result;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

fn transmittance_lut_uv(altitude: f32, cos_zenith: f32) -> vec2<f32> {
    let H = sqrt(max(0.0,
        atmo.atmosphere_radius * atmo.atmosphere_radius
        - atmo.planet_radius * atmo.planet_radius));
    let rho = sqrt(max(0.0,
        (atmo.planet_radius + altitude) * (atmo.planet_radius + altitude)
        - atmo.planet_radius * atmo.planet_radius));

    let d = ray_sphere_intersect(
        vec3<f32>(0.0, atmo.planet_radius + altitude, 0.0),
        vec3<f32>(0.0, cos_zenith, sqrt(max(0.0, 1.0 - cos_zenith * cos_zenith))),
        atmo.atmosphere_radius
    ).y;

    let d_min = atmo.atmosphere_radius - atmo.planet_radius - altitude;
    let d_max = rho + H;
    let x_mu = (d - d_min) / (d_max - d_min);
    let x_r = rho / H;
    return vec2<f32>(x_mu, x_r);
}

fn transmittance_lut_inv(uv: vec2<f32>) -> vec2<f32> {
    let H = sqrt(max(0.0,
        atmo.atmosphere_radius * atmo.atmosphere_radius
        - atmo.planet_radius * atmo.planet_radius));
    let rho = H * uv.y;
    let r = sqrt(rho * rho + atmo.planet_radius * atmo.planet_radius);
    let altitude = r - atmo.planet_radius;

    let d_min = atmo.atmosphere_radius - r;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);

    let cos_zenith = clamp(
        (H * H - rho * rho - d * d) / (2.0 * r * d),
        -1.0, 1.0
    );
    return vec2<f32>(altitude, cos_zenith);
}
