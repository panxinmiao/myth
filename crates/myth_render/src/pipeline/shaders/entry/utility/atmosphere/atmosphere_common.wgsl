// ============================================================================
// Hillaire 2020 Atmosphere — Shared Functions
// ============================================================================
//
// This file is shared by the atmosphere compute shaders and expects
// atmosphere_math.wgsl to be included first.
//
// Reference: S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere
//            Rendering Technique", EGSR 2020.

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
