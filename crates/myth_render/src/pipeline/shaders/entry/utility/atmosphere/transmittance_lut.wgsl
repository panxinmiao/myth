// ============================================================================
// Transmittance LUT Compute Shader
// ============================================================================
//
// Precomputes optical depth (transmittance) for rays starting at various
// altitudes and zenith angles through the atmosphere.
//
// Output: 2D storage texture (256×64, Rgba16Float)
//   UV.x → cos(zenith angle) via non-linear mapping
//   UV.y → altitude via non-linear mapping
//   RGB  → transmittance T(P → atmosphere boundary)
//
// NOTE: This file is prepended with atmosphere_common.wgsl at compile time.
//       AtmosphereParams, ray_sphere_intersect, sample_medium, etc. are
//       available from the common section.
//
// Dispatch: (256/8, 64/8, 1) = (32, 8, 1) workgroups

{$ include "entry/utility/atmosphere/atmosphere_common" $}

@group(0) @binding(1)
var dest: texture_storage_2d<rgba16float, write>;

const TRANSMITTANCE_STEPS: u32 = 40u;

fn compute_transmittance(altitude: f32, cos_zenith: f32) -> vec3<f32> {
    let r = atmo.planet_radius + altitude;
    let ray_origin = vec3<f32>(0.0, r, 0.0);
    let ray_dir = vec3<f32>(
        sqrt(max(0.0, 1.0 - cos_zenith * cos_zenith)),
        cos_zenith,
        0.0
    );

    let atmo_isect = ray_sphere_intersect(ray_origin, ray_dir, atmo.atmosphere_radius);
    let t_max = atmo_isect.y;
    if t_max < 0.0 {
        return vec3<f32>(1.0);
    }

    let ground_isect = ray_sphere_intersect(ray_origin, ray_dir, atmo.planet_radius);
    let ray_length = select(t_max, max(0.0, ground_isect.x), ground_isect.x > 0.0);

    let dt = ray_length / f32(TRANSMITTANCE_STEPS);
    var optical_depth = vec3<f32>(0.0);

    for (var i = 0u; i < TRANSMITTANCE_STEPS; i++) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin + ray_dir * t;
        let h = length(pos) - atmo.planet_radius;
        let medium = sample_medium(h);
        optical_depth += medium.extinction * dt;
    }

    return exp(-optical_depth);
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dest);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let params = transmittance_lut_inv(uv);
    let altitude = params.x;
    let cos_zenith = params.y;

    let transmittance = compute_transmittance(altitude, cos_zenith);
    textureStore(dest, vec2<u32>(id.xy), vec4<f32>(transmittance, 1.0));
}
