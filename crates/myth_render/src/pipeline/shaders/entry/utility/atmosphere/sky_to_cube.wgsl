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

{$ include "entry/utility/atmosphere/atmosphere_math" $}

struct BakeParams {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    moon_direction: vec3<f32>,
    moon_intensity: f32,
    star_axis: vec3<f32>,
    sun_disk_size: f32,
    moon_disk_size: f32,
    exposure: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    star_intensity: f32,
    star_rotation: f32,
    _pad2: vec2<f32>,
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

/// Approximate sun disk rendering.
fn sun_disk(dir: vec3<f32>, sun_dir: vec3<f32>, sun_size_deg: f32) -> vec3<f32> {
    let cos_angle = dot(dir, sun_dir);
    // The input is the diameter, so we divide by 2 to get the radius
    let sun_angular_radius = (sun_size_deg * 0.5) * PI / 180.0;
    let sun_cos = cos(sun_angular_radius);

    // fix smoothstep edge case: shrink the smoothing range inward to avoid exceeding 1.0
    let smoothing = 0.00002; 
    let t = smoothstep(sun_cos - smoothing, sun_cos + smoothing, cos_angle);
    
    if t > 0.0 {
        // Calculate the transmittance of sunlight through the atmosphere
        // Assume we are observing from 1.0 meters above the ground
        let altitude = 1.0; 
        let d_planet = ray_sphere_intersect(
            vec3<f32>(0.0, params.planet_radius + altitude, 0.0),
            sun_dir,
            params.planet_radius
        );
        // If the sunlight is blocked by the Earth, do not display the sun (transmittance is 0)
        if d_planet.y > 0.0 {
            return vec3<f32>(0.0);
        }
        let sun_cos_zenith = sun_dir.y;
        
        let trans_uv = transmittance_lut_uv(
            altitude,
            sun_cos_zenith,
            params.planet_radius,
            params.atmosphere_radius,
        );
        let transmittance = textureSampleLevel(transmittance_lut, lut_sampler, trans_uv, 0.0).rgb;

        // The sun's apparent brightness is modulated by the atmospheric transmittance, which accounts for the dimming effect of the atmosphere, especially near the horizon.
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
