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
var t_sky_view: texture_2d<f32>;

@group(0) @binding(1)
var s_skybox: sampler;

@group(0) @binding(2)
var<uniform> u_bake_params: BakeParams;

@group(0) @binding(3)
var dest: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(4)
var t_transmittance: texture_2d<f32>;

$$ if CELESTIAL_STARBOX_EQUIRECT
@group(0) @binding(5) var t_starbox_2d: texture_2d<f32>;
$$ endif

$$ if CELESTIAL_STARBOX_CUBE
@group(0) @binding(5) var t_starbox_cube: texture_cube<f32>;
$$ endif

{$ include "entry/utility/atmosphere/celestial_bodies" $}

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
    var color = textureSampleLevel(t_sky_view, s_skybox, sky_uv, 0.0).rgb;
    let view_transmittance = sample_direction_transmittance(dir);
    color += compute_celestial_lighting(dir, view_transmittance, 0.0);

    // Apply exposure
    color *= u_bake_params.exposure;

    let safe_color = clamp(color, vec3<f32>(0.0), vec3<f32>(65000.0));
    textureStore(dest, vec2<u32>(id.xy), face, vec4<f32>(safe_color, 1.0));
}
