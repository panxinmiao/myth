// === Skybox / Background Pass Shader ===
//
// Renders a fullscreen triangle at the far depth plane (Reverse-Z: Z = 0.0).
// Uses ray reconstruction from the inverse view-projection matrix to compute
// view-space direction for texture sampling.
//
// Pipeline variants (selected via ShaderDefines):
//   SKYBOX_GRADIENT      - Vertical color gradient (no texture)
//   SKYBOX_CUBE          - Cubemap sampling
//   SKYBOX_EQUIRECT      - Equirectangular (lat-long) 2D texture sampling
//   SKYBOX_PLANAR        - Screen-space planar 2D texture sampling

{$ include 'core/full_screen_vertex' $}
{$ include "entry/utility/atmosphere/atmosphere_math" $}

// Auto-generated struct definition for SkyboxParams
{{ struct_definitions }}

// Auto-injected global bind group bindings (Group 0: camera, environment, etc.)
{{ binding_code }}

// --- Skybox-specific bindings (Group 1) ---
$$ if SKYBOX_PROCEDURAL
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

@group(1) @binding(0) var<uniform> u_bake_params: BakeParams;
@group(1) @binding(1) var t_sky_view: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
@group(1) @binding(3) var t_transmittance: texture_2d<f32>;
$$ if SKYBOX_PROCEDURAL_STAR_EQUIRECT
@group(1) @binding(4) var t_starbox_2d: texture_2d<f32>;
$$ endif
$$ if SKYBOX_PROCEDURAL_STAR_CUBE
@group(1) @binding(4) var t_starbox_cube: texture_cube<f32>;
$$ endif
$$ else
@group(1) @binding(0) var<uniform> u_params: SkyboxParams;
$$ endif

$$ if SKYBOX_CUBE
@group(1) @binding(1) var t_skybox_cube: texture_cube<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_EQUIRECT
@group(1) @binding(1) var t_skybox_2d: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_PLANAR
@group(1) @binding(1) var t_skybox_2d: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_PROCEDURAL
fn sample_direction_transmittance(direction: vec3<f32>) -> vec3<f32> {
    let altitude = 1.0;
    let origin = vec3<f32>(0.0, u_bake_params.planet_radius + altitude, 0.0);
    let planet_hit = ray_sphere_intersect(origin, direction, u_bake_params.planet_radius);
    if planet_hit.y > 0.0 {
        return vec3<f32>(0.0);
    }

    let trans_uv = transmittance_lut_uv(
        altitude,
        direction.y,
        u_bake_params.planet_radius,
        u_bake_params.atmosphere_radius,
    );
    return textureSampleLevel(t_transmittance, s_skybox, trans_uv, 0.0).rgb;
}

fn disk_mask(view_dir: vec3<f32>, body_dir: vec3<f32>, angular_size_deg: f32, smoothing: f32) -> f32 {
    let cos_angle = dot(view_dir, body_dir);
    let angular_radius = (angular_size_deg * 0.5) * PI / 180.0;
    let disk_cos = cos(angular_radius);
    return smoothstep(disk_cos - smoothing, disk_cos + smoothing, cos_angle);
}

fn sun_disk(dir: vec3<f32>) -> vec3<f32> {
    let mask = disk_mask(dir, u_bake_params.sun_direction, u_bake_params.sun_disk_size, 0.00002);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let transmittance = sample_direction_transmittance(u_bake_params.sun_direction);
    return transmittance * (mask * u_bake_params.sun_intensity);
}

fn moon_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let mask = disk_mask(dir, u_bake_params.moon_direction, u_bake_params.moon_disk_size, 0.00015);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let moon_color = vec3<f32>(0.92, 0.94, 1.0);
    return moon_color * view_transmittance * (mask * u_bake_params.moon_intensity);
}

fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn procedural_star_layer(dir: vec3<f32>) -> vec3<f32> {
    let grid = dir * 720.0;
    let cell = floor(grid);
    let density = hash13(cell);
    if density < 0.997 {
        return vec3<f32>(0.0);
    }

    let local = fract(grid) - vec3<f32>(0.5);
    let size = mix(0.22, 0.08, hash13(cell + vec3<f32>(11.0, 17.0, 23.0)));
    let radial = 1.0 - clamp(length(local) / size, 0.0, 1.0);
    let brightness = pow((density - 0.997) / 0.003, 3.0);
    let twinkle_phase = hash13(cell + vec3<f32>(37.0, 19.0, 53.0)) * TAU;
    let twinkle_speed = mix(0.8, 2.4, hash13(cell + vec3<f32>(59.0, 29.0, 71.0)));
    let twinkle = 0.82 + 0.18 * sin(u_render_state.time * twinkle_speed + twinkle_phase);
    let tint = mix(
        vec3<f32>(0.72, 0.80, 1.0),
        vec3<f32>(1.0, 0.95, 0.86),
        hash13(cell + vec3<f32>(83.0, 41.0, 97.0))
    );
    let star = pow(max(radial, 0.0), 16.0) * (4.0 + 24.0 * brightness) * twinkle;
    return tint * star;
}

fn sample_starbox(dir: vec3<f32>) -> vec3<f32> {
$$ if SKYBOX_PROCEDURAL_STAR_EQUIRECT
    let star_uv = equirectangular_uv(dir);
    let wrapped_uv = vec2<f32>(fract(star_uv.x), clamp(star_uv.y, 0.0, 1.0));
    return textureSampleLevel(t_starbox_2d, s_skybox, wrapped_uv, 0.0).rgb;
$$ elif SKYBOX_PROCEDURAL_STAR_CUBE
    return textureSampleLevel(t_starbox_cube, s_skybox, dir, 0.0).rgb;
$$ else
    return vec3<f32>(0.0);
$$ endif
}
$$ endif

// --- Fragment Shader ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;

    // --- 1.Pixel-Perfect Ray Reconstruction ---
    let ndc = in.uv * 2.0 - 1.0;

    // Use Near Plane (Z=1.0) to get the direction vector without needing the actual depth value
    let clip_pos = vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    
    // Transform from clip space to world space (using global RenderState uniforms)
    let world_pos_h = u_render_state.view_projection_inverse * clip_pos;
    let world_pos = world_pos_h.xyz / world_pos_h.w;
    
    // Compute world-space ray direction
    let world_dir = normalize(world_pos - u_render_state.camera_position);

$$ if SKYBOX_PROCEDURAL
    let sky_uv = direction_to_sky_view_uv(world_dir);
    var procedural_color = textureSampleLevel(t_sky_view, s_skybox, sky_uv, 0.0).rgb;
    let view_transmittance = sample_direction_transmittance(world_dir);
    let night_factor = 1.0 - smoothstep(-0.12, 0.04, u_bake_params.sun_direction.y);
    let rotated_star_dir = rotate_about_axis(
        world_dir,
        u_bake_params.star_axis,
        u_bake_params.star_rotation,
    );
    var night_color = procedural_star_layer(rotated_star_dir);
    night_color += sample_starbox(rotated_star_dir) * u_bake_params.star_intensity;
    night_color *= view_transmittance * night_factor;
    night_color += moon_disk(world_dir, view_transmittance);
    procedural_color += sun_disk(world_dir);
    procedural_color += night_color;
    procedural_color *= u_bake_params.exposure;
    procedural_color = clamp(procedural_color, vec3<f32>(0.0), vec3<f32>(65000.0));
    color = vec4<f32>(procedural_color, 1.0);
$$ endif

$$ if SKYBOX_GRADIENT
    // Smooth vertical blend based on Y component of view direction
    let t = smoothstep(-0.5, 0.5, world_dir.y);
    color = mix(u_params.color_bottom, u_params.color_top, t);

    // Add Dithering to reduce banding in gradients, especially at low precision (e.g. 8-bit displays)
    let noise = fract(sin(dot(in.position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) / 255.0;
$$ endif

$$ if SKYBOX_CUBE or SKYBOX_EQUIRECT
    // Apply Y-axis rotation
    let s = sin(u_params.rotation);
    let c = cos(u_params.rotation);
    let rot_dir = vec3<f32>(
        -(world_dir.x * c - world_dir.z * s), // Negate X to convert from left-handed to right-handed coordinates for cubemap sampling
        world_dir.y,
        world_dir.x * s + world_dir.z * c
    );
$$ endif

$$ if SKYBOX_CUBE
    color = textureSample(t_skybox_cube, s_skybox, rot_dir);
$$ endif

$$ if SKYBOX_EQUIRECT
    // Convert direction to equirectangular UV
    let tex_uv = equirectangular_uv(rot_dir);

    // HDR Color
    color = textureSample(t_skybox_2d, s_skybox, tex_uv);

    color = clamp(color, vec4<f32>(0.0), vec4<f32>(65000.0));
$$ endif

$$ if SKYBOX_PLANAR
    // --- Planar mode (screen-space mapping) ---
    color = textureSample(t_skybox_2d, s_skybox, in.uv);
$$ endif

$$ if SKYBOX_PROCEDURAL
    return color;
$$ else
    return vec4<f32>(color.rgb * u_params.intensity, color.a);
$$ endif
}
