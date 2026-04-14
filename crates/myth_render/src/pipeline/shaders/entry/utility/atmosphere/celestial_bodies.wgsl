// ============================================================================
// Shared Celestial Bodies
// ============================================================================

{$ include "entry/utility/noise" $}

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

fn disk_mask(
    view_dir: vec3<f32>,
    body_dir: vec3<f32>,
    angular_size_deg: f32,
    smoothing: f32,
) -> f32 {
    let cos_angle = dot(view_dir, body_dir);
    let angular_radius = (angular_size_deg * 0.5) * PI / 180.0;
    let disk_cos = cos(angular_radius);
    return smoothstep(disk_cos - smoothing, disk_cos + smoothing, cos_angle);
}

$$ if USE_MOON_TEXTURE
fn sample_moon_albedo(moon_uv: vec2<f32>) -> vec3<f32> {
$$ if SKYBOX_PROCEDURAL
    // return textureSample(t_moon_albedo, s_skybox, moon_uv).rgb;
    return textureSampleLevel(t_moon_albedo, s_skybox, moon_uv, 0.0).rgb;
$$ else
    return textureSampleLevel(t_moon_albedo, s_skybox, moon_uv, 0.0).rgb;
$$ endif
}
$$ endif

fn _sun_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let dynamic_sun_size = u_bake_params.sun_disk_size
        * mix(3.0, 1.0, smoothstep(0.0, 0.5, u_bake_params.sun_direction.y));

    let mask = disk_mask(dir, u_bake_params.sun_direction, dynamic_sun_size, 0.00005);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let horizon_fade = smoothstep(-0.02, 0.15, u_bake_params.sun_direction.y);
    let visual_sun_intensity = mix(1.2, u_bake_params.sun_intensity * 200.0, horizon_fade);

    return view_transmittance * (mask * visual_sun_intensity);
}

// Another disk function
// todo: which one is better?
fn sun_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let sun_size = u_bake_params.sun_disk_size * 1.05; 

    let horizon_fade = smoothstep(-0.02, 0.5, u_bake_params.sun_direction.y);

    let dynamic_smooth = mix(0.00015, 0.00005, horizon_fade);
    // edge softening
    let mask = disk_mask(dir, u_bake_params.sun_direction, sun_size, dynamic_smooth);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let visual_sun_intensity = u_bake_params.sun_intensity * 200.0;

    return view_transmittance * (mask * visual_sun_intensity);
}

fn _moon_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let mask = disk_mask(
        dir,
        u_bake_params.moon_direction,
        u_bake_params.moon_disk_size * 1.05,
        0.00005,
    );
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let moon_color = vec3<f32>(0.92, 0.94, 1.0);
    return moon_color * view_transmittance * (mask * u_bake_params.moon_intensity * 10.0);
}

fn moon_disk(dir: vec3<f32>, view_transmittance: vec3<f32>, night_factor: f32) -> vec3<f32> {
    let cos_angle = dot(dir, u_bake_params.moon_direction);
    let angular_radius = (u_bake_params.moon_disk_size * 3.0 * 0.5) * PI / 180.0;
    let cos_alpha = cos(angular_radius);

    if (cos_angle < cos_alpha - 0.0001) {
        return vec3<f32>(0.0);
    }

    let sq = max(0.0, cos_angle * cos_angle - cos_alpha * cos_alpha);
    let t_prime = cos_angle - sqrt(sq);
    let sphere_normal = normalize(dir * t_prime - u_bake_params.moon_direction);

    let earthshine = 0.0015 * night_factor;
    let phase_shading = max(0.0, dot(sphere_normal, u_bake_params.sun_direction)) + earthshine;

$$ if USE_MOON_TEXTURE
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(u_bake_params.moon_direction.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let moon_right = normalize(cross(up, u_bake_params.moon_direction));
    let moon_up = cross(u_bake_params.moon_direction, moon_right);
    let local_x = dot(sphere_normal, moon_right);
    let local_y = dot(sphere_normal, moon_up);
    let moon_uv = clamp(
        vec2<f32>(local_x, local_y) * 0.498 + 0.5, // 0.498 to leave a tiny border to avoid sampling artifacts
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    let surface_albedo = sample_moon_albedo(moon_uv);
$$ else
    let dark_color = vec3<f32>(0.08, 0.08, 0.10);
    let bright_color = vec3<f32>(0.92, 0.94, 0.98);
    let moon_noise = fbm3D(sphere_normal * 4.0);
    let surface_albedo = mix(dark_color, bright_color, moon_noise);
$$ endif

    let mask = smoothstep(cos_alpha - 0.00005, cos_alpha + 0.00005, cos_angle);
    
    let final_albedo = pow(surface_albedo, vec3<f32>(1.5));
    return final_albedo * phase_shading * view_transmittance * (mask * u_bake_params.moon_intensity * 20.0);
}

// Energy-Conserving Analytical Anti-Aliasing
// todo: We should skip the AA in the skybox area in TAA and FXAA.
// Skybox itself does not require AA, while TAA or FXAA will break the energy conservation by blurring the stars.
fn procedural_star_layer(dir: vec3<f32>, time: f32) -> vec3<f32> {

    let grid = dir * 400.0;
    let cell = floor(grid);
    let density = hash13(cell);

     $$ if SKYBOX_PROCEDURAL
    // Keep derivative operations in uniform control flow for WebGPU validation.
    let pixel_size = length(fwidth(dir)) * 400.0 * 0.35;
    $$ else
    let pixel_size = 0.25;
    $$ endif
    
    if density < 0.997 {
        return vec3<f32>(0.0);
    }

    let local = fract(grid) - vec3<f32>(0.5);
    let dist = length(local);
    
    // real physical size of the star's core.
    let physical_size = mix(0.015, 0.06, hash13(cell + vec3<f32>(11.0, 17.0, 23.0))); 
    
    // ==========================================
    // Dynamic Anti-Aliasing Core:
    // Take the maximum of the actual pixel_size and physical_size.
    // ==========================================

    let render_size = max(physical_size, pixel_size);
    
    // Energy-Conserving Formula (Square of Area Ratio)
    let energy_scale = (physical_size * physical_size) / (render_size * render_size);
    
    let raw_core = clamp(1.0 - dist / render_size, 0.0, 1.0);
    let core = raw_core * raw_core; 
    
    let star_shape = core * energy_scale;

    let brightness = pow((density - 0.985) / 0.015, 2.0);
    let twinkle_phase = hash13(cell + vec3<f32>(37.0, 19.0, 53.0)) * TAU;
    let twinkle_speed = mix(0.1, 0.3, hash13(cell + vec3<f32>(59.0, 29.0, 71.0)));

    let t = time * twinkle_speed + twinkle_phase;
    let wave1 = sin(t);
    let wave2 = sin(t * 1.43);
    let chaotic_wave = (wave1 + wave2) * 0.5;

    let twinkle = 0.925 + 0.075 * chaotic_wave;
    let tint = mix(
        vec3<f32>(0.72, 0.80, 1.0),
        vec3<f32>(1.0, 0.95, 0.86),
        hash13(cell + vec3<f32>(83.0, 41.0, 97.0))
    );
    
    // Final star color with twinkle and energy conservation applied
    let star = star_shape * (100.0 + 400.0 * brightness) * twinkle;
    
    return tint * star;
}

fn sample_starbox(dir: vec3<f32>) -> vec3<f32> {
$$ if CELESTIAL_STARBOX_EQUIRECT
    let star_uv = equirectangular_uv(dir);
    let wrapped_uv = vec2<f32>(fract(star_uv.x), clamp(star_uv.y, 0.0, 1.0));
    var color = textureSampleLevel(t_starbox_2d, s_skybox, wrapped_uv, 0.0).rgb;
    color = max(color - vec3<f32>(0.05), vec3<f32>(0.0));
    return pow(color, vec3<f32>(1.5));
$$ elif CELESTIAL_STARBOX_CUBE
    var color = textureSampleLevel(t_starbox_cube, s_skybox, dir, 0.0).rgb;
    color = max(color - vec3<f32>(0.05), vec3<f32>(0.0));
    return pow(color, vec3<f32>(1.5));
$$ else
    return vec3<f32>(0.0);
$$ endif
}

fn compute_celestial_lighting(
    dir: vec3<f32>,
    view_transmittance: vec3<f32>,
    star_time: f32,
) -> vec3<f32> {

    var color = vec3<f32>(0.0);

    // skip sun disk in PMREM to avoid double counting
    $$ if SKYBOX_PROCEDURAL
    // sun
    color += sun_disk(dir, view_transmittance);
    $$ endif


    // star sky
    let night_factor = 1.0 - smoothstep(-0.12, 0.04, u_bake_params.sun_direction.y);
    let rotated_star_dir = rotate_about_axis(
        dir,
        u_bake_params.star_axis,
        u_bake_params.star_rotation,
    );

    var night_color = procedural_star_layer(rotated_star_dir, star_time);
    night_color += sample_starbox(rotated_star_dir) * u_bake_params.star_intensity;
    color += night_color * view_transmittance * night_factor;

    // moon
    color += moon_disk(dir, view_transmittance, night_factor);

    return color;
}