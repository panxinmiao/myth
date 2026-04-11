// ============================================================================
// Shared Celestial Bodies
// ============================================================================

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

fn sun_disk(dir: vec3<f32>) -> vec3<f32> {
    let dynamic_sun_size = u_bake_params.sun_disk_size
        * mix(2.5, 1.0, smoothstep(0.0, 0.5, u_bake_params.sun_direction.y));

    let mask = disk_mask(dir, u_bake_params.sun_direction, dynamic_sun_size, 0.00005);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let transmittance = sample_direction_transmittance(u_bake_params.sun_direction);
    // let height_fade = smoothstep(-0.05, 0.2, u_bake_params.sun_direction.y);
    // let visual_sun_intensity = u_bake_params.sun_intensity * mix(20.0, 200.0, height_fade);

    let horizon_fade = smoothstep(-0.02, 0.15, u_bake_params.sun_direction.y);
    let visual_sun_intensity = mix(1.2, u_bake_params.sun_intensity * 200.0, horizon_fade);

    return transmittance * (mask * visual_sun_intensity);
}

fn moon_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let mask = disk_mask(
        dir,
        u_bake_params.moon_direction,
        u_bake_params.moon_disk_size,
        0.00005,
    );
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

// Energy-Conserving Analytical Anti-Aliasing
fn procedural_star_layer(dir: vec3<f32>, time: f32, pixel_size: f32) -> vec3<f32> {

    let grid = dir * 400.0;
    let cell = floor(grid);
    let density = hash13(cell);
    
    if density < 0.985 {
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
    return pow(color, vec3<f32>(2.5));
$$ elif CELESTIAL_STARBOX_CUBE
    var color = textureSampleLevel(t_starbox_cube, s_skybox, dir, 0.0).rgb;
    color = max(color - vec3<f32>(0.05), vec3<f32>(0.0));
    return pow(color, vec3<f32>(2.5));
$$ else
    return vec3<f32>(0.0);
$$ endif
}

fn compute_celestial_lighting(
    dir: vec3<f32>,
    view_transmittance: vec3<f32>,
    star_time: f32,
    pixel_size: f32,
) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    color += sun_disk(dir);

    let night_factor = 1.0 - smoothstep(-0.12, 0.04, u_bake_params.sun_direction.y);
    let rotated_star_dir = rotate_about_axis(
        dir,
        u_bake_params.star_axis,
        u_bake_params.star_rotation,
    );

    var night_color = procedural_star_layer(rotated_star_dir, star_time, pixel_size);
    night_color += sample_starbox(rotated_star_dir) * u_bake_params.star_intensity;
    color += night_color * view_transmittance * night_factor;
    color += moon_disk(dir, view_transmittance);

    return color;
}