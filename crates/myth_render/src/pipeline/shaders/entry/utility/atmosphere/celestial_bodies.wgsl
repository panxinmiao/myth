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
    let mask = disk_mask(dir, u_bake_params.sun_direction, dynamic_sun_size, 0.0001);
    if mask <= 0.0 {
        return vec3<f32>(0.0);
    }

    let transmittance = sample_direction_transmittance(u_bake_params.sun_direction);
    let visual_sun_intensity = u_bake_params.sun_intensity * 200.0;
    return transmittance * (mask * visual_sun_intensity);
}

fn moon_disk(dir: vec3<f32>, view_transmittance: vec3<f32>) -> vec3<f32> {
    let mask = disk_mask(
        dir,
        u_bake_params.moon_direction,
        u_bake_params.moon_disk_size,
        0.00015,
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

fn procedural_star_layer(dir: vec3<f32>, time: f32) -> vec3<f32> {
    let grid = dir * 400.0;
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
    let twinkle = 0.82 + 0.18 * sin(time * twinkle_speed + twinkle_phase);
    let tint = mix(
        vec3<f32>(0.72, 0.80, 1.0),
        vec3<f32>(1.0, 0.95, 0.86),
        hash13(cell + vec3<f32>(83.0, 41.0, 97.0))
    );
    let star = pow(max(radial, 0.0), 8.0) * (4.0 + 24.0 * brightness) * twinkle;
    return tint * star * 3.0;
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
) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    color += sun_disk(dir);

    let night_factor = 1.0 - smoothstep(-0.12, 0.04, u_bake_params.sun_direction.y);
    let rotated_star_dir = rotate_about_axis(
        dir,
        u_bake_params.star_axis,
        u_bake_params.star_rotation,
    );

    var night_color = procedural_star_layer(rotated_star_dir, star_time);
    night_color += sample_starbox(rotated_star_dir) * u_bake_params.star_intensity;
    color += night_color * view_transmittance * night_factor;
    color += moon_disk(dir, view_transmittance);

    return color;
}