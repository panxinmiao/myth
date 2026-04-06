{$ include 'core/full_screen_vertex' $}

{{ struct_definitions }}
{{ binding_code }}

@group(1) @binding(0) var t_depth: texture_depth_2d;
@group(1) @binding(1) var t_normal: texture_2d<f32>;
@group(1) @binding(2) var t_noise: texture_2d<f32>;
@group(1) @binding(3) var s_linear: sampler;
@group(1) @binding(4) var s_noise: sampler;
@group(1) @binding(5) var s_point: sampler;

@group(2) @binding(0) var<uniform> u_ssao: SsaoUniforms;

fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let ndc = vec4<f32>(ndc_x, ndc_y, depth, 1.0);

    let view_pos = u_render_state.projection_inverse * ndc;

    var w = view_pos.w;
    if (abs(w) < 1e-6) {
        w = 1e-6 * sign(w + 1e-8); 
    }
    return view_pos.xyz / w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let depth = textureSampleLevel(t_depth, s_point, uv, 0u);
    if (depth <= 0.0) { 
        return vec4<f32>(1.0);
    }

    let view_pos = reconstruct_view_position(uv, depth);

    let view_dir = normalize(-view_pos);

    let packed_normal = textureSampleLevel(t_normal, s_linear, uv, 0.0);
    if (packed_normal.a < 0.5) { 
        return vec4<f32>(1.0);
    }
    
    var n_raw = packed_normal.xyz * 2.0 - 1.0;
    if (dot(n_raw, n_raw) < 0.0001) {
        n_raw = vec3<f32>(0.0, 0.0, 1.0);
    }
    var view_normal = normalize(n_raw);

    let ndotv = dot(view_normal, view_dir);
    if (ndotv < 0.0) {
        view_normal = normalize(view_normal - view_dir * ndotv); 
    }

    let safe_ndotv = max(dot(view_normal, view_dir), 0.05);

    let random_vec = normalize(
        textureSampleLevel(t_noise, s_noise, uv * u_ssao.noise_scale, 0.0).xyz * 2.0 - 1.0
    );
    
    var tangent_unnormalized = random_vec - view_normal * dot(random_vec, view_normal);
    if (dot(tangent_unnormalized, tangent_unnormalized) < 0.0001) {
        tangent_unnormalized = cross(view_normal, vec3<f32>(1.0, 0.0, 0.0));
        if (dot(tangent_unnormalized, tangent_unnormalized) < 0.0001) {
            tangent_unnormalized = cross(view_normal, vec3<f32>(0.0, 1.0, 0.0));
        }
    }
    let tangent = normalize(tangent_unnormalized);
    let bitangent = cross(view_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, view_normal);

    let dynamic_bias = u_ssao.bias * (1.0 + (1.0 - safe_ndotv) * 8.0);

    let origin_pos = view_pos + view_normal * (u_ssao.radius * 0.05 + dynamic_bias * 0.2) + view_dir * (dynamic_bias * 0.1);

    let sample_count = u_ssao.sample_count;

    var occlusion_sum: f32 = 0.0;

    for (var i: u32 = 0u; i < sample_count; i++) {
        let sample_dir = tbn * u_ssao.samples[i].xyz;
        let sample_pos = origin_pos + sample_dir * u_ssao.radius;

        var offset_clip = u_render_state.projection_matrix * vec4<f32>(sample_pos, 1.0);
        
        if (offset_clip.w <= 0.0001) {
            continue;
        }

        offset_clip /= offset_clip.w;
        let offset_uv = vec2<f32>(
            offset_clip.x * 0.5 + 0.5,
            0.5 - offset_clip.y * 0.5
        );

        if (offset_uv.x < 0.0 || offset_uv.x > 1.0 || offset_uv.y < 0.0 || offset_uv.y > 1.0) {
            continue; 
        }

        let real_depth = textureSampleLevel(t_depth, s_point, offset_uv, 0u);

        if (real_depth > 0.00001) {
            let real_view_pos = reconstruct_view_position(offset_uv, real_depth);

            let distance_diff = abs(origin_pos.z - real_view_pos.z);
            let range_check = smoothstep(1.0, 0.0, distance_diff / u_ssao.radius);

            if (real_view_pos.z >= sample_pos.z + dynamic_bias) {
                occlusion_sum += range_check;
            }
        }

    }

    var occlusion = 1.0 - (occlusion_sum / f32(u_ssao.sample_count));

    occlusion = max(occlusion, 0.0);
    let ao = pow(occlusion, u_ssao.intensity);

    return vec4<f32>(ao, ao, ao, 1.0);
}