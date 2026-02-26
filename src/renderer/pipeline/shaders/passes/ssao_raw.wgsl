{$ include 'full_screen_vertex.wgsl' $}

{{ struct_definitions }}

// Auto-injected binding code for global resources (camera matrices, etc.)
{{ binding_code }}

// --- Group 1: Depth / Normal / Noise Textures ---
@group(1) @binding(0) var t_depth: texture_depth_2d;
@group(1) @binding(1) var t_normal: texture_2d<f32>;
@group(1) @binding(2) var t_noise: texture_2d<f32>;
@group(1) @binding(3) var s_linear: sampler;
@group(1) @binding(4) var s_noise: sampler;

// --- Group 2: SSAO Uniforms ---
@group(2) @binding(0) var<uniform> u_ssao: SsaoUniforms;

// Reconstruct view-space position from screen UV + depth (reverse-Z).
// WebGPU NDC: x [-1, 1], y [1, -1] (Y-down), z [0, 1]
fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let ndc = vec4<f32>(ndc_x, ndc_y, depth, 1.0);

    let view_pos = u_render_state.projection_inverse * ndc;
    return view_pos.xyz / view_pos.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // 1. Read depth — skip skybox pixels (reverse-Z: near=1, far=0; skybox=0)
    let depth = textureSampleLevel(t_depth, s_linear, uv, 0u);
    if (depth <= 0.0) { // background / skybox — no occlusion
        return vec4<f32>(1.0);
    }

    // 2. Reconstruct view-space position
    let view_pos = reconstruct_view_position(uv, depth);

    // 3. Unpack normal from [0,1] → [-1,1]
    let packed_normal = textureSampleLevel(t_normal, s_linear, uv, 0.0);
    // Alpha encoding (Thin G-Buffer):
    //   0.0       = background (cleared); no geometry drawn here
    //   1.0       = valid geometry, no SS effects
    //   (0, 1)    = SS geometry (profile ID encoded as round(a * 255))
    // Any non-zero alpha means geometry was drawn — include it in SSAO.
    if (packed_normal.a <= 0.0) {
        return vec4<f32>(1.0);
    }
    let view_normal = normalize(packed_normal.xyz * 2.0 - 1.0);

    // 4. Read tiled noise & build TBN matrix (tangent space → view space)
    let random_vec = normalize(
        textureSampleLevel(t_noise, s_noise, uv * u_ssao.noise_scale, 0.0).xyz * 2.0 - 1.0
    );
    let tangent = normalize(random_vec - view_normal * dot(random_vec, view_normal));
    let bitangent = cross(view_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, view_normal);

    // 5. Core sampling loop — hemisphere occlusion with range check
    var occlusion: f32 = 0.0;
    let sample_count = u_ssao.sample_count;

    for (var i: u32 = 0u; i < sample_count; i++) {
        // Rotate sample into view space and offset by radius
        let sample_dir = tbn * u_ssao.samples[i].xyz;
        let sample_pos = view_pos + sample_dir * u_ssao.radius;

        // Project to screen UV
        var offset_clip = u_render_state.projection_matrix * vec4<f32>(sample_pos, 1.0);
        offset_clip /= offset_clip.w;
        let offset_uv = vec2<f32>(
            offset_clip.x * 0.5 + 0.5,
            0.5 - offset_clip.y * 0.5
        );

        // Read actual depth at that screen position and reconstruct its view-space Z
        let real_depth = textureSampleLevel(t_depth, s_linear, offset_uv, 0u);
        let real_view_pos = reconstruct_view_position(offset_uv, real_depth);

        // Range check: smooth falloff prevents far-away geometry from casting false AO.
        // Uses the absolute depth difference vs radius to create a [0,1] weight.
        let distance_diff = abs(view_pos.z - real_view_pos.z);
        let range_check = smoothstep(0.0, 1.0, u_ssao.radius / max(distance_diff, 0.0001));

        // Occlusion test: if the real surface is closer to the camera than our
        // sample point (plus bias), it occludes.
        // In view space with reverse-Z projection, closer means *larger* Z.
        if (real_view_pos.z >= sample_pos.z + u_ssao.bias) {
            occlusion += range_check;
        }
    }

    // Normalize and invert: 1.0 = fully lit, 0.0 = fully occluded
    occlusion = 1.0 - (occlusion / f32(sample_count));

    // Apply intensity as an exponent for contrast control
    let ao = pow(occlusion, u_ssao.intensity);

    return vec4<f32>(ao, ao, ao, 1.0);
}
