// TAA Resolve — Temporal Anti-Aliasing full-screen post-process pass.
//
// Inputs:
//   t_current_color  — current frame scene colour (HDR)
//   t_history_color  — previous frame TAA output   (HDR, bilinear)
//   t_velocity       — screen-space motion vectors  (Rg16Float, point)
//
// Algorithm:
//   1. Reprojection     — sample history at (uv - velocity)
//   2. Neighbourhood clamp — 3×3 YCoCg AABB around current pixel
//   3. Temporal blend   — mix(current, clamped_history, weight)

{$ include 'full_screen_vertex.wgsl' $}

// ── Bindings ────────────────────────────────────────────────────────────

@group(0) @binding(0) var t_current_color: texture_2d<f32>;
@group(0) @binding(1) var t_history_color: texture_2d<f32>;
@group(0) @binding(2) var t_velocity: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var s_nearest: sampler;

struct TaaParams {
    feedback_weight: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};
@group(0) @binding(5) var<uniform> u_params: TaaParams;

// ── Colour-space helpers ────────────────────────────────────────────────

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(rgb, vec3<f32>(0.25, 0.5, 0.25)),
        dot(rgb, vec3<f32>(0.5, 0.0, -0.5)),
        dot(rgb, vec3<f32>(-0.25, 0.5, -0.25))
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z
    );
}

// ── Fragment shader ─────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let tex_dim = vec2<f32>(textureDimensions(t_current_color));
    let texel_size = 1.0 / tex_dim;

    // 1. Sample current colour and velocity
    let current_color = textureSampleLevel(t_current_color, s_nearest, uv, 0.0).rgb;
    let velocity = textureSampleLevel(t_velocity, s_nearest, uv, 0.0).rg;

    // 2. Reprojection
    let history_uv = uv - velocity;

    // Reject out-of-screen history
    if (history_uv.x < 0.0 || history_uv.x > 1.0 || history_uv.y < 0.0 || history_uv.y > 1.0) {
        return vec4<f32>(current_color, 1.0);
    }

    var history_color = textureSampleLevel(t_history_color, s_linear, history_uv, 0.0).rgb;

    // 3. Neighbourhood clamp (YCoCg AABB, plus-shaped 5-tap kernel)
    let cc = rgb_to_ycocg(current_color);
    var c_min = cc;
    var c_max = cc;

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>( texel_size.x, 0.0),
        vec2<f32>(-texel_size.x, 0.0),
        vec2<f32>(0.0,  texel_size.y),
        vec2<f32>(0.0, -texel_size.y)
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        let s = rgb_to_ycocg(
            textureSampleLevel(t_current_color, s_nearest, uv + offsets[i], 0.0).rgb
        );
        c_min = min(c_min, s);
        c_max = max(c_max, s);
    }

    var history_ycocg = rgb_to_ycocg(history_color);
    history_ycocg = clamp(history_ycocg, c_min, c_max);
    history_color = ycocg_to_rgb(history_ycocg);

    // 4. Dynamic feedback weight — reduce history influence with motion
    let speed = length(velocity);
    var weight = u_params.feedback_weight;
    weight = mix(weight, 0.7, saturate(speed * 100.0));

    // 5. Final blend
    let resolved = mix(current_color, history_color, weight);
    return vec4<f32>(resolved, 1.0);
}
