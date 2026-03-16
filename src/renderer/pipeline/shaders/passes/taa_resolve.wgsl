// TAA Resolve — Industrial-Grade Temporal Anti-Aliasing
//
// Pipeline:
//   1. Velocity Dilation    — 3×3 closest-depth → robust edge velocity
//   2. Depth Rejection      — disocclusion detection via history depth
//   3. Catmull-Rom 5-Tap    — high-quality bicubic history sampling
//   4. Reversible Tonemap   — HDR-safe neighbourhood operations
//   5. Variance Clipping    — soft AABB clamp in YCoCg space
//   6. Luminance-weighted blend + inverse tonemap

{$ include 'full_screen_vertex.wgsl' $}

// ── Bindings ────────────────────────────────────────────────────────────

@group(0) @binding(0) var t_current_color: texture_2d<f32>;
@group(0) @binding(1) var t_history_color: texture_2d<f32>;
@group(0) @binding(2) var t_velocity:      texture_2d<f32>;
@group(0) @binding(3) var t_scene_depth:   texture_depth_2d;
@group(0) @binding(4) var t_history_depth: texture_depth_2d;
@group(0) @binding(5) var s_linear:  sampler;
@group(0) @binding(6) var s_nearest: sampler;

struct TaaParams {
    feedback_weight: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};
@group(0) @binding(7) var<uniform> u_params: TaaParams;

// ── Constants ───────────────────────────────────────────────────────────

const DEPTH_REJECTION_TOLERANCE: f32 = 0.05;
const VARIANCE_CLIP_GAMMA: f32 = 1.25;

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

// ── Reversible Tonemapping (perceptual-space operations) ────────────────

fn tonemap_per_channel(c: vec3<f32>) -> vec3<f32> {
    return c / (1.0 + c);
}

fn inverse_tonemap_per_channel(c: vec3<f32>) -> vec3<f32> {
    return c / max(vec3<f32>(1.0) - c, vec3<f32>(0.0001));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── Reverse-Z depth → linear depth ─────────────────────────────────────
// The engine uses reverse-Z infinite projection, so Z=1 is near, Z=0 is far.
// Linear depth = near / z_ndc for reverse-Z infinite perspective.

fn depth_to_linear(z: f32, near: f32) -> f32 {
    return near / max(z, 0.0001);
}

// ── Variance Clipping (soft AABB in YCoCg space) ────────────────────────

fn clip_towards_aabb_center(
    history_ycocg: vec3<f32>,
    aabb_center: vec3<f32>,
    aabb_extent: vec3<f32>,
) -> vec3<f32> {
    let d = history_ycocg - aabb_center;
    let abs_d = abs(d);
    let safe_extent = max(aabb_extent, vec3<f32>(0.0001));
    let ratio = safe_extent / abs_d;
    let t = saturate(min(ratio.x, min(ratio.y, ratio.z)));
    return aabb_center + d * t;
}

// ── Catmull-Rom 5-Tap bicubic-approximation ─────────────────────────────
// Uses hardware bilinear filtering to approximate a 4×4 Catmull-Rom kernel
// with only 5 texture samples instead of 16.

fn sample_catmull_rom_5tap(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, tex_size: vec2<f32>) -> vec3<f32> {
    let sample_pos = uv * tex_size;
    let tc = floor(sample_pos - 0.5) + 0.5;

    let f = sample_pos - tc;
    let f2 = f * f;
    let f3 = f2 * f;

    // Catmull-Rom weights along each axis
    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + vec2<f32>(1.0);
    let w3 = 0.5 * (f3 - f2);
    let w2 = vec2<f32>(1.0) - w0 - w1 - w3;

    let w12 = w1 + w2;
    let offset12 = w2 / max(w12, vec2<f32>(0.0001));

    let tc0 = (tc - 1.0) / tex_size;
    let tc12 = (tc + offset12) / tex_size;
    let tc3 = (tc + 2.0) / tex_size;

    // 5 bilinear taps weighted to approximate 16-tap Catmull-Rom
    var color = textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y);
    color += textureSampleLevel(tex, samp, vec2<f32>(tc0.x,  tc12.y), 0.0).rgb * (w0.x  * w12.y);
    color += textureSampleLevel(tex, samp, vec2<f32>(tc3.x,  tc12.y), 0.0).rgb * (w3.x  * w12.y);
    color += textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc0.y),  0.0).rgb * (w12.x * w0.y);
    color += textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc3.y),  0.0).rgb * (w12.x * w3.y);

    return max(color, vec3<f32>(0.0));
}

// ── Fragment Shader ─────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let tex_dim = vec2<f32>(textureDimensions(t_current_color));
    let texel_size = 1.0 / tex_dim;

    // ════════════════════════════════════════════════════════════════════
    // 1. Velocity Dilation — find the 3×3 pixel closest to the camera
    //    and use its velocity.  Prevents edge-pull artifacts on moving
    //    object silhouettes.
    // ════════════════════════════════════════════════════════════════════

    var closest_depth = 0.0;  // reverse-Z: larger = closer
    var closest_offset = vec2<i32>(0, 0);

    let center_coord = vec2<i32>(in.position.xy);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let coord = center_coord + vec2<i32>(x, y);
            let d = textureLoad(t_scene_depth, coord, 0);
            if (d > closest_depth) {
                closest_depth = d;
                closest_offset = vec2<i32>(x, y);
            }
        }
    }

    let dilated_uv = uv + vec2<f32>(closest_offset) * texel_size;
    let velocity = textureSampleLevel(t_velocity, s_nearest, dilated_uv, 0.0).rg;

    // ════════════════════════════════════════════════════════════════════
    // 2. Reprojection + Depth Rejection
    // ════════════════════════════════════════════════════════════════════

    let history_uv = uv - velocity;

    // Reject out-of-screen history immediately
    if (history_uv.x < 0.0 || history_uv.x > 1.0 || history_uv.y < 0.0 || history_uv.y > 1.0) {
        return vec4<f32>(textureSampleLevel(t_current_color, s_nearest, uv, 0.0).rgb, 1.0);
    }

    // Depth rejection: compare current linear depth with history linear depth.
    // Uses camera near plane = 0.1 as a reasonable default for reverse-Z.
    let current_z = closest_depth;
    let history_z = textureSampleLevel(t_history_depth, s_nearest, history_uv, 0);
    let current_linear = depth_to_linear(current_z, 0.1);
    let history_linear = depth_to_linear(history_z, 0.1);
    let depth_diff = abs(current_linear - history_linear) / max(current_linear, 0.0001);
    let is_disoccluded = depth_diff > DEPTH_REJECTION_TOLERANCE;

    // ════════════════════════════════════════════════════════════════════
    // 3. Sample current frame colour (center pixel)
    // ════════════════════════════════════════════════════════════════════

    let current_color_hdr = textureSampleLevel(t_current_color, s_nearest, uv, 0.0).rgb;

    // If disoccluded, output current frame directly — zero ghosting.
    if (is_disoccluded) {
        return vec4<f32>(current_color_hdr, 1.0);
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. Catmull-Rom 5-Tap history sampling (high-quality bicubic)
    // ════════════════════════════════════════════════════════════════════

    let history_color_hdr = sample_catmull_rom_5tap(t_history_color, s_linear, history_uv, tex_dim);

    // ════════════════════════════════════════════════════════════════════
    // 5. Reversible Tonemap → YCoCg → Variance Clipping
    // ════════════════════════════════════════════════════════════════════

    // Tonemap all samples into perceptual space for stable neighbourhood ops
    let current_tm = tonemap_per_channel(current_color_hdr);
    let history_tm = tonemap_per_channel(history_color_hdr);

    let cc = rgb_to_ycocg(current_tm);
    var moment1 = cc;
    var moment2 = cc * cc;

    // 3×3 neighbourhood statistics (mean + variance)
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let s_hdr = textureSampleLevel(t_current_color, s_nearest, uv + offset, 0.0).rgb;
            let s = rgb_to_ycocg(tonemap_per_channel(s_hdr));
            moment1 += s;
            moment2 += s * s;
        }
    }

    let mean = moment1 / 9.0;
    let variance = sqrt(max(moment2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let aabb_extent = variance * VARIANCE_CLIP_GAMMA;

    // Clip history towards AABB center (soft clip, not hard clamp)
    let history_ycocg = rgb_to_ycocg(history_tm);
    let clipped_ycocg = clip_towards_aabb_center(history_ycocg, mean, aabb_extent);
    let clipped_history = ycocg_to_rgb(clipped_ycocg);

    // ════════════════════════════════════════════════════════════════════
    // 6. Luminance-weighted temporal blend
    // ════════════════════════════════════════════════════════════════════

    // Dynamic feedback: reduce history weight with motion speed
    let speed = length(velocity * tex_dim);
    var weight = u_params.feedback_weight;
    weight = mix(weight, 0.5, saturate(speed * 0.02));

    // Luminance-based weighting: reduce temporal weight for very bright pixels
    // let lum_current = luminance(current_tm);
    // let lum_history = luminance(clipped_history);
    // let unbiased_weight = weight * lum_history / max(lum_current + lum_history * weight, 0.0001);

    // let resolved_tm = mix(current_tm, clipped_history, unbiased_weight);

    let resolved_tm = mix(current_tm, clipped_history, weight);

    // ════════════════════════════════════════════════════════════════════
    // 7. Inverse Tonemap → HDR output
    // ════════════════════════════════════════════════════════════════════

    let resolved_hdr = inverse_tonemap_per_channel(resolved_tm);

    return vec4<f32>(resolved_hdr, 1.0);
}
