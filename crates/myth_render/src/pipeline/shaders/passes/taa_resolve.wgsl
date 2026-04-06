// TAA Resolve — Industrial-Grade Temporal Anti-Aliasing
//
// Pipeline:
//   1. Velocity Dilation    — 3×3 closest-depth → robust edge velocity
//   2. Depth Rejection      — disocclusion detection via history depth
//   3. Catmull-Rom 5-Tap    — high-quality bicubic history sampling
//   4. Reversible Tonemap   — HDR-safe neighbourhood operations
//   5. Variance Clipping    — soft AABB clamp in YCoCg space
//   6. Luminance-weighted blend + inverse tonemap

{$ include 'chunks/full_screen_vertex' $}

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
    camera_near: f32,
    _padding0: f32,
    _padding1: f32,
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
    let weight0 = w12.x * w12.y;
    let weight1 = w0.x  * w12.y;
    let weight2 = w3.x  * w12.y;
    let weight3 = w12.x * w0.y;
    let weight4 = w12.x * w3.y;
    let weight_sum = weight0 + weight1 + weight2 + weight3 + weight4;

    var c0 = tonemap_per_channel(textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc12.y), 0.0).rgb);
    var c1 = tonemap_per_channel(textureSampleLevel(tex, samp, vec2<f32>(tc0.x,  tc12.y), 0.0).rgb);
    var c2 = tonemap_per_channel(textureSampleLevel(tex, samp, vec2<f32>(tc3.x,  tc12.y), 0.0).rgb);
    var c3 = tonemap_per_channel(textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc0.y),  0.0).rgb);
    var c4 = tonemap_per_channel(textureSampleLevel(tex, samp, vec2<f32>(tc12.x, tc3.y),  0.0).rgb);

    var color_tm = c0 * weight0 + c1 * weight1 + c2 * weight2 + c3 * weight3 + c4 * weight4;

    return max(color_tm / weight_sum, vec3<f32>(0.0));
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
    let center_depth = textureLoad(t_scene_depth, center_coord, 0);

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


    let current_coord = vec2<i32>(in.position.xy);
    let current_depth = textureLoad(t_scene_depth, current_coord, 0);
    
    let history_coord = vec2<i32>(history_uv * tex_dim);
    let history_depth = textureLoad(t_history_depth, history_coord, 0);

    let current_linear_z = depth_to_linear(current_depth, u_params.camera_near);
    let history_linear_z = depth_to_linear(history_depth, u_params.camera_near);

    let depth_diff = abs(current_linear_z - history_linear_z);
    let is_disoccluded = depth_diff > (current_linear_z * DEPTH_REJECTION_TOLERANCE);

    // // ════════════════════════════════════════════════════════════════════
    // // 3. Sample current frame colour (center pixel)
    // // ════════════════════════════════════════════════════════════════════

    let current_color = tonemap_per_channel(textureSampleLevel(t_current_color, s_nearest, uv, 0.0).rgb);

    // ════════════════════════════════════════════════════════════════════
    // 4. Catmull-Rom 5-Tap history sampling (high-quality bicubic)
    // ════════════════════════════════════════════════════════════════════

    // let history_color_hdr = sample_catmull_rom_5tap(t_history_color, s_linear, history_uv, tex_dim);
    let history_color = sample_catmull_rom_5tap(t_history_color, s_linear, history_uv, tex_dim);

    // ════════════════════════════════════════════════════════════════════
    // 5. Reversible Tonemap → YCoCg → Variance Clipping
    // ════════════════════════════════════════════════════════════════════

    // Tonemap all samples into perceptual space for stable neighbourhood ops
    // let current_tm = tonemap_per_channel(current_color);

    let cc = rgb_to_ycocg(current_color);
    var moment1 = cc;
    var moment2 = cc * cc;

    var sample_count = 1.0;

    // Include the center pixel in the variance stats (important for flat areas)
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>( 0.0, -1.0), vec2<f32>(-1.0,  0.0),
        vec2<f32>( 1.0,  0.0), vec2<f32>( 0.0,  1.0)
    );

    for (var i = 0; i < 4; i++) {
        let offset = offsets[i] * texel_size;
        let s_hdr = textureSampleLevel(t_current_color, s_nearest, uv + offset, 0.0).rgb;
        let s = rgb_to_ycocg(tonemap_per_channel(s_hdr));
        moment1 += s;
        moment2 += s * s;
        sample_count += 1.0;
    }

    // // 3×3 neighbourhood statistics (mean + variance)
    // for (var y = -1; y <= 1; y++) {
    //     for (var x = -1; x <= 1; x++) {
    //         if (x == 0 && y == 0) { continue; }
    //         let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
    //         let s_hdr = textureSampleLevel(t_current_color, s_nearest, uv + offset, 0.0).rgb;
    //         let s = rgb_to_ycocg(tonemap_per_channel(s_hdr));
    //         moment1 += s;
    //         moment2 += s * s;
    //     }
    // }

    let mean = moment1 / sample_count;
    let variance = sqrt(max(moment2 / sample_count - mean * mean, vec3<f32>(0.0)));
    let aabb_extent = variance * VARIANCE_CLIP_GAMMA;

    // Clip history towards AABB center (soft clip, not hard clamp)
    let history_ycocg = rgb_to_ycocg(history_color);
    let clipped_ycocg = clip_towards_aabb_center(history_ycocg, mean, aabb_extent);
    let clipped_history = ycocg_to_rgb(clipped_ycocg);

    // ════════════════════════════════════════════════════════════════════
    // 6. Luminance-weighted temporal blend
    // ════════════════════════════════════════════════════════════════════

    // Dynamic feedback: reduce history weight with motion speed
    let speed = length(velocity * tex_dim);
    var base_weight = u_params.feedback_weight;

    base_weight = mix(base_weight, 0.1, saturate(speed * 0.1));

    if (is_disoccluded) {
        base_weight = 0.0; 
    }

    let lum_current = luminance(current_color);
    let lum_history = luminance(clipped_history);

    let lum_diff = abs(lum_current - lum_history);

    let final_weight = base_weight * (1.0 / (1.0 + lum_diff * 4.0)); // 4.0 is a tunable sensitivity factor

    let resolved_tm = mix(current_color, clipped_history, final_weight);

    // ════════════════════════════════════════════════════════════════════
    // 7. Inverse Tonemap → HDR output
    // ════════════════════════════════════════════════════════════════════

    let resolved_hdr = inverse_tonemap_per_channel(resolved_tm);

    return vec4<f32>(resolved_hdr, 1.0);
}
