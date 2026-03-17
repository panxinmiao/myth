// FXAA (Fast Approximate Anti-Aliasing) Post-Processing Shader
//
// Based on NVIDIA FXAA 3.11 algorithm. Identifies aliased edges via luma
// contrast detection and applies sub-pixel smoothing.
//
// Quality presets control the edge exploration iteration count:
// - Low:    4 iterations  (mobile / low-end)
// - Medium: 8 iterations  (default)
// - High:  12 iterations  (maximum quality)

{$ include 'full_screen_vertex.wgsl' $}

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

// === Quality preset selection (mutually exclusive defines) ===

$$ if FXAA_QUALITY_LOW is defined
const EDGE_THRESHOLD_MIN: f32 = 0.0833;
const EDGE_THRESHOLD_MAX: f32 = 0.250;
const ITERATIONS: i32 = 4;
$$ elif FXAA_QUALITY_HIGH is defined
const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const EDGE_THRESHOLD_MAX: f32 = 0.125;
const ITERATIONS: i32 = 12;
$$ else
// Default: MEDIUM
const EDGE_THRESHOLD_MIN: f32 = 0.0625;
const EDGE_THRESHOLD_MAX: f32 = 0.166;
const ITERATIONS: i32 = 8;
$$ endif

const SUBPIXEL_QUALITY: f32 = 0.75;

// Exploration step quality multiplier — accelerates search along edges
// at higher iteration counts to cover more distance with fewer samples.
fn get_exploration_quality(q: i32) -> f32 {
    switch (q) {
        default:          { return 1.0; }
        case 5:           { return 1.5; }
        case 6, 7, 8, 9:  { return 2.0; }
        case 10:          { return 4.0; }
        case 11:          { return 8.0; }
    }
}

// Perceptual luminance (gamma-aware via sqrt approximation)
fn rgb2luma(rgb: vec3<f32>) -> f32 {
    return sqrt(dot(rgb, vec3<f32>(0.299, 0.587, 0.114)));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let resolution = vec2<f32>(textureDimensions(screen_texture));
    let inverse_screen_size = 1.0 / resolution;
    let tex_coord = in.uv;

    // ── Center sample ───────────────────────────────────────────────────
    let center_sample = textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0);
    let luma_center = rgb2luma(center_sample.rgb);

    // ── 4-neighbourhood luma ────────────────────────────────────────────
    let luma_down  = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>( 0, -1)).rgb);
    let luma_up    = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>( 0,  1)).rgb);
    let luma_left  = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>(-1,  0)).rgb);
    let luma_right = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>( 1,  0)).rgb);

    let luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    let luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));
    let luma_range = luma_max - luma_min;

    // ── Early exit: insufficient contrast ───────────────────────────────
    if (luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX)) {
        return center_sample;
    }

    // ── Diagonal neighbourhood luma ─────────────────────────────────────
    let luma_dl = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>(-1, -1)).rgb);
    let luma_ur = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>( 1,  1)).rgb);
    let luma_ul = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>(-1,  1)).rgb);
    let luma_dr = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, tex_coord, 0.0, vec2<i32>( 1, -1)).rgb);

    let luma_down_up     = luma_down + luma_up;
    let luma_left_right  = luma_left + luma_right;
    let luma_left_corners  = luma_dl + luma_ul;
    let luma_down_corners  = luma_dl + luma_dr;
    let luma_right_corners = luma_dr + luma_ur;
    let luma_up_corners    = luma_ur + luma_ul;

    // ── Edge direction detection ────────────────────────────────────────
    let edge_horizontal = abs(-2.0 * luma_left   + luma_left_corners)
                        + abs(-2.0 * luma_center + luma_down_up) * 2.0
                        + abs(-2.0 * luma_right  + luma_right_corners);

    let edge_vertical   = abs(-2.0 * luma_up     + luma_up_corners)
                        + abs(-2.0 * luma_center + luma_left_right) * 2.0
                        + abs(-2.0 * luma_down   + luma_down_corners);

    let is_horizontal = (edge_horizontal >= edge_vertical);

    // ── Step along the edge normal ──────────────────────────────────────
    var step_length = select(inverse_screen_size.x, inverse_screen_size.y, is_horizontal);

    var luma_1 = select(luma_left, luma_down, is_horizontal);
    var luma_2 = select(luma_right, luma_up, is_horizontal);

    let grad_1 = luma_1 - luma_center;
    let grad_2 = luma_2 - luma_center;

    let is_1_steepest = abs(grad_1) >= abs(grad_2);
    let grad_scaled = 0.25 * max(abs(grad_1), abs(grad_2));

    var luma_local_avg = 0.0;
    if (is_1_steepest) {
        step_length = -step_length;
        luma_local_avg = 0.5 * (luma_1 + luma_center);
    } else {
        luma_local_avg = 0.5 * (luma_2 + luma_center);
    }

    // ── Start position: shift half a pixel along the edge normal ────────
    var current_uv = tex_coord;
    var offset = vec2<f32>(0.0, 0.0);
    if (is_horizontal) {
        current_uv.y += step_length * 0.5;
        offset.x = inverse_screen_size.x;
    } else {
        current_uv.x += step_length * 0.5;
        offset.y = inverse_screen_size.y;
    }

    // ── Initial exploration samples ─────────────────────────────────────
    var uv1 = current_uv - offset;
    var uv2 = current_uv + offset;

    var luma_end_1 = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, uv1, 0.0).rgb) - luma_local_avg;
    var luma_end_2 = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, uv2, 0.0).rgb) - luma_local_avg;

    var reached_1 = abs(luma_end_1) >= grad_scaled;
    var reached_2 = abs(luma_end_2) >= grad_scaled;
    var reached_both = reached_1 && reached_2;

    uv1 = select(uv1 - offset, uv1, reached_1);
    uv2 = select(uv2 + offset, uv2, reached_2);

    // ── Edge exploration loop ───────────────────────────────────────────
    if (!reached_both) {
        for (var i: i32 = 2; i < ITERATIONS; i++) {
            if (!reached_1) {
                luma_end_1 = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, uv1, 0.0).rgb) - luma_local_avg;
            }
            if (!reached_2) {
                luma_end_2 = rgb2luma(textureSampleLevel(screen_texture, tex_sampler, uv2, 0.0).rgb) - luma_local_avg;
            }

            reached_1 = abs(luma_end_1) >= grad_scaled;
            reached_2 = abs(luma_end_2) >= grad_scaled;
            reached_both = reached_1 && reached_2;

            if (!reached_1) { uv1 -= offset * get_exploration_quality(i); }
            if (!reached_2) { uv2 += offset * get_exploration_quality(i); }

            if (reached_both) { break; }
        }
    }

    // ── Compute final offset ────────────────────────────────────────────
    var dist_1 = select(tex_coord.y - uv1.y, tex_coord.x - uv1.x, is_horizontal);
    var dist_2 = select(uv2.y - tex_coord.y, uv2.x - tex_coord.x, is_horizontal);

    let is_dir_1 = dist_1 < dist_2;
    let dist_final = min(dist_1, dist_2);
    let edge_thickness = dist_1 + dist_2;

    let is_luma_center_smaller = luma_center < luma_local_avg;
    let correct_variation_1 = (luma_end_1 < 0.0) != is_luma_center_smaller;
    let correct_variation_2 = (luma_end_2 < 0.0) != is_luma_center_smaller;
    var correct_variation = select(correct_variation_2, correct_variation_1, is_dir_1);

    let pixel_offset = -dist_final / edge_thickness + 0.5;
    var final_offset = select(0.0, pixel_offset, correct_variation);

    // ── Sub-pixel anti-aliasing ─────────────────────────────────────────
    let luma_average = (1.0 / 12.0) * (2.0 * (luma_down_up + luma_left_right) + luma_left_corners + luma_right_corners);
    let subpixel_offset_1 = clamp(abs(luma_average - luma_center) / luma_range, 0.0, 1.0);
    let subpixel_offset_2 = (-2.0 * subpixel_offset_1 + 3.0) * subpixel_offset_1 * subpixel_offset_1;
    let subpixel_offset_final = subpixel_offset_2 * subpixel_offset_2 * SUBPIXEL_QUALITY;

    final_offset = max(final_offset, subpixel_offset_final);

    // ── Final sample ────────────────────────────────────────────────────
    var final_uv = tex_coord;
    if (is_horizontal) {
        final_uv.y += final_offset * step_length;
    } else {
        final_uv.x += final_offset * step_length;
    }

    let final_color = textureSampleLevel(screen_texture, tex_sampler, final_uv, 0.0).rgb;
    return vec4<f32>(final_color, center_sample.a);
}
