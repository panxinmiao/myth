{$ include 'full_screen_vertex.wgsl' $}

// --- Group 0: Bilateral Blur Inputs ---
@group(0) @binding(0) var t_raw_ao: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_normal: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;

// Cross-bilateral filter configuration
const BLUR_RADIUS: i32 = 2; // 5×5 kernel (2*2+1)
const DEPTH_SIGMA: f32 = 0.5;
const NORMAL_THRESHOLD: f32 = 0.9;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_raw_ao, 0));
    let texel_size = 1.0 / tex_size;
    let uv = in.uv;

    // Centre pixel reference values
    let center_ao = textureSampleLevel(t_raw_ao, s_linear, uv, 0.0).r;
    let center_depth = textureSampleLevel(t_depth, s_linear, uv, 0u);
    let center_normal_packed = textureSampleLevel(t_normal, s_linear, uv, 0.0);

    // If no geometry (skybox), return full-lit
    if (center_normal_packed.a < 0.5 || center_depth <= 0.0) {
        return vec4<f32>(1.0);
    }

    let center_normal = normalize(center_normal_packed.xyz * 2.0 - 1.0);

    var ao_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    // Cross-bilateral 5×5 filter: combine spatial + depth + normal weights
    for (var y: i32 = -BLUR_RADIUS; y <= BLUR_RADIUS; y++) {
        for (var x: i32 = -BLUR_RADIUS; x <= BLUR_RADIUS; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = uv + offset;

            let sample_ao = textureSampleLevel(t_raw_ao, s_linear, sample_uv, 0.0).r;
            let sample_depth = textureSampleLevel(t_depth, s_linear, sample_uv, 0u);
            let sample_normal_packed = textureSampleLevel(t_normal, s_linear, sample_uv, 0.0);

            // Skip background pixels
            if (sample_normal_packed.a < 0.5 || sample_depth <= 0.0) {
                continue;
            }

            let sample_normal = normalize(sample_normal_packed.xyz * 2.0 - 1.0);

            // --- Depth weight: exponential falloff ---
            let depth_diff = abs(center_depth - sample_depth);
            let depth_weight = exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));

            // --- Normal weight: hard-ish cutoff preserving geometric edges ---
            let normal_similarity = max(dot(center_normal, sample_normal), 0.0);
            let normal_weight = select(0.0, normal_similarity, normal_similarity > NORMAL_THRESHOLD);

            // --- Spatial weight: simple Gaussian (σ ≈ radius) ---
            let dist2 = f32(x * x + y * y);
            let spatial_sigma = f32(BLUR_RADIUS);
            let spatial_weight = exp(-dist2 / (2.0 * spatial_sigma * spatial_sigma));

            let w = spatial_weight * depth_weight * normal_weight;
            ao_sum += sample_ao * w;
            weight_sum += w;
        }
    }

    let final_ao = select(center_ao, ao_sum / weight_sum, weight_sum > 0.0001);

    return vec4<f32>(final_ao, final_ao, final_ao, 1.0);
}
