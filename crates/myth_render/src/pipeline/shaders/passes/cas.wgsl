// CAS (Contrast Adaptive Sharpening) — AMD FidelityFX CAS
//
// Applied after TAA resolve to recover fine detail lost to temporal
// filtering.  Uses a 3×3 neighbourhood to compute adaptive sharpening
// weights, avoiding over-sharpening in low-contrast regions.
//
// Reference: AMD FidelityFX CAS, GPUOpen (2019)

{$ include 'full_screen_vertex.wgsl' $}

// ── Bindings ────────────────────────────────────────────────────────────

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_nearest: sampler;

struct CasParams {
    sharpness: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};
@group(0) @binding(2) var<uniform> u_params: CasParams;

// ── Fragment Shader ─────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_dim = vec2<f32>(textureDimensions(t_source));
    let texel_size = 1.0 / tex_dim;
    let uv = in.uv;

    // 3×3 neighbourhood (cross + corners loaded via center tap)
    let c = textureSampleLevel(t_source, s_nearest, uv, 0.0).rgb;
    let n = textureSampleLevel(t_source, s_nearest, uv + vec2<f32>(0.0, -texel_size.y), 0.0).rgb;
    let s = textureSampleLevel(t_source, s_nearest, uv + vec2<f32>(0.0,  texel_size.y), 0.0).rgb;
    let w = textureSampleLevel(t_source, s_nearest, uv + vec2<f32>(-texel_size.x, 0.0), 0.0).rgb;
    let e = textureSampleLevel(t_source, s_nearest, uv + vec2<f32>( texel_size.x, 0.0), 0.0).rgb;

    // Min/max of the cross pattern
    let mn = min(c, min(min(n, s), min(w, e)));
    let mx = max(c, max(max(n, s), max(w, e)));

    // Compute per-channel adaptive sharpening weight.
    // A_peak controls how aggressive the sharpening is; sharpness=0 → no
    // sharpening, sharpness=1 → maximum.
    let a_peak = vec3<f32>(-1.0 / mix(8.0, 5.0, u_params.sharpness));

    // Soft minimum/maximum ratio drives the filter weight.
    let d = min(mn, vec3<f32>(2.0) - mx);
    var w_s = d * a_peak;
    w_s = clamp(w_s, vec3<f32>(-0.1875), vec3<f32>(0.0));

    // Weighted 5-tap sharpening filter
    let rcp_weight = 1.0 / (1.0 + 4.0 * w_s);
    let result = (c + (n + s + w + e) * w_s) * rcp_weight;

    return vec4<f32>(max(result, vec3<f32>(0.0)), 1.0);
}
