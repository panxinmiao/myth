// ============================================================================
// Screen-Space Sub-Surface Scattering (SSSSS) — Separable Blur Pass
//
// Implements a physically-motivated separable Gaussian blur for skin /
// wax / translucent materials.  Two instances of this shader are run
// back-to-back (H pass then V pass) to reconstruct a full 2-D scatter.
//
// # Alpha Channel Encoding (Thin G-Buffer)
//   alpha == 0.0           → background; skip entirely
//   alpha == 1.0 (255/255) → valid geometry, no SS effects; pass-through
//   alpha ∈ (0, 1)          → SS geometry; SSS profile ID = round(alpha*255)
//
// # Scatter-Radius Convention
//   `ScreenSpaceProfile.data_0.w` is the scatter radius as a **fraction of
//   screen height** (e.g. 0.03 = 3 %).   At 720 p that is ≈ 22 pixels spread.
//
// # Layout
//   @group(0): SsssUniforms (direction, aspect, texel_size)
//   @group(1): textures + SSS profiles storage buffer + sampler
// ============================================================================

// ─── Structs ─────────────────────────────────────────────────────────────────

struct SsssUniforms {
    texel_size:   vec2<f32>,  // [1/width, 1/height]
    direction:    vec2<f32>,  // [1,0] = H pass | [0,1] = V pass
    aspect_ratio: f32,        // width / height — corrects H step to equal physical size
    _p0:          f32,        // padding (Rust: _padding[0])
    _p1:          f32,        // padding (Rust: _padding[1])
    _p2:          f32,        // padding (Rust: _padding[2])
};

struct ScreenSpaceMaterialData {
    feature_flags: u32,
    _padding_0:    u32,
    _padding_1:    u32,
    _padding_2:    u32,
    data_0:        vec4<f32>, // [scatter_r, scatter_g, scatter_b, radius_fraction]
    data_1:        vec4<f32>, // reserved
};

// ─── Bindings ────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> u_ssss: SsssUniforms;

@group(1) @binding(0) var t_color:   texture_2d<f32>;
@group(1) @binding(1) var t_normal:  texture_2d<f32>;
@group(1) @binding(2) var t_depth:   texture_depth_2d;
@group(1) @binding(3) var<storage, read> u_profiles: array<ScreenSpaceMaterialData>;
@group(1) @binding(4) var s_linear:  sampler;

// ─── Constants ───────────────────────────────────────────────────────────────

/// SSS feature flag bit (must match `FEATURE_SSS` in Rust).
const FEATURE_SSS: u32 = 1u;

/// Half-extent of the 1-D kernel (full kernel = 2*KERNEL_HALF + 1 taps).
const KERNEL_HALF: i32 = 5;

/// Gaussian sigma² factor: exp(-GAUSS_SIGMA2 * t²), t ∈ [-1, 1].
/// σ² ≈ 1/6 → soft roll-off; increase to narrow the effective kernel.
const GAUSS_SIGMA2: f32 = 3.0;

/// Depth discontinuity threshold as a multiple of scatter_radius.
/// Samples whose depth difference exceeds `center_depth * radius * DEPTH_CUTOFF`
/// are down-weighted to prevent bleed across geometric edges.
const DEPTH_CUTOFF: f32 = 1.5;

// ─── Vertex Shader ───────────────────────────────────────────────────────────

/// Emits a full-screen triangle with vertex indices 0, 1, 2.
/// NDC coordinates: (-1,-1), (3,-1), (-1,3) cover the entire clip-space quad.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, vi == 1u);
    let y = select(-1.0, 3.0, vi == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// ─── Fragment Shader ─────────────────────────────────────────────────────────

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy * u_ssss.texel_size;

    let center_color  = textureSampleLevel(t_color,  s_linear, uv, 0.0);
    let center_packed = textureSampleLevel(t_normal, s_linear, uv, 0.0);
    let center_depth  = textureSampleLevel(t_depth,  s_linear, uv, 0u);
    let center_alpha  = center_packed.a;

    // ── Early-out ────────────────────────────────────────────────────────────
    // alpha <= 0   → background; never modified.
    // alpha >= 1 - 0.5/255 → non-SSS geometry (id=255 sentinel); pass-through.
    if (center_alpha <= 0.0 || center_alpha >= (254.5 / 255.0)) {
        return center_color;
    }

    // ── Profile lookup ───────────────────────────────────────────────────────
    let sss_id  = u32(round(center_alpha * 255.0));
    let profile = u_profiles[sss_id];

    // Check that the SSS feature flag is active.
    if ((profile.feature_flags & FEATURE_SSS) == 0u) {
        return center_color;
    }

    let scatter_color  = profile.data_0.rgb;
    let scatter_radius = profile.data_0.a;   // fraction of screen height

    // ── Compute per-sample UV step ───────────────────────────────────────────
    // For H pass: direction = (1,0), step_x = scatter_radius / aspect / KERNEL_HALF
    // For V pass: direction = (0,1), step_y = scatter_radius / KERNEL_HALF
    // This keeps the physical size equal between the two passes.
    let radius_uv = vec2<f32>(scatter_radius / u_ssss.aspect_ratio, scatter_radius);
    let step_uv   = u_ssss.direction * radius_uv / f32(KERNEL_HALF);

    // ── Centre normal (view-space, decoded from [0,1] → [-1,1]) ─────────────
    let center_normal = normalize(center_packed.xyz * 2.0 - 1.0);

    // Depth threshold for bilateral weighting: proportional to radius and depth
    // to give a consistent world-space cutoff.
    let depth_threshold = max(center_depth * scatter_radius * DEPTH_CUTOFF, 0.001);

    var color_sum  = vec3<f32>(0.0);
    var weight_sum = 0.0;

    // ── Separable bilateral Gaussian kernel ──────────────────────────────────
    for (var i: i32 = -KERNEL_HALF; i <= KERNEL_HALF; i++) {
        let t          = f32(i) / f32(KERNEL_HALF);          // normalised position [-1, 1]
        let sample_uv  = uv + step_uv * f32(i);

        let sample_packed = textureSampleLevel(t_normal, s_linear, sample_uv, 0.0);
        let sample_depth  = textureSampleLevel(t_depth,  s_linear, sample_uv, 0u);

        // Background samples: clamp to centre colour to avoid dark halos
        // at silhouette edges where the kernel extends into the skybox.
        if (sample_packed.a <= 0.0) {
            let gaussian = exp(-GAUSS_SIGMA2 * t * t);
            color_sum  += center_color.rgb * gaussian;
            weight_sum += gaussian;
            continue;
        }

        let sample_color  = textureSampleLevel(t_color, s_linear, sample_uv, 0.0).rgb;
        let sample_normal = normalize(sample_packed.xyz * 2.0 - 1.0);

        // Gaussian spatial weight
        let gaussian = exp(-GAUSS_SIGMA2 * t * t);

        // Bilateral depth weight: prevent SSS from bleeding across geometric edges.
        // Hard-clips when the depth gap exceeds the world-space radius threshold.
        let depth_diff    = abs(center_depth - sample_depth);
        let depth_weight  = select(0.0, 1.0, depth_diff < depth_threshold);

        // Normal similarity weight: further suppress bleeding around sharp edges.
        let normal_dot    = max(dot(center_normal, sample_normal), 0.0);
        let normal_weight = normal_dot * normal_dot; // power-2 tightens cutoff

        let weight  = gaussian * depth_weight * normal_weight;
        color_sum  += sample_color * weight;
        weight_sum += weight;
    }

    // ── Reconstruct blurred colour ───────────────────────────────────────────
    var blurred = center_color.rgb;
    if (weight_sum > 0.001) {
        blurred = color_sum / weight_sum;
    }

    // ── Apply scatter tint and blend ─────────────────────────────────────────
    // Physically, scattered light picks up the material's sub-surface colour.
    // We blend the tinted blurred colour with the original (blend_factor = 0.85
    // for strong but not physically-overblown SSS; tune via ScreenSpaceSettings
    // in future if per-profile control is desired).
    let sss_result  = blurred * scatter_color;
    let final_color = mix(center_color.rgb, sss_result, 0.85);

    return vec4<f32>(final_color, center_color.a);
}
