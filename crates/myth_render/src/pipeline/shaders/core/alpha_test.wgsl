// ── Alpha Test Function ─────────────────────────────────────────────────
//
// Conditionally discards fragments based on opacity threshold.
// Include this file at module scope; call apply_alpha_test() from the
// fragment shader body.

$$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"

/// Applies alpha test with optional alpha-to-coverage smoothing.
/// Modifies `opacity` in-place and issues `discard` when below threshold.
fn apply_alpha_test(opacity: ptr<function, f32>, alpha_threshold: f32) {
    $$ if ALPHA_TO_COVERAGE is defined

    let aa_width = max(fwidth(*opacity) * 1.5, 0.0001);
    *opacity = smoothstep(alpha_threshold - aa_width * 0.5, alpha_threshold + aa_width * 0.5, *opacity);
    *opacity = clamp(*opacity, 0.0, 1.0);

    if ( *opacity <= 0.01 ) {
        discard;
    }

    $$ else

    if *opacity < alpha_threshold {
        discard;
    }

    $$ endif
}

$$ endif
