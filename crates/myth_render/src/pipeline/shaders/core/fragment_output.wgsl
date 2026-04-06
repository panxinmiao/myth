// ── Standard Fragment Output ────────────────────────────────────────────
//
// Provides a unified `FragmentOutput` struct and a convenience
// `pack_fragment_output` helper so that ALL material shaders (built-in
// and custom) produce an output layout that matches the Opaque Pass's
// color attachment configuration.
//
// When `HAS_MRT_SSSS` is defined the struct contains an additional
// `@location(1)` target for SSSS specular data.  Materials that do not
// perform their own specular split should call `pack_fragment_output`
// which fills extra MRT targets with safe default values.

struct FragmentOutput {
    @location(0) color: vec4<f32>,
$$ if HAS_MRT_SSSS is defined
    @location(1) specular: vec4<f32>,
$$ endif
};

/// Packs a single main color into a valid `FragmentOutput`.
///
/// Extra MRT targets (e.g. SSSS specular) are filled with zeros so that
/// non-PBR materials remain pipeline-compatible without any manual work.
fn pack_fragment_output(main_color: vec4<f32>) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = main_color;
$$ if HAS_MRT_SSSS is defined
    out.specular = vec4<f32>(0.0);
$$ endif
    return out;
}
