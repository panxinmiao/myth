// ── Depth Prepass / Shadow Pass Entry Point ─────────────────────────────
//
// Renders depth-only, with optional normal/velocity output for screen-
// space effects (SSAO, TAA).  Also used as shadow pass geometry shader.

{{ vertex_input_code }}
{{ binding_code }}
{$ include 'modules/geometry/morphing' $}
{$ include 'modules/geometry/skinning' $}
{$ include 'core/alpha_test' $}

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    $$ if HAS_UV
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if OUTPUT_NORMAL and HAS_NORMAL
    @location({{ loc.next() }}) world_normal: vec3<f32>,
    $$ endif
    $$ if HAS_VELOCITY_TARGET is defined
    @location({{ loc.next() }}) curr_unjittered_clip_position: vec4<f32>,
    @location({{ loc.next() }}) prev_clip_position: vec4<f32>,
    $$ endif
};

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);

    $$ if HAS_VELOCITY_TARGET is defined
    var prev_local_position = vec3<f32>(in.position.xyz);
    $$ endif

    $$ if OUTPUT_NORMAL and HAS_NORMAL
    var local_normal = vec3<f32>(in.normal.xyz);
    $$ endif

    // ── Morph Target Blending ────────────────────────────────────────
    $$ if HAS_MORPH_TARGETS
        let morphed = apply_morph_targets(
            vertex_index,
            local_position,
            $$ if HAS_VELOCITY_TARGET is defined
            prev_local_position,
            $$ endif
            $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
            local_normal,
            $$ endif
        );
        local_position = morphed.position;
        $$ if HAS_VELOCITY_TARGET is defined
        prev_local_position = morphed.prev_position;
        $$ endif
        $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
        local_normal = morphed.normal;
        $$ endif
    $$ endif

    var local_pos = vec4<f32>(local_position, 1.0);
    $$ if HAS_VELOCITY_TARGET is defined
    var prev_local_pos = vec4<f32>(prev_local_position, 1.0);
    $$ endif

    // ── Skeletal Skinning ────────────────────────────────────────────
    $$ if HAS_SKINNING and SUPPORT_SKINNING
        let skinned = compute_skinned_vertex(
            local_pos,
            $$ if HAS_VELOCITY_TARGET is defined
            prev_local_pos,
            $$ endif
            $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
            local_normal,
            $$ endif
            vec4<u32>(in.joints),
            in.weights,
        );
        local_pos = skinned.position;
        $$ if HAS_VELOCITY_TARGET is defined
        prev_local_pos = skinned.prev_position;
        $$ endif
        $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
        local_normal = skinned.normal;
        $$ endif
    $$ endif

    let world_pos = u_model.world_matrix * local_pos;

    $$ if SHADOW_PASS
    out.position = u_shadow_light.view_projection * world_pos;
    $$ else
    out.position = u_render_state.view_projection * world_pos;
    $$ endif

    $$ if HAS_VELOCITY_TARGET is defined
    let prev_world_pos = u_model.previous_world_matrix * prev_local_pos;
    out.prev_clip_position = u_render_state.prev_unjittered_view_projection * prev_world_pos;
    out.curr_unjittered_clip_position = u_render_state.unjittered_view_projection * world_pos;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    $$ if OUTPUT_NORMAL and HAS_NORMAL
    out.world_normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    return out;
}

$$ if OUTPUT_NORMAL

struct FragmentOutput {
    @location(0) normal: vec4<f32>,

    $$ if USE_SCREEN_SPACE_FEATURES
    @location(1) feature_id: vec2<u32>,
    $$ endif

    $$ if HAS_VELOCITY_TARGET is defined
        $$ if USE_SCREEN_SPACE_FEATURES
    @location(2) velocity: vec2<f32>,
        $$ else
    @location(1) velocity: vec2<f32>,
        $$ endif
    $$ endif
};

@fragment
fn fs_main(varyings: VertexOutput) -> FragmentOutput {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif

    var out: FragmentOutput;

    $$ if HAS_NORMAL
    let view_normal = normalize((u_render_state.view_matrix * vec4<f32>(varyings.world_normal, 0.0)).xyz);
    out.normal = vec4<f32>(view_normal * 0.5 + 0.5, 1.0);
    $$ else
    out.normal = vec4<f32>(0.5, 0.5, 1.0, 1.0);
    $$ endif

    $$ if USE_SCREEN_SPACE_FEATURES
    out.feature_id = vec2<u32>(u_material.sss_id, u_material.ssr_id);
    $$ endif

    $$ if HAS_VELOCITY_TARGET is defined
    let ndc_curr = varyings.curr_unjittered_clip_position.xy / varyings.curr_unjittered_clip_position.w;
    let ndc_prev = varyings.prev_clip_position.xy / varyings.prev_clip_position.w;
    out.velocity = (ndc_curr - ndc_prev) * vec2<f32>(0.5, -0.5);
    $$ endif

    return out;
}

$$ elif HAS_VELOCITY_TARGET is defined

struct FragmentOutput {
    @location(0) velocity: vec2<f32>,
};

@fragment
fn fs_main(varyings: VertexOutput) -> FragmentOutput {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif

    var out: FragmentOutput;
    let ndc_curr = varyings.curr_unjittered_clip_position.xy / varyings.curr_unjittered_clip_position.w;
    let ndc_prev = varyings.prev_clip_position.xy / varyings.prev_clip_position.w;
    out.velocity = (ndc_curr - ndc_prev) * vec2<f32>(0.5, -0.5);
    return out;
}

$$ else

@fragment
fn fs_main(varyings: VertexOutput) {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif
}

$$ endif
