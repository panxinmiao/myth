{{ vertex_input_code }}
{{ binding_code }}
{$ include 'morph_pars' $}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    $$ if HAS_UV
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if OUTPUT_NORMAL and HAS_NORMAL
    @location({{ loc.next() }}) world_normal: vec3<f32>,
    $$ endif
};

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = in.position;

    $$ if OUTPUT_NORMAL and HAS_NORMAL
    var local_normal = in.normal;
    $$ endif

    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);
    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    $$ if SHADOW_PASS
    out.position = u_shadow_light.view_projection * world_pos;
    $$ else
    out.position = u_render_state.view_projection * world_pos;
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
@fragment
fn fs_main(varyings: VertexOutput) -> @location(0) vec4<f32> {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    {$ include 'alpha_test' $}

    // Encode screen-space profile ID into Normal.a (Thin G-Buffer channel).
    // Encoding:
    //   alpha == 0.0          → background (cleared, never written by geometry)
    //   alpha == 1.0 (255/255) → valid geometry, no SS effects
    //   alpha ∈ (0, 1)         → SS geometry; round(alpha * 255) = profile ID (1–254)
    $$ if USE_SCREEN_SPACE_FEATURES
    let ss_alpha = select(f32(u_material.screen_space_id) / 255.0, 1.0, u_material.screen_space_id == 0u);
    $$ else
    let ss_alpha = 1.0;
    $$ endif

    $$ if HAS_NORMAL
    // Transform world-space normal to view-space, then encode [-1,1] → [0,1]
    let view_normal = normalize((u_render_state.view_matrix * vec4<f32>(varyings.world_normal, 0.0)).xyz);
    return vec4<f32>(view_normal * 0.5 + 0.5, ss_alpha);
    $$ else
    // Fallback: camera-facing default (view-space Z-forward)
    return vec4<f32>(0.5, 0.5, 1.0, ss_alpha);
    $$ endif
}
$$ else
@fragment
fn fs_main(varyings: VertexOutput) {
    var opacity = u_material.opacity;

    $$ if HAS_MAP
    let tex_color = textureSample(t_map, s_map, varyings.uv);
    opacity *= tex_color.a;
    $$ endif

    {$ include 'alpha_test' $}
}
$$ endif
