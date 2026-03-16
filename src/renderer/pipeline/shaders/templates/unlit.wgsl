{{ vertex_input_code }} 
{{ binding_code }}      
{$ include 'vertex_output_def' $}



{$ include 'morph_pars' $}


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);

    $$ if HAS_NORMAL
    var local_normal = vec3<f32>(in.normal.xyz);
    $$ endif

    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);
    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    let clip_pos = u_render_state.view_projection * world_pos;
    out.position = clip_pos;
    out.clip_position = clip_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_VELOCITY_TARGET is defined
    let prev_world_pos = u_model.previous_world_matrix * local_pos;
    out.prev_clip_position = u_render_state.prev_unjittered_view_projection * prev_world_pos;
    $$ endif

    $$ if HAS_COLOR
        out.color = in.color;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    $$ if HAS_NORMAL
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    {$ include 'uv_vertex' $}
    return out;
}


$$ if HAS_VELOCITY_TARGET is defined
struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};
$$ endif

@fragment
$$ if HAS_VELOCITY_TARGET is defined
fn fs_main(in: VertexOutput) -> FragmentOutput {
$$ else
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
$$ endif
    var diffuse_color = u_material.color;
    {$ if HAS_MAP $}
    let tex_color = textureSample(t_map, s_map, in.uv);
    diffuse_color = diffuse_color * tex_color;
    {$ endif $}

    {$ include 'alpha_test' $}

    $$ if HAS_VELOCITY_TARGET is defined
    var out: FragmentOutput;
    out.color = diffuse_color;
    let unjittered_clip = u_render_state.unjittered_view_projection * vec4<f32>(in.world_position, 1.0);
    let ndc_curr = unjittered_clip.xy / unjittered_clip.w;
    let ndc_prev = in.prev_clip_position.xy / in.prev_clip_position.w;
    out.velocity = (ndc_curr - ndc_prev) * vec2<f32>(0.5, -0.5);
    return out;
    $$ else
    return diffuse_color;
    $$ endif
}