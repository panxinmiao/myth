// ── Blinn-Phong Material Entry Point ────────────────────────────────────
//
// Forward-rendered Blinn-Phong shading with Schlick Fresnel, normal
// mapping, ambient occlusion, light maps, and emissive support.

{{ vertex_input_code }} 
{{ binding_code }}
{$ include 'core/vertex_output' $}
{$ include 'core/fragment_output' $}

{$ include 'modules/geometry/morphing' $}
{$ include 'modules/geometry/skinning' $}
{$ include 'core/common' $}
{$ include 'modules/lighting/punctual' $}

// ── Screen / Transient BindGroup (Group 3) ──────────────────────────
@group(3) @binding(1) var s_screen_sampler: sampler;
@group(3) @binding(2) var t_ssao: texture_2d<f32>;
@group(3) @binding(3) var t_shadow_map_2d_array: texture_depth_2d_array;
@group(3) @binding(4) var t_shadow_map_cube_array: texture_depth_cube_array;
@group(3) @binding(5) var s_shadow_map_compare: sampler_comparison;

{$ include 'modules/bsdf/phong' $}
{$ include 'core/alpha_test' $}


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);
    var local_normal = vec3<f32>(in.normal.xyz);

    $$ if HAS_TANGENT is defined
    var object_tangent = vec3<f32>(in.tangent.xyz);
    $$ endif

    // ── Morph Target Blending ────────────────────────────────────────
    $$ if HAS_MORPH_TARGETS
        let morphed = apply_morph_targets(
            vertex_index,
            local_position,
            $$ if HAS_MORPH_NORMALS and HAS_NORMAL
            local_normal,
            $$ endif
            $$ if HAS_MORPH_TANGENTS and HAS_TANGENT
            object_tangent,
            $$ endif
        );
        local_position = morphed.position;
        $$ if HAS_MORPH_NORMALS and HAS_NORMAL
        local_normal = morphed.normal;
        $$ endif
        $$ if HAS_MORPH_TANGENTS and HAS_TANGENT
        object_tangent = morphed.tangent;
        $$ endif
    $$ endif

    var local_pos = vec4<f32>(local_position, 1.0);

    // ── Skeletal Skinning ────────────────────────────────────────────
    $$ if HAS_SKINNING and SUPPORT_SKINNING
        let skinned = compute_skinned_vertex(
            local_pos,
            local_normal,
            $$ if HAS_TANGENT
            object_tangent,
            $$ endif
            vec4<u32>(in.joints),
            in.weights,
        );
        local_pos = skinned.position;
        local_normal = skinned.normal;
        $$ if HAS_TANGENT
        object_tangent = skinned.tangent;
        $$ endif
    $$ endif

    let world_pos = u_model.world_matrix * local_pos;

    $$ if IN_TRANSPARENT_PASS is defined
        out.position = u_render_state.unjittered_view_projection * world_pos;
    $$ else
        out.position = u_render_state.view_projection * world_pos;
    $$ endif

    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_COLOR
        out.color = in.color;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);

    $$ if HAS_TANGENT is defined
        let v_tangent = normalize(( u_model.world_matrix  * vec4f(object_tangent, 0.0) ).xyz);
        let v_bitangent = normalize(cross(out.normal, v_tangent) * in.tangent.w);
        out.v_tangent = vec3<f32>(v_tangent);
        out.v_bitangent = vec3<f32>(v_bitangent);
    $$ endif
    {$ include 'mixins/uv_vertex' $}
    return out;
}

@fragment
fn fs_main(varyings: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var normal = normalize(varyings.normal);
    $$ if FLAT_SHADING
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        normal = normalize(cross(u, v));
    $$ else
        normal = select(-normal, normal, is_front);
    $$ endif

    var diffuse_color = u_material.color;

    $$ if HAS_COLOR
        diffuse_color *= varyings.color;
    $$ endif

    {$ if HAS_MAP $}
        let tex_color = textureSample(t_map, s_map, varyings.map_uv);
        diffuse_color *= tex_color;
    {$ endif $}

    // Apply opacity
    var opacity = diffuse_color.a * u_material.opacity;
    diffuse_color.a = opacity;

    // Alpha test
    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif

    let view = normalize(u_render_state.camera_position - varyings.world_position);

    $$ if HAS_NORMAL_MAP is defined

        let tbn = getTangentFrame(view, normal, varyings.normal_map_uv );

        let normal_map = textureSample( t_normal_map, s_normal_map, varyings.normal_map_uv ) * 2.0 - 1.0;
        let map_n = vec3f(normal_map.xy * u_material.normal_scale, normal_map.z);
        normal = normalize(tbn * map_n);
    $$ endif


    $$ if HAS_SPECULAR_MAP is defined
        let specular_map = textureSample( t_specular_map, s_specular_map, varyings.specular_map_uv );
        let specular_strength = specular_map.r;
    $$ else
        let specular_strength = 1.0;
    $$ endif

    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    var geometry: GeometricContext;
    geometry.position = varyings.world_position;
    geometry.normal = normal;
    geometry.view_dir = view;

    let material = build_phong_material(diffuse_color.rgb, u_material.specular.rgb, u_material.shininess, specular_strength);

    evaluate_punctual_lights(geometry, material, &reflected_light);

    // Indirect Diffuse Light
    let ambient_color = u_environment.ambient_light.rgb;
    var irradiance = getAmbientLightIrradiance( ambient_color );

    $$ if HAS_LIGHT_MAP is defined
        let light_map_color = textureSample(t_light_map, s_light_map, varyings.light_map_uv ).rgb;
        irradiance += light_map_color * u_material.light_map_intensity;
    $$ endif

    RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

    // Ambient occlusion
    $$ if HAS_AO_MAP is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let ambient_occlusion = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;
        reflected_light.indirect_diffuse *= ambient_occlusion;
    $$ endif

    var out_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if HAS_EMISSIVE_MAP is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_color += emissive_color;

    return pack_fragment_output(vec4<f32>(out_color, diffuse_color.a));
}
