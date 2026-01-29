{{ vertex_input_code }} 
{{ binding_code }}
{$ include 'vertex_output_def' $}

{$ include 'morph_pars' $}
{$ include 'light_common_pars' $}
{$ include 'light_punctual_pars' $}
{$ include 'bsdf/phong' $}


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = in.position;
    var local_normal = in.normal;

    $$ if HAS_TANGENT is defined
    var object_tangent = in.tangent.xyz;
    $$ endif

    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);

    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    out.position = u_render_state.view_projection * world_pos;
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
    {$ include 'uv_vertex' $}
    return out;
}

@fragment
fn fs_main(varyings: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var normal = normalize(varyings.normal);
    $$ if FLAT_SHADING
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        normal = normalize(cross(u, v));
    $$ else
        normal = select(-normal, normal, is_front);
    $$ endif


    $$ if COLOR_MODE == 'normal'
        var diffuse_color = vec4<f32>((normalize(surface_normal) * 0.5 + 0.5), 1.0);
    $$ else
        var diffuse_color = u_material.color;

        $$ if HAS_COLOR
            diffuse_color *= varyings.color;
        $$ endif

        {$ if HAS_MAP $}
            let tex_color = textureSample(t_map, s_map, varyings.map_uv);
            diffuse_color *= tex_color;
        {$ endif $}

    $$ endif


    // Apply opacity
    diffuse_color.a = diffuse_color.a * u_material.opacity;

    // alpha test
    {$ include 'alpha_test' $}

    let view = normalize(u_render_state.camera_position - varyings.world_position);

    // let face_direction = f32(is_front) * 2.0 - 1.0;

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

    // Init the reflected light. Defines diffuse and specular, both direct and indirect
    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    var geometry: GeometricContext;
    geometry.position = varyings.world_position;
    geometry.normal = normal;
    geometry.view_dir = view;

    {$ include 'light_phong_fragment' $}

    {$ include 'light_punctual_fragment' $}

    // Indirect Diffuse Light
    let ambient_color = u_environment.ambient_light.rgb;
    var irradiance = getAmbientLightIrradiance( ambient_color );
    // Light map (pre-baked lighting)
    $$ if HAS_LIGHT_MAP is defined
        let light_map_color = textureSample(t_light_map, s_light_map, varyings.light_map_uv ).rgb;
        irradiance += light_map_color * u_material.light_map_intensity;
    $$ endif

    // Process irradiance
    RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

    // Ambient occlusion
    $$ if HAS_AO_MAP is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let ambient_occlusion = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;

        reflected_light.indirect_diffuse *= ambient_occlusion;
    $$ endif

    // Combine direct and indirect light
    var out_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if HAS_EMISSIVE_MAP is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_color += emissive_color;

    return vec4<f32>(out_color, diffuse_color.a);
}
