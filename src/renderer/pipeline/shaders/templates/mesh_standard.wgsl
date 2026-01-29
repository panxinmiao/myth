{{ vertex_input_code }} 
{{ binding_code }}
{$ include 'vertex_output_def' $}

{$ include 'morph_pars' $}
{$ include 'light_common_pars' $}
{$ include 'light_punctual_pars' $}
{$ include 'bsdf/standard' $}


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
    var surface_normal = normalize(vec3<f32>(varyings.normal));
    $$ if FLAT_SHADING
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        surface_normal = normalize(cross(u, v));
    $$ endif

    $$ if COLOR_MODE == 'normal'
        var diffuse_color = vec4<f32>((normalize(surface_normal) * 0.5 + 0.5), 1.0);
    $$ else
        var diffuse_color = u_material.color;

        $$ if HAS_COLOR is defined
            diffuse_color *= varyings.color;
        $$ endif

        $$ if HAS_MAP is defined
            let tex_color = textureSample(t_map, s_map, varyings.map_uv);
            diffuse_color *= tex_color;
        $$ endif

    $$ endif


    // Apply opacity
    diffuse_color.a = diffuse_color.a * u_material.opacity;

    // alpha test
    {$ include 'alpha_test' $}

    let view = normalize(u_render_state.camera_position - varyings.world_position);  //todo orthographic camera

    let face_direction = f32(is_front) * 2.0 - 1.0;

    $$ if HAS_NORMAL_MAP is defined or USE_ANISOTROPY is defined
        $$ if HAS_TANGENT is defined
            var tbn = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
        $$ else
            $$ if HAS_NORMAL_MAP is defined
                let n_uv = varyings.normal_map_uv; 
            $$ elif HAS_CLEARCOAT_NORMAL_MAP is defined
                let n_uv = varyings.clearcoat_normal_map_uv;
            $$ elif HAS_MAP is defined
                let n_uv = varyings.map_uv;
            $$ else
                let n_uv = varyings.uv;
            $$ endif
            var tbn = getTangentFrame(view, surface_normal, n_uv );
        $$ endif

        tbn[0] = tbn[0] * face_direction;
        tbn[1] = tbn[1] * face_direction;
    $$ endif

    $$ if HAS_NORMAL_MAP is defined
        let normal_map = textureSample( t_normal_map, s_normal_map, varyings.normal_map_uv ) * 2.0 - 1.0;
        let map_n = vec3f(normal_map.xy * u_material.normal_scale, normal_map.z);
        let normal = normalize(tbn * map_n);
    $$ else
        let normal = surface_normal;
    $$ endif

    $$ if USE_CLEARCOAT is defined
        $$ if HAS_CLEARCOAT_NORMAL_MAP is defined
            $$ if HAS_TANGENT is defined
                var tbn_cc = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
            $$ else
                var tbn_cc = getTangentFrame( view, surface_normal, varyings.clearcoat_normal_map_uv );
            $$ endif

            tbn_cc[0] = tbn_cc[0] * face_direction;
            tbn_cc[1] = tbn_cc[1] * face_direction;

            var clearcoat_normal_map = textureSample(t_clearcoat_normal_map, s_clearcoat_normal_map, varyings.clearcoat_normal_map_uv ) * 2.0 - 1.0;
            let clearcoat_map_n = vec3f(clearcoat_normal_map.xy * u_material.clearcoat_normal_scale, clearcoat_normal_map.z);
            let clearcoat_normal = normalize(tbn_cc * clearcoat_map_n);
        $$ else
            let clearcoat_normal = surface_normal;
        $$ endif
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

    $$ if USE_CLEARCOAT is defined
        geometry.clearcoat_normal = clearcoat_normal;
    $$ endif

    {$ include 'light_physical_fragment' $}

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

    $$ if USE_IBL is defined
        $$ if USE_ANISOTROPY is defined
            let ibl_radiance = getIBLAnisotropyRadiance( view, normal, material.roughness, material.anisotropy_b, material.anisotropy );
        $$ else
            let ibl_radiance = getIBLRadiance( view, normal, material.roughness);
        $$ endif

        var clearcoat_ibl_radiance = vec3<f32>(0.0);
        $$ if USE_CLEARCOAT is defined
            clearcoat_ibl_radiance += getIBLRadiance( view, clearcoat_normal, material.clearcoat_roughness );
        $$ endif

        let ibl_irradiance = getIBLIrradiance( geometry.normal );
        RE_IndirectSpecular(ibl_radiance, ibl_irradiance, clearcoat_ibl_radiance, geometry, material, &reflected_light);
    $$ endif

    // Ambient occlusion
    $$ if HAS_AO_MAP is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let ambient_occlusion = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;

        reflected_light.indirect_diffuse *= ambient_occlusion;

        $$ if USE_CLEARCOAT is defined
            clearcoat_specular_indirect *= ambient_occlusion;
        $$ endif

        $$ if USE_SHEEN is defined
            sheen_specular_indirect *= ambient_occlusion;
        $$ endif

        $$ if USE_IBL is defined
            let dot_nv = saturate( dot( geometry.normal, geometry.view_dir ) );
            reflected_light.indirect_specular *= computeSpecularOcclusion( dot_nv, ambient_occlusion, material.roughness );
        $$ endif
    $$ endif

    // Combine direct and indirect light
    var out_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if HAS_EMISSIVE_MAP is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_color += emissive_color;

    $$ if USE_SHEEN is defined
        // Sheen energy compensation approximation calculation can be found at the end of
        // https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
        let sheen_energy_comp = 1.0 - 0.157 * max(material.sheen_color.r, max(material.sheen_color.g, material.sheen_color.b));
        out_color = out_color * sheen_energy_comp + (sheen_specular_direct + sheen_specular_indirect);
    $$ endif

    $$ if USE_CLEARCOAT is defined
        let dot_nv_cc = saturate(dot(clearcoat_normal, view));
        let fcc = F_Schlick( material.clearcoat_f0, material.clearcoat_f90, dot_nv_cc );
        out_color = out_color * (1.0 - material.clearcoat * fcc) + (clearcoat_specular_direct + clearcoat_specular_indirect) * material.clearcoat;
    $$ endif

    {$ include 'pbr_tone_mapping' $}

    return vec4<f32>(out_color, diffuse_color.a);
}
