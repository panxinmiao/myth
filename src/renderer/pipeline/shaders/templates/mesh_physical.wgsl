{{ vertex_input_code }} 
{{ binding_code }}
{$ include 'vertex_output_def' $}

{$ include 'morph_pars' $}
{$ include 'light_common_pars' $}
{$ include 'light_punctual_pars' $}
{$ include 'shadow_pars' $}
{$ include 'bsdf/physical' $}

{$ include 'iridescence' $}

$$ if USE_TRANSMISSION is defined
    {$ include 'transmission' $}
$$ endif

// SSAO texture (Group 3, Binding 2) — always bound.
// When SSAO is disabled, this is a 1×1 white texture (AO = 1.0).
@group(3) @binding(1) var s_screen_sampler: sampler;
@group(3) @binding(2) var t_ssao: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = in.position;

    $$ if HAS_NORMAL is defined
    var local_normal = in.normal;
    $$ endif

    $$ if HAS_TANGENT is defined
    var object_tangent = in.tangent.xyz;
    $$ endif

    {$ include 'morph_vertex' $}

    var local_pos = vec4<f32>(local_position, 1.0);

    {$ include 'skin_vertex' $}

    let world_pos = u_model.world_matrix * local_pos;

    let clip_pos = u_render_state.view_projection * world_pos;

    out.position = clip_pos;
    out.clip_position = clip_pos;
    out.world_position = world_pos.xyz / world_pos.w;

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

    $$ if HAS_TANGENT
    let v_tangent = normalize(( u_model.world_matrix  * vec4f(object_tangent, 0.0) ).xyz);
    let v_bitangent = normalize(cross(out.normal, v_tangent) * in.tangent.w);
    out.v_tangent = vec3<f32>(v_tangent);
    out.v_bitangent = vec3<f32>(v_bitangent);
    $$ endif

    {$ include 'uv_vertex' $}
    return out;
}


struct FragmentOutput {
    @location(0) color: vec4<f32>,
    $$ if USE_SSS
    @location(1) specular: vec4<f32>,
    $$ endif
};

@fragment
fn fs_main(varyings: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {

    let face_direction = f32(is_front) * 2.0 - 1.0;

    $$ if FLAT_SHADING or HAS_NORMAL is not defined
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        var surface_normal = normalize(cross(u, v));
    $$ else
        var surface_normal = normalize(vec3<f32>(varyings.normal));
        surface_normal = surface_normal * face_direction;
    $$ endif

    $$ if COLOR_MODE == 'normal'
        var diffuse_color = vec4<f32>((normalize(surface_normal) * 0.5 + 0.5), 1.0);
    $$ else
        var diffuse_color = u_material.color;

        $$ if HAS_COLOR
            diffuse_color *= varyings.color;
        $$ endif

        $$ if HAS_MAP
            let tex_color = textureSample(t_map, s_map, varyings.map_uv);
            diffuse_color *= tex_color;
        $$ endif

    $$ endif

    // Apply opacity
    var opacity = diffuse_color.a * u_material.opacity;

    // alpha test
    {$ include 'alpha_test' $}

    let view = normalize(u_render_state.camera_position - varyings.world_position);  //todo orthographic camera

    $$ if HAS_NORMAL_MAP is defined or USE_ANISOTROPY is defined
        $$ if HAS_TANGENT is defined
            var tbn = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
        $$ else
            $$ if HAS_NORMAL_MAP is defined
                let n_uv = varyings.normal_map_uv; 
            $$ elif HAS_CLEARCOAT_NORMAL_MAP is defined
                let n_uv = varyings.clearcoat_normal_map_uv;
            $$ elif HAS_MAP_UV is defined
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

    var ambient_occlusion = 1.0;
    $$ if HDR and USE_SSAO
    // Sample screen-space AO
    let screen_ndc = varyings.clip_position.xy / varyings.clip_position.w;

    // let screen_clip = u_render_state.view_projection * vec4<f32>(varyings.world_position, 1.0);
    // let screen_ndc = screen_clip.xyz / screen_clip.w;
    let screen_uv = vec2<f32>(
        screen_ndc.x * 0.5 + 0.5,
        screen_ndc.y * -0.5 + 0.5
    );


    ambient_occlusion = textureSampleLevel(t_ssao, s_screen_sampler, screen_uv, 0.0).r;
    $$ endif

    $$ if HAS_AO_MAP is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let material_ao = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;
        ambient_occlusion *= material_ao;
    $$ endif

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


    var total_diffuse = reflected_light.direct_diffuse + reflected_light.indirect_diffuse;
    var total_specular = reflected_light.direct_specular + reflected_light.indirect_specular;

    $$ if USE_TRANSMISSION is defined
        let pos = varyings.world_position;
        let v = normalize(u_render_state.camera_position - pos);
        let n = surface_normal;
        let model_matrix = u_model.world_matrix;
        // let view_matrix = u_render_state.view_matrix;
        let view_projection_matrix = u_render_state.view_projection;

        let transmitted = getIBLVolumeRefraction(
            n, v, material.roughness, material.diffuse_color, material.specular_color, material.specular_f90,
            pos, model_matrix, view_projection_matrix, material.dispersion, material.ior, material.thickness,
            material.attenuation_color, material.attenuation_distance );

        material.transmission_alpha = mix( material.transmission_alpha, transmitted.a, material.transmission );

        total_diffuse = mix( total_diffuse, transmitted.rgb, material.transmission );
    $$ endif


    // Combine direct and indirect light
    // var out_color = total_diffuse + total_specular;

    // 1. 初始状态：分离漫反射与基础高光
    var out_diffuse = total_diffuse;
    var out_specular = total_specular;

    // 2. 自发光 (Emissive)：通常视作从物体内部或表面发出的光，归入 Diffuse 以参与 SSS 模糊
    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if HAS_EMISSIVE_MAP is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_diffuse += emissive_color;

    // 3. 绒毛层 (Sheen)：纯高光附加，同时吸收底层能量
    $$ if USE_SHEEN is defined
        // Sheen energy compensation approximation calculation can be found at the end of
        // https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
        let sheen_energy_comp = 1.0 - 0.157 * max(material.sheen_color.r, max(material.sheen_color.g, material.sheen_color.b));

        out_diffuse *= sheen_energy_comp;
        out_specular *= sheen_energy_comp;

        // 增加高光：绒毛反光只加到 specular 通道
        out_specular += (sheen_specular_direct + sheen_specular_indirect);

        // out_color = out_color * sheen_energy_comp + (sheen_specular_direct + sheen_specular_indirect);
    $$ endif

    // 4. 清漆层 (Clearcoat)：最高层的高光，吸收所有底层的能量
    $$ if USE_CLEARCOAT is defined
        let dot_nv_cc = saturate(dot(clearcoat_normal, view));
        let fcc = F_Schlick( material.clearcoat_f0, material.clearcoat_f90, dot_nv_cc );

        let clearcoat_attenuation = 1.0 - material.clearcoat * fcc;

        // 能量衰减：同时作用于底层的所有 diffuse 和 specular
        out_diffuse *= clearcoat_attenuation;
        out_specular *= clearcoat_attenuation;

        // 增加高光：清漆反光只加到 specular 通道
        out_specular += (clearcoat_specular_direct + clearcoat_specular_indirect) * material.clearcoat;

       // out_color = out_color * (1.0 - material.clearcoat * fcc) + (clearcoat_specular_direct + clearcoat_specular_indirect) * material.clearcoat;
    $$ endif

    $$ if OPAQUE is defined
        opacity = 1.0;
    $$ endif

    $$ if USE_TRANSMISSION is defined
        opacity *= material.transmission_alpha;
    $$ endif

    var out: FragmentOutput;

    $$ if USE_SCREEN_SPACE_FEATUREs
        out.specular = vec4<f32>(total_specular, material.roughness);
        if (u_material.sss_id != 0u) {
            // SSSSS 材质：分离漫反射与高光
            out.color = vec4<f32>(out_diffuse, opacity);
            out.specular = vec4<f32>(out_specular, 1.0); 
        } else {
            // 普通材质：保持合并，高光纯黑
            out.color = vec4<f32>(out_diffuse + out_specular, opacity);
            out.specular = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    $$ else
        var out_color = out_diffuse + out_specular;
        $$ if not HDR
            // builtin tone mapping for simple SDR output
            {$ include 'pbr_tone_mapping' $}
        $$ endif
        out.color = vec4<f32>(out_color, opacity);
    $$ endif

    return out;

    // return vec4<f32>(out_color, opacity);
}
