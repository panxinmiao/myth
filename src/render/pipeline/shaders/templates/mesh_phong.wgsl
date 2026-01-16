{{ vertex_input_code }} 

{{ binding_code }}

{$ include 'light_common' $}
{$ include 'bsdf/phong' $}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    @location({{ loc.next() }}) geometry_normal: vec3<f32>,
    $$ endif
    $$ if use_vertex_color
    @location({{ loc.next() }}) color: vec4<f32>,
    $$ endif
    {$ include 'uv_vetex_output' $}
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var local_pos = vec4<f32>(in.position, 1.0);
    var local_normal = in.normal;

    {$ include 'skin' $}

    let world_pos = u_model.world_matrix * local_pos;


    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if use_vertex_color
        out.color = in.color;
    $$ endif

    out.uv = in.uv;
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    {$ include 'uv' $}
    return out;
}

@fragment
fn fs_main(varyings: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var normal = normalize(varyings.normal);
    $$ if flat_shading
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        normal = normalize(cross(u, v));
    $$ else
        normal = select(-normal, normal, is_front);
    $$ endif


    $$ if color_mode == 'normal'
        var diffuse_color = vec4<f32>((normalize(surface_normal) * 0.5 + 0.5), 1.0);
    $$ else
        var diffuse_color = u_material.color;

        $$ if use_vertex_color
            diffuse_color *= varyings.color;
        $$ endif

        {$ if use_map $}
            let tex_color = textureSample(t_map, s_map, varyings.uv);
            diffuse_color *= tex_color;
        {$ endif $}

    $$ endif


    // Apply opacity
    diffuse_color.a = diffuse_color.a * u_material.opacity;

    // todo alpha test

    let view = normalize(u_render_state.camera_position - varyings.world_position);

    // let face_direction = f32(is_front) * 2.0 - 1.0;

    $$ if use_normal_map is defined

        let tbn = getTangentFrame(view, normal, varyings.normal_map_uv );

        let normal_map = textureSample( t_normal_map, s_normal_map, varyings.normal_map_uv ) * 2.0 - 1.0;
        let map_n = vec3f(normal_map.xy * u_material.normal_scale, normal_map.z);
        normal = normalize(tbn * map_n);
    $$ endif


    $$ if use_specular_map is defined
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

    var material: BlinnPhongMaterial;
    material.diffuse_color = diffuse_color.rgb;
    material.specular_color = u_material.specular.rgb;
    material.specular_shininess = u_material.shininess;
    material.specular_strength = specular_strength;

    {$ include 'light_punctual' $}

    // Indirect Diffuse Light
    let ambient_color = u_environment.ambient_light.rgb;
    var irradiance = getAmbientLightIrradiance( ambient_color );
    // Light map (pre-baked lighting)
    $$ if use_light_map is defined
        let light_map_color = textureSample(t_light_map, s_light_map, varyings.light_map_uv ).rgb;
        irradiance += light_map_color * u_material.light_map_intensity;
    $$ endif

    // Process irradiance
    RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

    // Ambient occlusion
    $$ if use_ao_map is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let ambient_occlusion = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;

        reflected_light.indirect_diffuse *= ambient_occlusion;
    $$ endif

    // Combine direct and indirect light
    var out_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;

    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if use_emissive_map is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_color += emissive_color;

    return vec4<f32>(out_color, diffuse_color.a);
}
