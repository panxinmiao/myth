fn get_light_info( light: Struct_lights, geometry: GeometricContext ) -> IncidentLight {
    let light_type = light.light_type;
    var light_info: IncidentLight;

    light_info.visible = true;
    light_info.color = light.color.rgb * light.intensity;

    if ( light_type == 0u ) {
        light_info.direction = -light.direction.xyz;
    } else if ( light_type == 1u ) {
        let i_vector = light.position - geometry.position;
        light_info.direction = normalize(i_vector);
        let light_distance = length(i_vector);
        light_info.color *= getDistanceAttenuation( light_distance, light.range, light.decay );
        light_info.visible = any(light_info.color != vec3<f32>(0.0));
    } else if ( light_type == 2u ) {
        let i_vector = light.position - geometry.position;
        light_info.direction = normalize(i_vector);
        let angle_cos = dot(light_info.direction, -light.direction.xyz);
        let spot_attenuation = getSpotAttenuation(light.outer_cone_cos, light.inner_cone_cos, angle_cos);
        if ( spot_attenuation > 0.0 ) {
            let light_distance = length( i_vector );
            light_info.color = light.color.rgb * light.intensity;
            light_info.color *= spot_attenuation;
            light_info.color *= getDistanceAttenuation( light_distance, light.range, light.decay );
            light_info.visible = any(light_info.color != vec3<f32>(0.0));
        } else {
            light_info.color = vec3<f32>( 0.0 );
            light_info.visible = false;
        }

    }
    return light_info;
}