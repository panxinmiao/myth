for (var i = 0u; i < u_environment.num_lights; i ++ ) {
    let light = st_lights[i];
    var punctual_light = get_light_info( light, geometry );

    $$ if USE_SHADOWS is defined and RECEIVE_SHADOWS is defined
    if (punctual_light.visible && light.shadow_layer_index >= 0) {
        let shadow = sample_shadow(light.shadow_matrix, light.shadow_layer_index, geometry.position);
        punctual_light.color *= shadow;
    }
    $$ endif

    RE_Direct( punctual_light, geometry, material, &reflected_light );
}