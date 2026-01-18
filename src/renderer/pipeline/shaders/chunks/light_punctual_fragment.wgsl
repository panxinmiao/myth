for (var i = 0u; i < u_environment.num_lights; i ++ ) {
    let light = st_lights[i];
    let punctual_light = get_light_info( light, geometry );
    RE_Direct( punctual_light, geometry, material, &reflected_light );
}