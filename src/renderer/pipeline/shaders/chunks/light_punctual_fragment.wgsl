for (var i = 0u; i < u_environment.num_lights; i ++ ) {
    let light = st_lights[i];
    var punctual_light = get_light_info( light, geometry );

    $$ if HAS_SHADOWS and RECEIVE_SHADOWS
    if (punctual_light.visible && light.shadow_layer_index >= 0) {
        // Apply Normal Bias: offset world position along the surface normal
        let shadow_pos = geometry.position + geometry.normal * light.shadow_normal_bias;
        var shadow = 1.0;

        if (light.light_type == 0u && light.cascade_count > 1u) {
            // Directional light with Cascaded Shadow Maps
            let view_pos = u_render_state.view_matrix * vec4<f32>(geometry.position, 1.0);
            let view_depth = -view_pos.z; // RH convention: positive depth

            // Select cascade level based on view depth vs split distances
            var cascade_idx = light.cascade_count - 1u;
            if (view_depth < light.cascade_splits.x) {
                cascade_idx = 0u;
            } else if (light.cascade_count > 1u && view_depth < light.cascade_splits.y) {
                cascade_idx = 1u;
            } else if (light.cascade_count > 2u && view_depth < light.cascade_splits.z) {
                cascade_idx = 2u;
            }

            let layer = light.shadow_layer_index + i32(cascade_idx);
            let matrix = light.shadow_matrices[cascade_idx];
            shadow = sample_shadow(matrix, layer, shadow_pos, light.shadow_bias);
        } else {
            // Spot light or single-cascade directional
            shadow = sample_shadow(
                light.shadow_matrices[0],
                light.shadow_layer_index,
                shadow_pos,
                light.shadow_bias
            );
        }

        punctual_light.color *= shadow;
    }
    $$ endif

    RE_Direct( punctual_light, geometry, material, &reflected_light );
}