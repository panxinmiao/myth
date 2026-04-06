// ── Punctual Light Loop (Inline Include) ────────────────────────────────
//
// Iterates all active lights, applies shadow sampling, and accumulates
// direct lighting via RE_Direct.
//
// Required local variables:
//   - geometry: GeometricContext
//   - material: PhysicalMaterial | BlinnPhongMaterial
//   - reflected_light: ReflectedLight (mutable)
//
// Required global resources:
//   - u_environment.num_lights, st_lights[]
//   - u_render_state.view_matrix (for cascade selection)
//   - t_shadow_map_2d_array, t_shadow_map_cube_array, s_shadow_map_compare
//
// Required includes:
//   - lighting/punctual_pars.wgsl  (get_light_info)
//   - lighting/shadow.wgsl         (sample_shadow, sample_point_shadow)
//   - materials/bsdf_*.wgsl        (RE_Direct)

for (var i = 0u; i < u_environment.num_lights; i ++ ) {
    let light = st_lights[i];
    var punctual_light = get_light_info( light, geometry );

    $$ if HAS_SHADOWS and RECEIVE_SHADOWS
    if (punctual_light.visible) {
        let shadow_pos = geometry.position + geometry.normal * light.shadow_normal_bias;
        var shadow = 1.0;

        if (light.light_type == 1u && light.point_shadow_index >= 0) {
            shadow = sample_point_shadow(
                light.position,
                shadow_pos,
                light.range,
                light.point_shadow_index,
                light.shadow_bias,
            );
        } else if (light.shadow_layer_index >= 0) {
            if (light.light_type == 0u && light.cascade_count > 1u) {
                let view_pos = u_render_state.view_matrix * vec4<f32>(geometry.position, 1.0);
                let view_depth = -view_pos.z;

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
                shadow = sample_shadow(
                    light.shadow_matrices[0],
                    light.shadow_layer_index,
                    shadow_pos,
                    light.shadow_bias
                );
            }
        }

        punctual_light.color *= shadow;
    }
    $$ endif

    RE_Direct( punctual_light, geometry, material, &reflected_light );
}
