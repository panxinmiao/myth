$$ if HAS_SHADOWS and RECEIVE_SHADOWS
fn sample_shadow(shadow_matrix: mat4x4<f32>, shadow_layer_index: i32, world_position: vec3<f32>, bias: f32) -> f32 {
    if (shadow_layer_index < 0) {
        return 1.0;
    }

    let shadow_clip = shadow_matrix * vec4<f32>(world_position, 1.0);
    if (abs(shadow_clip.w) <= EPSILON) {
        return 1.0;
    }

    let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    let shadow_uv = vec2<f32>(
        shadow_ndc.x * 0.5 + 0.5,
        shadow_ndc.y * -0.5 + 0.5
    );

    if (shadow_uv.x <= 0.0 || shadow_uv.x >= 1.0 || shadow_uv.y <= 0.0 || shadow_uv.y >= 1.0) {
        return 1.0;
    }

    let shadow_depth = shadow_ndc.z;
    if (shadow_depth <= 0.0 || shadow_depth >= 1.0) {
        return 1.0;
    }

    // PCF (Percentage Closer Filtering)
    let biased_depth = saturate(shadow_depth - bias);
    var shadow_sum = 0.0;

    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth);
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(1, 0));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(0, 1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(1, 1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-1, 0));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(0, -1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-1, -1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-1, 1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(1, -1));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(2, 0));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(0, 2));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(2, 2));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-2, 0));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(0, -2));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-2, -2));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(-2, 2));
    shadow_sum += textureSampleCompareLevel(t_shadow_map_2d_array, s_shadow_map_compare, shadow_uv, shadow_layer_index, biased_depth, vec2<i32>(2, -2));

    return shadow_sum / 17.0;
}
$$ endif