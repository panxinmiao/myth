    $$ if HAS_MORPH_TARGETS
    let morph_result = apply_morph_targets(vertex_index, in.position, in.normal);
    local_pos = vec4<f32>(morph_result.position, 1.0);
    local_normal = morph_result.normal;
    $$ endif