$$ if HAS_MORPH_TARGETS
    let active_count = u_morph_targets.count;
    
    // 稀疏循环：只遍历激活的 targets
    // weights 和 indices 打包在 vec4 中: weights[i/4][i%4]
    for (var i = 0u; i < active_count; i++) {
        let vec_idx = i / 4u;
        let component = i % 4u;
        let weight = u_morph_targets.weights[vec_idx][component];
        let target_idx = u_morph_targets.indices[vec_idx][component];
        
        // 应用 Position 位移
        local_position += fetch_morph_position(vertex_index, target_idx) * weight;
        
        $$ if HAS_MORPH_NORMALS and HAS_NORMAL
        // 应用 Normal 位移
        local_normal += fetch_morph_normal(vertex_index, target_idx) * weight;
        $$ endif

        // 应用 Tangent 位移
        $$ if HAS_MORPH_TANGENTS and HAS_TANGENT
        object_tangent += fetch_morph_tangent(vertex_index, target_idx) * weight;
        $$ endif
    }
    
    // local_pos = vec4<f32>(local_pos, 1.0);
    // 归一化法线
    $$ if HAS_NORMAL
    local_normal = normalize(local_normal);
    $$ endif
    $$ if HAS_TANGENT
    object_tangent = normalize(object_tangent);
    $$ endif

$$ endif