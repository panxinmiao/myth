// Morph Target 变形 Shader Chunk
// 使用稀疏索引 + 紧凑存储架构

$$ if HAS_MORPH_TARGETS

/// 从紧凑 Storage Buffer 获取 Morph Position 位移
/// 布局: [ Target 0 所有顶点 | Target 1 所有顶点 | ... ]
/// 每个顶点存储 3 个连续的 f32
fn fetch_morph_position(vertex_index: u32, target_index: u32) -> vec3<f32> {
    // 索引 = (TargetID * 顶点总数 + 顶点ID) * 3
    let start_idx = (target_index * u_morph_targets.vertex_count + vertex_index) * 3u;
    return vec3<f32>(
        st_morph_positions[start_idx],
        st_morph_positions[start_idx + 1u],
        st_morph_positions[start_idx + 2u]
    );
}

$$ if HAS_MORPH_NORMALS
/// 从紧凑 Storage Buffer 获取 Morph Normal 位移
fn fetch_morph_normal(vertex_index: u32, target_index: u32) -> vec3<f32> {
    let start_idx = (target_index * u_morph_targets.vertex_count + vertex_index) * 3u;
    return vec3<f32>(
        st_morph_normals[start_idx],
        st_morph_normals[start_idx + 1u],
        st_morph_normals[start_idx + 2u]
    );
}
$$ endif

$$ if HAS_MORPH_TANGENTS
/// 从紧凑 Storage Buffer 获取 Morph Tangent 位移
fn fetch_morph_tangent(vertex_index: u32, target_index: u32) -> vec3<f32> {
    let start_idx = (target_index * u_morph_targets.vertex_count + vertex_index) * 3u;
    return vec3<f32>(
        st_morph_tangents[start_idx],
        st_morph_tangents[start_idx + 1u],
        st_morph_tangents[start_idx + 2u]
    );
}
$$ endif


$$ endif
