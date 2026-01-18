// Morph Target 变形 Shader Chunk
// 使用稀疏索引 + 紧凑存储架构

$$ if use_morphing

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

$$ if use_morph_normals
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

$$ if use_morph_tangents
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

/// 应用 Morph Target 变形
/// 使用稀疏循环遍历激活的 targets
fn apply_morph_targets(
    vertex_index: u32,
    base_position: vec3<f32>,
    base_normal: vec3<f32>
) -> MorphResult {
    var result: MorphResult;
    result.position = base_position;
    result.normal = base_normal;
    
    let active_count = u_morph_targets.count;
    
    // 稀疏循环：只遍历激活的 targets
    // weights 和 indices 打包在 vec4 中: weights[i/4][i%4]
    for (var i = 0u; i < active_count; i++) {
        let vec_idx = i / 4u;
        let component = i % 4u;
        let weight = u_morph_targets.weights[vec_idx][component];
        let target_idx = u_morph_targets.indices[vec_idx][component];
        
        // 应用 Position 位移
        result.position += fetch_morph_position(vertex_index, target_idx) * weight;
        
        $$ if use_morph_normals
        // 应用 Normal 位移
        result.normal += fetch_morph_normal(vertex_index, target_idx) * weight;
        $$ endif
    }
    
    // 归一化法线
    result.normal = normalize(result.normal);
    
    return result;
}

struct MorphResult {
    position: vec3<f32>,
    normal: vec3<f32>,
}

$$ endif
