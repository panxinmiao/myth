// ── Morph Target Application (Inline Include) ──────────────────────────
//
// Blends all active morph targets into local vertex attributes.
//
// Required local variables (caller must declare before including):
//   - vertex_index: u32          (builtin vertex index)
//   - local_position: vec3<f32>  (mutable, receives morph position delta)
//   - local_normal: vec3<f32>    (mutable, when HAS_NORMAL, receives morph normal delta)
//   - object_tangent: vec3<f32>  (mutable, when HAS_TANGENT, receives morph tangent delta)
//   - prev_local_position: vec3<f32> (mutable, when HAS_VELOCITY_TARGET)
//
// Required includes (must be included before this file):
//   - geometry/morph_pars.wgsl   (fetch_morph_position/normal/tangent)

$$ if HAS_MORPH_TARGETS
    let active_count = u_morph_targets.count;

    for (var i = 0u; i < active_count; i++) {
        let vec_idx = i / 4u;
        let component = i % 4u;
        let weight = u_morph_targets.weights[vec_idx][component];
        let target_idx = u_morph_targets.indices[vec_idx][component];

        let morph_pos = fetch_morph_position(vertex_index, target_idx);
        local_position += morph_pos * weight;

        $$ if HAS_VELOCITY_TARGET is defined
        let prev_weight = u_morph_targets.prev_weights[vec_idx][component];
        prev_local_position += morph_pos * prev_weight;
        $$ endif

        $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
        local_normal += fetch_morph_normal(vertex_index, target_idx) * weight;
        $$ endif

        $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
        object_tangent += fetch_morph_tangent(vertex_index, target_idx) * weight;
        $$ endif
    }

    $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    local_normal = normalize(local_normal);
    $$ endif
    $$ if HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    object_tangent = normalize(object_tangent);
    $$ endif

$$ endif
