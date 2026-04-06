// ── Morph Target Functions ───────────────────────────────────────────────
//
// Sparse-index + compact storage architecture for morph target blending.
// Provides fetch functions for position/normal/tangent deltas, and
// an `apply_morph_targets` function to blend all active targets.
//
// Required global resources:
//   - u_morph_targets  (uniform: count, vertex_count, weights, indices, prev_weights)
//   - st_morph_positions (storage buffer)
//   - st_morph_normals   (storage buffer, when HAS_MORPH_NORMALS)
//   - st_morph_tangents  (storage buffer, when HAS_MORPH_TANGENTS)

$$ if HAS_MORPH_TARGETS

fn fetch_morph_position(vertex_index: u32, target_index: u32) -> vec3<f32> {
    let start_idx = (target_index * u_morph_targets.vertex_count + vertex_index) * 3u;
    return vec3<f32>(
        st_morph_positions[start_idx],
        st_morph_positions[start_idx + 1u],
        st_morph_positions[start_idx + 2u]
    );
}

$$ if HAS_MORPH_NORMALS and not SHADOW_PASS
fn fetch_morph_normal(vertex_index: u32, target_index: u32) -> vec3<f32> {
    let start_idx = (target_index * u_morph_targets.vertex_count + vertex_index) * 3u;
    return vec3<f32>(
        st_morph_normals[start_idx],
        st_morph_normals[start_idx + 1u],
        st_morph_normals[start_idx + 2u]
    );
}
$$ endif

$$ if HAS_MORPH_TANGENTS and not SHADOW_PASS
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
