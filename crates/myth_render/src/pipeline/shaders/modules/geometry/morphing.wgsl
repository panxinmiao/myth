// ── Morph Target Module (Pure Function Library) ─────────────────────────
//
// Sparse-index + compact storage architecture for morph target blending.
// Provides fetch functions for position/normal/tangent deltas, and
// an `apply_morph_targets` pure function to blend all active targets.
//
// Required global resources:
//   - u_morph_targets  (uniform: count, vertex_count, weights, indices, prev_weights)
//   - st_morph_positions (storage buffer)
//   - st_morph_normals   (storage buffer, when HAS_MORPH_NORMALS)
//   - st_morph_tangents  (storage buffer, when HAS_MORPH_TANGENTS)

$$ if HAS_MORPH_TARGETS

// ── Fetch Helpers ───────────────────────────────────────────────────────

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

// ── Result Struct ───────────────────────────────────────────────────────

/// Result of morph target blending.
struct MorphedVertex {
    position: vec3<f32>,
    $$ if HAS_VELOCITY_TARGET is defined
    prev_position: vec3<f32>,
    $$ endif
    $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    normal: vec3<f32>,
    $$ endif
    $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    tangent: vec3<f32>,
    $$ endif
};

// ── Pure Function ───────────────────────────────────────────────────────

/// Blends all active morph targets into the given vertex attributes and
/// returns the result as a typed struct.
fn apply_morph_targets(
    vertex_index: u32,
    position: vec3<f32>,
    $$ if HAS_VELOCITY_TARGET is defined
    prev_position: vec3<f32>,
    $$ endif
    $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    in_normal: vec3<f32>,
    $$ endif
    $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    in_tangent: vec3<f32>,
    $$ endif
) -> MorphedVertex {
    var out: MorphedVertex;
    out.position = position;
    $$ if HAS_VELOCITY_TARGET is defined
    out.prev_position = prev_position;
    $$ endif
    $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    out.normal = in_normal;
    $$ endif
    $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    out.tangent = in_tangent;
    $$ endif

    let active_count = u_morph_targets.count;

    for (var i = 0u; i < active_count; i++) {
        let vec_idx = i / 4u;
        let component = i % 4u;
        let weight = u_morph_targets.weights[vec_idx][component];
        let target_idx = u_morph_targets.indices[vec_idx][component];

        let morph_pos = fetch_morph_position(vertex_index, target_idx);
        out.position += morph_pos * weight;

        $$ if HAS_VELOCITY_TARGET is defined
        let prev_weight = u_morph_targets.prev_weights[vec_idx][component];
        out.prev_position += morph_pos * prev_weight;
        $$ endif

        $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
        out.normal += fetch_morph_normal(vertex_index, target_idx) * weight;
        $$ endif

        $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
        out.tangent += fetch_morph_tangent(vertex_index, target_idx) * weight;
        $$ endif
    }

    $$ if HAS_MORPH_NORMALS and HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    out.normal = normalize(out.normal);
    $$ endif
    $$ if HAS_MORPH_TANGENTS and HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    out.tangent = normalize(out.tangent);
    $$ endif

    return out;
}

$$ endif
