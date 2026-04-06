// ── Skeletal Skinning (Pure Function Module) ────────────────────────────
//
// Applies bone matrix blending to transform vertex attributes from
// local space into skinned space.  Returns a typed struct so that the
// caller can pick exactly the fields it needs.
//
// Required global resources:
//   - st_skins: array<mat4x4<f32>>       (storage buffer)
//   - st_prev_skins: array<mat4x4<f32>>  (storage buffer, when HAS_VELOCITY_TARGET)

$$ if HAS_SKINNING and SUPPORT_SKINNING

/// Result of skeletal skinning computation.
struct SkinnedVertex {
    position: vec4<f32>,
    $$ if HAS_VELOCITY_TARGET is defined
    prev_position: vec4<f32>,
    $$ endif
    $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    normal: vec3<f32>,
    $$ endif
    $$ if HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    tangent: vec3<f32>,
    $$ endif
};

/// Computes skinned vertex attributes by blending bone matrices
/// weighted by the four most influential joints.
fn compute_skinned_vertex(
    local_pos: vec4<f32>,
    $$ if HAS_VELOCITY_TARGET is defined
    prev_local_pos: vec4<f32>,
    $$ endif
    $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    local_normal: vec3<f32>,
    $$ endif
    $$ if HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    object_tangent: vec3<f32>,
    $$ endif
    joints: vec4<u32>,
    weights: vec4<f32>,
) -> SkinnedVertex {
    var out: SkinnedVertex;

    let bone_mat =
        weights.x * st_skins[joints.x] +
        weights.y * st_skins[joints.y] +
        weights.z * st_skins[joints.z] +
        weights.w * st_skins[joints.w];

    out.position = bone_mat * local_pos;

    $$ if HAS_VELOCITY_TARGET is defined
    let prev_bone_mat =
        weights.x * st_prev_skins[joints.x] +
        weights.y * st_prev_skins[joints.y] +
        weights.z * st_prev_skins[joints.z] +
        weights.w * st_prev_skins[joints.w];

    out.prev_position = prev_bone_mat * prev_local_pos;
    $$ endif

    $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    let skin_normal_mat = mat3x3<f32>(
        bone_mat[0].xyz,
        bone_mat[1].xyz,
        bone_mat[2].xyz
    );

    out.normal = normalize(skin_normal_mat * local_normal);
    $$ endif

    $$ if HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
    // HAS_TANGENT implies HAS_NORMAL; skin_normal_mat is guaranteed.
    out.tangent = (skin_normal_mat * object_tangent).xyz;
    $$ endif

    return out;
}

$$ endif
