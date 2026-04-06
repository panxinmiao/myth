// ── Skeletal Skinning (Inline Include) ──────────────────────────────────
//
// Applies bone matrix blending to transform vertex attributes from
// local space into skinned space.
//
// Required local variables (caller must declare before including):
//   - in.joints: vec4<u32>       (bone indices from vertex input)
//   - in.weights: vec4<f32>      (bone weights from vertex input)
//   - local_pos: vec4<f32>       (mutable, receives skinned position)
//   - local_normal: vec3<f32>    (mutable, when HAS_NORMAL, receives skinned normal)
//   - object_tangent: vec3<f32>  (mutable, when HAS_TANGENT, receives skinned tangent)
//   - prev_local_pos: vec4<f32>  (mutable, when HAS_VELOCITY_TARGET)
//
// Required global resources:
//   - st_skins: array<mat4x4<f32>>       (storage buffer)
//   - st_prev_skins: array<mat4x4<f32>>  (storage buffer, when HAS_VELOCITY_TARGET)

$$ if HAS_SKINNING and SUPPORT_SKINNING
    let skin_index = vec4<u32>(in.joints);
    let skin_weight = in.weights;

    let bone_mat = 
        skin_weight.x * st_skins[skin_index.x] +
        skin_weight.y * st_skins[skin_index.y] +
        skin_weight.z * st_skins[skin_index.z] +
        skin_weight.w * st_skins[skin_index.w];

    local_pos = bone_mat * local_pos;

    $$ if HAS_VELOCITY_TARGET is defined
    let prev_bone_mat =
        skin_weight.x * st_prev_skins[skin_index.x] +
        skin_weight.y * st_prev_skins[skin_index.y] +
        skin_weight.z * st_prev_skins[skin_index.z] +
        skin_weight.w * st_prev_skins[skin_index.w];

    prev_local_pos = prev_bone_mat * prev_local_pos;
    $$ endif

    $$ if HAS_NORMAL and not SHADOW_PASS and (OUTPUT_NORMAL or not IS_PREPASS)
    let skin_normal_mat = mat3x3<f32>(
        bone_mat[0].xyz,
        bone_mat[1].xyz,
        bone_mat[2].xyz
    );

    local_normal = normalize(skin_normal_mat * local_normal);
    $$ endif

    $$ if HAS_TANGENT and not SHADOW_PASS and not IS_PREPASS
        object_tangent = (skin_normal_mat * object_tangent).xyz;
    $$ endif

$$ endif
