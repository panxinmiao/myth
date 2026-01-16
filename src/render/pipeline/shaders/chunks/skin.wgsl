    
$$ if use_skinning
    let skin_index = vec4<u32>(in.joints);
    let skin_weight = in.weights;

    let bone_mat = 
        skin_weight.x * u_skin[skin_index.x] +
        skin_weight.y * u_skin[skin_index.y] +
        skin_weight.z * u_skin[skin_index.z] +
        skin_weight.w * u_skin[skin_index.w];

    local_pos = bone_mat * local_pos;

    let skin_normal_mat = mat3x3<f32>(
        bone_mat[0].xyz,
        bone_mat[1].xyz,
        bone_mat[2].xyz
    );

    local_normal = normalize(skin_normal_mat * local_normal);

    $$ if use_tangent is defined
        object_tangent = (skin_normal_mat * object_tangent).xyz;
    $$ endif

$$ endif