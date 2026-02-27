{$ include 'full_screen_vertex.wgsl' $}

struct SssProfileData {
    scatter_color: vec3<f32>,
    scatter_radius: f32,
};

// 纹理绑定
@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_normal: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var<storage, read> u_profiles: array<SssProfileData>;
@group(0) @binding(4) var s_sampler: sampler;
@group(0) @binding(5) var t_feature_id: texture_2d<u32>; // 纯无符号整数纹理
@group(0) @binding(6) var t_specular: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_coord = vec2<i32>(in.position.xy);
    let center_color = textureLoad(t_color, center_coord, 0);
    
    let sss_id = textureLoad(t_feature_id, center_coord, 0).r;
    if (sss_id == 0u) { 
        return center_color; 
    }
    
    let profile = u_profiles[sss_id];
    let scatter_color = profile.scatter_color;
    // 限制最大物理半径，防止参数爆炸导致画面崩坏
    let max_radius = clamp(profile.scatter_radius, 0.001, 0.1); 
    
    let center_depth = textureLoad(t_depth, center_coord, 0);
    let center_packed = textureLoad(t_normal, center_coord, 0);
    let center_normal = normalize(center_packed.xyz * 2.0 - 1.0);
    
    // ==========================================
    // 1. 物理透视与像素半径计算
    // ==========================================
    // 假设使用 Reverse-Z 且无投影矩阵参数时的经验公式
    // 如果发现远处的脸太糊，近处的脸太清晰，调整 3000.0 这个视口缩放常数
    let pixel_radius = max_radius * center_depth * 3000.0; 
    
    var final_color = vec3<f32>(0.0);
    var total_weight = vec3<f32>(0.0); 
    
    // 采样半步数：6 意味着一共采 13 个点 (6*2 + 1)
    let steps = 6; 
    // 保证步长至少为 1 像素，否则会导致无意义的原地重复采样
    let step_size = max(pixel_radius / f32(steps), 1.0);
    
    // 高斯分布的 Sigma 标准差系数 (控制模糊的"胖瘦")
    let sigma = f32(steps) / 2.0; 

    $$ if SSSSS_VERTICAL_PASS is defined
        let blur_dir = vec2<f32>(0.0, 1.0);
    $$ else
        let blur_dir = vec2<f32>(1.0, 0.0);
    $$ endif
    
    for (var i = -steps; i <= steps; i++) {
        let offset = vec2<i32>(blur_dir * (f32(i) * step_size));
        let sample_coord = center_coord + offset;

        // 【材质保护】防止皮肤糊到金属上
        let sample_sss_id = textureLoad(t_feature_id, sample_coord, 0).r;
        if (sample_sss_id != sss_id) {
            continue; 
        }
        
        // ==========================================
        // 2. 双边滤波惩罚 (Bilateral Penalties)
        // ==========================================
        let sample_depth = textureLoad(t_depth, sample_coord, 0);
        let sample_packed = textureLoad(t_normal, sample_coord, 0);
        let sample_normal = normalize(sample_packed.xyz * 2.0 - 1.0);
        
        // 深度惩罚：使用平方差，使断崖边缘切断更干脆
        let depth_diff = abs(center_depth - sample_depth);
        let depth_weight = exp(-(depth_diff * depth_diff) * 50000.0); 
        
        // 法线惩罚：平滑表面融合，直角边缘切断
        let normal_dot = max(dot(center_normal, sample_normal), 0.0);
        // acos(dot) 能更真实地反映角度差，但运算偏贵。这里用高次幂近似
        let normal_weight = pow(normal_dot, 8.0); 
        
        // ==========================================
        // 3. 核心：空间高斯与 RGB 颜色分离 (Color Shift)
        // ==========================================
        // 标准正态高斯衰减
        let x = f32(i);
        let spatial_weight = exp(-(x * x) / (2.0 * sigma * sigma)); 
        
        // 距离越远，白光被吸收得越多，只剩下 scatter_color (透射色)
        // 距离越近(i接近0)，越保留完整的白光 (vec3(1.0))
        let distance_ratio = abs(x) / f32(steps);
        let color_shift = mix(vec3<f32>(1.0), scatter_color, distance_ratio);
        
        // 综合权重（这是一个 vec3，R G B 通道权重独立！）
        let weight = color_shift * spatial_weight * depth_weight * normal_weight;
        
        final_color += textureLoad(t_color, sample_coord, 0).rgb * weight;
        total_weight += weight;
    }

    // 将积累的颜色除以积累的 rgb 权重
    let final_diffuse = final_color / total_weight;

    $$ if SSSSS_VERTICAL_PASS is defined
        // 高光部分
        let crisp_specular = textureLoad(t_specular, center_coord, 0).rgb;
        // 合成最终颜色：漫反射 + 保留高光（不模糊）
        return vec4<f32>(final_diffuse + crisp_specular, center_color.a);
    $$ else
        // 水平 Pass：仅输出模糊后的漫反射
        return vec4<f32>(final_diffuse, center_color.a);
    $$ endif
    
    // 合成最终颜色：漫反射 + 保留高光（不模糊）
    // return vec4<f32>(final_diffuse + crisp_specular, center_color.a);
}
