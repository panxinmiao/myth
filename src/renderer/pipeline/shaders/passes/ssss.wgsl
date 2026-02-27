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

// 模糊方向：水平为 vec2(1.0, 0.0), 垂直为 vec2(0.0, 1.0)
@group(1) @binding(0) var<uniform> blur_dir: vec2<f32>; 

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_coord = vec2<i32>(in.position.xy);
    let center_color = textureLoad(t_color, center_coord, 0);
    
    // 1. 获取准确的 8-bit ID
    let sss_id = textureLoad(t_feature_id, center_coord, 0).r;
    
    // 防御性判定 (虽然硬件 Stencil 已经剔除了大部分，但防止边缘精度误差)
    if (sss_id == 0u) { 
        return center_color; 
    }
    
    // 2. 查表获取物理参数
    let profile = u_profiles[sss_id];
    let scatter_color = profile.scatter_color;
    let max_radius = profile.scatter_radius;
    
    // 3. 读取中心深度和法线
    let center_depth = textureLoad(t_depth, center_coord, 0);
    let center_packed = textureLoad(t_normal, center_coord, 0);
    let center_normal = normalize(center_packed.xyz * 2.0 - 1.0);
    
    // 根据深度计算屏幕空间的模糊半径 (透视除法原理)
    // 真实世界中越远的物体，其屏幕像素半径越小
    let linear_depth = max(center_depth, 0.0001); // 避免除以零
    let pixel_radius = max_radius / linear_depth; 
    
    var final_color = center_color.rgb;
    var total_weight = 1.0;
    
    // 4. 边缘保护模糊 (以 9 Tap 为例)
    let steps = 4;
    let step_size = pixel_radius / f32(steps);
    
    for (var i = -steps; i <= steps; i++) {
        if (i == 0) { continue; }
        
        let offset = vec2<i32>(blur_dir * (f32(i) * step_size));
        let sample_coord = center_coord + offset;
        
        // 读取邻采样点数据
        let sample_depth = textureLoad(t_depth, sample_coord, 0);
        let sample_packed = textureLoad(t_normal, sample_coord, 0);
        let sample_normal = normalize(sample_packed.xyz * 2.0 - 1.0);
        
        // 【核心】：边缘保护权重 (Bilateral Filter)
        // a. 深度断层惩罚：如果两个像素深度差异过大，拒绝混合（防止前方的脸糊到后方的背景上）
        let depth_diff = abs(center_depth - sample_depth);
        let depth_weight = exp(-depth_diff * 1000.0); // 1000.0 是敏感度因子，可微调
        
        // b. 法线夹角惩罚：如果两个像素法线朝向完全不同，拒绝混合（防止鼻子正面和侧面糊在一起）
        let normal_dot = max(dot(center_normal, sample_normal), 0.0);
        let normal_weight = pow(normal_dot, 4.0);
        
        // c. 高斯空间衰减
        let spatial_weight = 1.0 - (abs(f32(i)) / f32(steps)); 
        
        let weight = spatial_weight * depth_weight * normal_weight;
        
        // 累加计算 (混入材质特有的透射颜色 scatter_color)
        final_color += textureLoad(t_color, sample_coord, 0).rgb * weight * scatter_color;
        total_weight += weight;
    }
    
    return vec4<f32>(final_color / total_weight, center_color.a);
}
