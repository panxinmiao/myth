{{ vertex_input_code }} 

{{ binding_code }}    

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    $$ if has_uv
    @location({{ loc.next() }}) uv: vec2<f32>,
    $$ endif
    $$ if has_normal
    @location({{ loc.next() }}) normal: vec3<f32>,
    $$ endif
    @location({{ loc.next() }}) world_normal: vec3<f32>,
    @location({{ loc.next() }}) world_position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = u_model.model_matrix * vec4<f32>(input.position, 1.0);
    out.world_position = world_pos.xyz;
    out.position = u_render_state.view_projection * world_pos;
    out.world_normal = normalize(u_model.normal_matrix * input.normal);
    
    out.uv = input.uv;
    return out;
}

// === Fragment Shader ===
fn get_attenuation(light: Struct_lights, dist: f32) -> f32 {
    if (light.light_type == 0u) { return 1.0; } 
    let d = min(dist, light.range);
    let attn = 1.0 - pow(d / light.range, 4.0);
    return max(attn * attn, 0.0) / (1.0 + dist * dist); 
}

fn get_spot_factor(light: Struct_lights, L: vec3<f32>) -> f32 {
    if (light.light_type != 2u) { return 1.0; }
    let actual_cos = dot(-L, normalize(light.direction));
    return smoothstep(light.outer_cone_cos, light.inner_cone_cos, actual_cos);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = u_material.color;
    $$ if use_map
        let tex_color = textureSample(t_map, s_map, in.uv);
        base_color = base_color * tex_color;
    $$ endif

    let N = normalize(in.world_normal);
    let V = normalize(u_render_state.camera_position - in.world_position);

    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);

    let num_lights = arrayLength(&st_lights);
    
    for (var i = 0u; i < num_lights; i = i + 1u) {
        // 直接索引结构体数组
        let light = st_lights[i];
        
        var L: vec3<f32>;
        var dist: f32 = 1.0;
        
        if (light.light_type == 0u) {
            L = normalize(-light.direction);
            dist = 0.0;
        } else {
            let light_dir = light.position - in.world_position;
            dist = length(light_dir);
            L = normalize(light_dir);
        }

        var attenuation = light.intensity;
        if (light.light_type != 0u) {
            if (dist > light.range) { continue; }
            attenuation *= get_attenuation(light, dist);
        }
        
        if (light.light_type == 2u) {
            attenuation *= get_spot_factor(light, L);
        }

        if (attenuation <= 0.001) { continue; }

        // Phong 计算
        let n_dot_l = max(dot(N, L), 0.0);
        total_diffuse += n_dot_l * light.color * attenuation;
        
        let H = normalize(L + V);
        let n_dot_h = max(dot(N, H), 0.0);
        let spec_factor = pow(n_dot_h, u_material.shininess);
        total_specular += spec_factor * u_material.specular.rgb * light.color * attenuation;
    }

    // 组合
    let ambient = vec3<f32>(0.1) * base_color.rgb; // 简单环境光
    let final_color = ambient + (total_diffuse * base_color.rgb) + total_specular + u_material.emissive.rgb;

    return vec4<f32>(final_color, base_color.a);
}