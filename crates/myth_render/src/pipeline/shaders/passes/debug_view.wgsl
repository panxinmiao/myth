// Debug View — visualise intermediate RDG textures.
//
// A full-screen post-process pass that remaps arbitrary texture formats
// into a displayable [0, 1] RGB range.  The `view_mode` uniform selects
// the mapping strategy so a single pipeline can handle depth, normals,
// single-channel occlusion, and standard colour buffers alike.

{$ include 'full_screen_vertex.wgsl' $}

struct DebugUniforms {
    view_mode: u32,
    custom_scale: f32,
    z_near: f32,
    z_far: f32,
};

// Group 0 (static): sampler + uniforms — owned by the Feature.
@group(0) @binding(0) var debug_sampler: sampler;
@group(0) @binding(1) var<uniform> uniforms: DebugUniforms;

// Group 1 (transient): source texture — rebuilt each frame.
$$ if IS_DEPTH
@group(1) @binding(0) var debug_texture: texture_depth_2d;
$$ else
@group(1) @binding(0) var debug_texture: texture_2d<f32>;
$$ endif

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    $$ if IS_DEPTH
    let depth_val = textureSampleLevel(debug_texture, debug_sampler, in.uv, 0i);
    let tex_color = vec4<f32>(depth_val, depth_val, depth_val, 1.0);
    $$ else
    let tex_color = textureSampleLevel(debug_texture, debug_sampler, in.uv, 0.0);
    $$ endif

    switch uniforms.view_mode {
        // Mode 1: SSAO / Roughness / Metallic
        case 1u: {
            return vec4<f32>(tex_color.rrr, 1.0);
        }
        // Mode 2: World/View Normals
        case 2u: {
            return vec4<f32>(tex_color.rgb * 0.5 + 0.5, 1.0);
        }
        // Mode 3: Velocity / Motion Vectors
        case 3u: {
            let vel = tex_color.xy * uniforms.custom_scale;
            let abs_vel = abs(vel);
            let positive_vel = max(vel, vec2<f32>(0.0));
            let negative_vel = max(-vel, vec2<f32>(0.0));
            
            let color = vec3<f32>(
                positive_vel.x + negative_vel.y, // R
                positive_vel.y + negative_vel.x, // G
                negative_vel.x + negative_vel.y  // B
            );

            return vec4<f32>(color + vec3<f32>(length(vel)), 1.0);
        }
        // Mode 4: Depth (Reverse-Z)
        case 4u: {
            let ndc_z = tex_color.r; 
            let linear_depth = (uniforms.z_near * uniforms.z_far) / 
                               (uniforms.z_near + ndc_z * (uniforms.z_far - uniforms.z_near));
            
            let display_depth = linear_depth / uniforms.z_far;
            
            let fract_depth = fract(display_depth * uniforms.custom_scale);

            return vec4<f32>(vec3<f32>(display_depth * 0.8 + fract_depth * 0.2), 1.0);
        }
        // Default: colour pass-through
        default: {
            return vec4<f32>(tex_color.rgb, 1.0);
        }
    }
}
