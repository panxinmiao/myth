// === Skybox / Background Pass Shader ===
//
// Renders a fullscreen triangle at the far depth plane (Reverse-Z: Z = 0.0).
// Uses ray reconstruction from the inverse view-projection matrix to compute
// view-space direction for texture sampling.
//
// Pipeline variants (selected via ShaderDefines):
//   SKYBOX_GRADIENT      - Vertical color gradient (no texture)
//   SKYBOX_CUBE          - Cubemap sampling
//   SKYBOX_EQUIRECT      - Equirectangular (lat-long) 2D texture sampling
//   SKYBOX_PLANAR        - Screen-space planar 2D texture sampling

{$ include 'full_screen_vertex.wgsl' $}

// Auto-generated struct definition for SkyboxParams
{{ struct_definitions }}

// Auto-injected global bind group bindings (Group 0: camera, environment, etc.)
{{ binding_code }}

// --- Skybox-specific bindings (Group 1) ---
@group(1) @binding(0) var<uniform> u_params: SkyboxParams;

$$ if SKYBOX_CUBE
@group(1) @binding(1) var t_skybox_cube: texture_cube<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_EQUIRECT
@group(1) @binding(1) var t_skybox_2d: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_PLANAR
@group(1) @binding(1) var t_skybox_2d: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
$$ endif

// --- Constants ---
const PI: f32 = 3.14159265359;
const INV_ATAN: vec2<f32> = vec2<f32>(0.15915494, 0.31830989); // (1/2π, 1/π)

// --- Fragment Shader ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;

    // --- 1.Pixel-Perfect Ray Reconstruction ---
    let ndc = in.uv * 2.0 - 1.0;

    // Use Near Plane (Z=1.0) to get the direction vector without needing the actual depth value
    let clip_pos = vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    
    // Transform from clip space to world space (using global RenderState uniforms)
    let world_pos_h = u_render_state.view_projection_inverse * clip_pos;
    let world_pos = world_pos_h.xyz / world_pos_h.w;
    
    // Compute world-space ray direction
    let world_dir = normalize(world_pos - u_render_state.camera_position);

$$ if SKYBOX_GRADIENT
    // Smooth vertical blend based on Y component of view direction
    let t = smoothstep(-0.5, 0.5, world_dir.y);
    color = mix(u_params.color_bottom, u_params.color_top, t);

    // Add Dithering to reduce banding in gradients, especially at low precision (e.g. 8-bit displays)
    let noise = fract(sin(dot(in.position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) / 255.0;
$$ endif

$$ if SKYBOX_CUBE or SKYBOX_EQUIRECT
    // Apply Y-axis rotation
    let s = sin(u_params.rotation);
    let c = cos(u_params.rotation);
    let rot_dir = vec3<f32>(
        -(world_dir.x * c - world_dir.z * s), // Negate X to convert from left-handed to right-handed coordinates for cubemap sampling
        world_dir.y,
        world_dir.x * s + world_dir.z * c
    );
$$ endif

$$ if SKYBOX_CUBE
    color = textureSample(t_skybox_cube, s_skybox, rot_dir);
$$ endif

$$ if SKYBOX_EQUIRECT
    // Convert direction to equirectangular UV
    let eq_uv = vec2<f32>(
        atan2(rot_dir.z, rot_dir.x),
        acos(clamp(rot_dir.y, -1.0, 1.0))
    );

    let tex_uv = vec2<f32>(
        eq_uv.x * INV_ATAN.x + 0.5,
        eq_uv.y * INV_ATAN.y
    );

    // HDR Color
    color = textureSample(t_skybox_2d, s_skybox, tex_uv);

    color = clamp(color, vec4<f32>(0.0), vec4<f32>(65000.0));
$$ endif

$$ if SKYBOX_PLANAR
    // --- Planar mode (screen-space mapping) ---
    color = textureSample(t_skybox_2d, s_skybox, in.uv);
$$ endif

    return vec4<f32>(color.rgb * u_params.intensity, color.a);
}
