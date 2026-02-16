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

// --- Uniforms: Camera data ---
struct SkyboxCamera {
    view_projection_inverse: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad0: f32,
};
@group(0) @binding(0) var<uniform> u_camera: SkyboxCamera;

// --- Uniforms: Skybox parameters ---
struct SkyboxParams {
    color_top: vec4<f32>,
    color_bottom: vec4<f32>,
    rotation: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(1) var<uniform> u_params: SkyboxParams;

$$ if SKYBOX_CUBE
@group(0) @binding(2) var t_skybox_cube: texture_cube<f32>;
@group(0) @binding(3) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_EQUIRECT
@group(0) @binding(2) var t_skybox_2d: texture_2d<f32>;
@group(0) @binding(3) var s_skybox: sampler;
$$ endif

$$ if SKYBOX_PLANAR
@group(0) @binding(2) var t_skybox_2d: texture_2d<f32>;
@group(0) @binding(3) var s_skybox: sampler;
$$ endif

// --- Vertex output ---
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// --- Vertex Shader: Fullscreen Triangle ---
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle covering the entire viewport:
    //   vertex 0: (-1, -1)  uv (0, 0)
    //   vertex 1: ( 3, -1)  uv (2, 0)
    //   vertex 2: (-1,  3)  uv (0, 2)
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    let ndc = uv * 2.0 - 1.0;

    // Force Z = 0.0 for Reverse-Z far plane (maximum distance).
    // Depth test (GreaterEqual) will cull this behind any opaque geometry.
    out.position = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);

    out.uv = uv;
    return out;
}

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
    let clip_pos = vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    
    // Transform from clip space to world space
    let world_pos_h = u_camera.view_projection_inverse * clip_pos;
    let world_pos = world_pos_h.xyz / world_pos_h.w;
    
    // Compute world-space ray direction
    let world_dir = normalize(world_pos - u_camera.camera_position);

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
        world_dir.x * c - world_dir.z * s,
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

    // var u = atan2(output.z, output.x) / TWO_PI + 0.5;
    // // var v = asin(clamp(output.y, -1.0, 1.0)) / PI + 0.5;
    // var v = acos(clamp(output.y, -1.0, 1.0)) / PI;

    let tex_uv = vec2<f32>(
        eq_uv.x * INV_ATAN.x + 0.5,
        eq_uv.y * INV_ATAN.y
    );

    color = textureSample(t_skybox_2d, s_skybox, tex_uv);
$$ endif

$$ if SKYBOX_PLANAR
    // --- Planar mode (screen-space mapping) ---
    let planar_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    color = textureSample(t_skybox_2d, s_skybox, planar_uv);
$$ endif

    return vec4<f32>(color.rgb * u_params.intensity, color.a);
}
