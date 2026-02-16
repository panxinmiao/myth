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
$$ if SKYBOX_PLANAR
    @location(0) uv: vec2<f32>,
$$ else
    @location(0) world_dir: vec3<f32>,
$$ endif
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

$$ if SKYBOX_PLANAR
    // Planar mode: pass UV directly (screen-space mapping)
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
$$ else
    // Reconstruct world-space view direction from clip coordinates.
    // Use two depth values for numerical robustness (avoids singularity at far plane).
    let clip_near = vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let clip_mid  = vec4<f32>(ndc.x, ndc.y, 0.5, 1.0);

    let world_near = u_camera.view_projection_inverse * clip_near;
    let world_mid  = u_camera.view_projection_inverse * clip_mid;

    let p_near = world_near.xyz / world_near.w;
    let p_mid  = world_mid.xyz / world_mid.w;

    out.world_dir = normalize(p_mid - p_near);
$$ endif

    return out;
}

// --- Constants ---
const PI: f32 = 3.14159265359;
const INV_ATAN: vec2<f32> = vec2<f32>(0.15915494, 0.31830989); // (1/2π, 1/π)

// --- Fragment Shader ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;

$$ if SKYBOX_GRADIENT
    // --- Gradient mode ---
    let dir = normalize(in.world_dir);
    // Smooth vertical blend based on Y component of view direction
    let t = smoothstep(-0.5, 0.5, dir.y);
    color = mix(u_params.color_bottom, u_params.color_top, t);
$$ endif

$$ if SKYBOX_CUBE
    // --- Cubemap mode ---
    let dir = normalize(in.world_dir);

    // Apply Y-axis rotation
    let s = sin(u_params.rotation);
    let c = cos(u_params.rotation);
    let rot_dir = vec3<f32>(
        dir.x * c - dir.z * s,
        dir.y,
        dir.x * s + dir.z * c
    );

    color = textureSample(t_skybox_cube, s_skybox, rot_dir);
$$ endif

$$ if SKYBOX_EQUIRECT
    // --- Equirectangular mode ---
    let dir = normalize(in.world_dir);

    // Apply Y-axis rotation
    let s = sin(u_params.rotation);
    let c = cos(u_params.rotation);
    let rot_dir = vec3<f32>(
        dir.x * c - dir.z * s,
        dir.y,
        dir.x * s + dir.z * c
    );

    // Convert direction to equirectangular UV
    let eq_uv = vec2<f32>(
        atan2(rot_dir.z, rot_dir.x),
        asin(clamp(rot_dir.y, -1.0, 1.0))
    );
    let tex_uv = eq_uv * INV_ATAN + 0.5;

    color = textureSample(t_skybox_2d, s_skybox, tex_uv);
$$ endif

$$ if SKYBOX_PLANAR
    // --- Planar mode (screen-space mapping) ---
    color = textureSample(t_skybox_2d, s_skybox, in.uv);
$$ endif

    return vec4<f32>(color.rgb * u_params.intensity, color.a);
}
