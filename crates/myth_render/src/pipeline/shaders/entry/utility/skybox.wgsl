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

{$ include 'core/full_screen_vertex' $}
{$ include "entry/utility/atmosphere/atmosphere_math" $}

// Auto-generated struct definition for SkyboxParams
{{ struct_definitions }}

// Auto-injected global bind group bindings (Group 0: camera, environment, etc.)
{{ binding_code }}

// --- Skybox-specific bindings (Group 1) ---
$$ if SKYBOX_PROCEDURAL
struct BakeParams {
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    moon_direction: vec3<f32>,
    moon_intensity: f32,
    star_axis: vec3<f32>,
    sun_disk_size: f32,
    moon_disk_size: f32,
    exposure: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    star_intensity: f32,
    star_rotation: f32,
    _pad2: vec2<f32>,
};

@group(1) @binding(0) var<uniform> u_bake_params: BakeParams;
@group(1) @binding(1) var t_sky_view: texture_2d<f32>;
@group(1) @binding(2) var s_skybox: sampler;
@group(1) @binding(3) var t_transmittance: texture_2d<f32>;
$$ if CELESTIAL_STARBOX_EQUIRECT
@group(1) @binding(5) var t_starbox_2d: texture_2d<f32>;
$$ endif
$$ if CELESTIAL_STARBOX_CUBE
@group(1) @binding(5) var t_starbox_cube: texture_cube<f32>;
$$ endif
$$ else
@group(1) @binding(0) var<uniform> u_params: SkyboxParams;
$$ endif

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

$$ if SKYBOX_PROCEDURAL
{$ include "entry/utility/atmosphere/celestial_bodies" $}
$$ endif

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

$$ if SKYBOX_PROCEDURAL
    let sky_uv = direction_to_sky_view_uv(world_dir);
    var procedural_color = textureSampleLevel(t_sky_view, s_skybox, sky_uv, 0.0).rgb;
    let view_transmittance = sample_direction_transmittance(world_dir);
    procedural_color += compute_celestial_lighting(
        world_dir,
        view_transmittance,
        u_render_state.time,
    );
    procedural_color *= u_bake_params.exposure;
    procedural_color = clamp(procedural_color, vec3<f32>(0.0), vec3<f32>(65000.0));
    color = vec4<f32>(procedural_color, 1.0);
$$ endif

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
    let tex_uv = equirectangular_uv(rot_dir);

    // HDR Color
    color = textureSample(t_skybox_2d, s_skybox, tex_uv);

    color = clamp(color, vec4<f32>(0.0), vec4<f32>(65000.0));
$$ endif

$$ if SKYBOX_PLANAR
    // --- Planar mode (screen-space mapping) ---
    color = textureSample(t_skybox_2d, s_skybox, in.uv);
$$ endif

$$ if SKYBOX_PROCEDURAL
    return color;
$$ else
    return vec4<f32>(color.rgb * u_params.intensity, color.a);
$$ endif
}
