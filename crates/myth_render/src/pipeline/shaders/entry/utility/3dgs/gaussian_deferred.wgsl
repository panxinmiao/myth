{$ include 'core/full_screen_vertex' $}

{{ binding_code }}

{$ include 'core/common' $}
{$ include 'modules/lighting/punctual' $}
{$ include 'modules/bsdf/physical' $}

struct DeferredSettings {
    flags: vec4<u32>,
    material_params: vec4<f32>,
    lighting_params: vec4<f32>,
};

@group(1) @binding(0)
var t_albedo: texture_2d<f32>;
@group(1) @binding(1)
var t_normal: texture_2d<f32>;
@group(1) @binding(2)
var t_depth: texture_2d<f32>;
@group(1) @binding(3)
var s_point: sampler;
@group(1) @binding(4)
var<uniform> u_settings: DeferredSettings;

fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b),
    );
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = u_render_state.view_projection_inverse * ndc;
    let w = select(world.w, 1e-6, abs(world.w) < 1e-6);
    return world.xyz / w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_accum = textureSampleLevel(t_albedo, s_point, in.uv, 0.0);
    let normal_accum = textureSampleLevel(t_normal, s_point, in.uv, 0.0);
    let depth_accum = textureSampleLevel(t_depth, s_point, in.uv, 0.0);

    let alpha_cutoff = max(u_settings.lighting_params.z, 1e-4);
    let alpha = albedo_accum.a;
    if alpha <= alpha_cutoff || depth_accum.a <= alpha_cutoff || normal_accum.a <= alpha_cutoff {
        return vec4<f32>(0.0);
    }

    let encoded_albedo = clamp(albedo_accum.rgb / alpha, vec3<f32>(0.0), vec3<f32>(65504.0));
    let albedo = select(
        encoded_albedo,
        srgb_to_linear(clamp(encoded_albedo, vec3<f32>(0.0), vec3<f32>(1.0))),
        u_settings.flags.x != 0u,
    ) * u_settings.material_params.w;

    var normal = normal_accum.rgb / normal_accum.a;
    if dot(normal, normal) <= 1e-6 {
        return vec4<f32>(0.0);
    }
    normal = normalize(normal);

    let depth = depth_accum.r / depth_accum.a;
    let world_pos = reconstruct_world_position(in.uv, depth);
    let view_dir = normalize(u_render_state.camera_position - world_pos);
    if dot(normal, view_dir) < 0.0 {
        normal = -normal;
    }

    var geometry: GeometricContext;
    geometry.position = world_pos;
    geometry.normal = normal;
    geometry.view_dir = view_dir;

    let roughness = clamp(u_settings.material_params.x, 0.045, 1.0);
    let metallic = clamp(u_settings.material_params.y, 0.0, 1.0);
    let reflectance = clamp(u_settings.material_params.z, 0.02, 1.0);

    var material: SurfaceContext;
    material.diffuse_color = albedo * (1.0 - metallic);
    material.roughness = roughness;
    material.specular_color = mix(vec3<f32>(reflectance), albedo, metallic);
    material.specular_f90 = 1.0;

    var reflected_light = ReflectedLight(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );

    evaluate_punctual_lights(geometry, material, &reflected_light);
    RE_IndirectDiffuse(
        getAmbientLightIrradiance(u_environment.ambient_light.rgb * u_settings.lighting_params.y),
        geometry,
        material,
        &reflected_light,
    );

    let direct = (reflected_light.direct_diffuse + reflected_light.direct_specular)
        * u_settings.lighting_params.x;
    let indirect = reflected_light.indirect_diffuse;
    let lit = max(direct + indirect, vec3<f32>(0.0));

    return vec4<f32>(lit * alpha, alpha);
}