fn pow2( x: f32 ) -> f32 {
    return x * x;
}

fn pow4( x: f32 ) -> f32 {
    let x2 = x * x;
    return x2 * x2;
}

const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 0.3183098861837907;
const EPSILON = 1e-6;


fn getDistanceAttenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
    var distance_falloff: f32 = 1.0 / max( pow( light_distance, decay_exponent ), 0.01 );
    if ( cutoff_distance > 0.0 ) {
        distance_falloff *= pow2( saturate( 1.0 - pow4( light_distance / cutoff_distance ) ) );
    }
    return distance_falloff;
}

fn getSpotAttenuation( cone_cosine: f32, penumbra_cosine: f32, angle_cosine: f32 ) -> f32 {
    return smoothstep( cone_cosine, penumbra_cosine, angle_cosine );
}

fn getAmbientLightIrradiance( ambientlight_color: vec3<f32> ) -> vec3<f32> {
    let irradiance = ambientlight_color;
    return irradiance;
}

struct IncidentLight {
    color: vec3<f32>,
    visible: bool,
    direction: vec3<f32>,
};

struct GeometricContext {
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    $$ if USE_CLEARCOAT is defined
    clearcoat_normal: vec3<f32>,
    $$ endif
};

struct ReflectedLight {
    direct_diffuse: vec3<f32>,
    direct_specular: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    indirect_specular: vec3<f32>,
};

$$ if USE_SHADOWS is defined
fn sample_shadow(shadow_matrix: mat4x4<f32>, shadow_layer_index: i32, world_position: vec3<f32>, bias: f32) -> f32 {
    if (shadow_layer_index < 0) {
        return 1.0;
    }

    let shadow_clip = shadow_matrix * vec4<f32>(world_position, 1.0);
    if (abs(shadow_clip.w) <= EPSILON) {
        return 1.0;
    }

    let shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    let shadow_uv = vec2<f32>(
        shadow_ndc.x * 0.5 + 0.5,
        shadow_ndc.y * -0.5 + 0.5
    );

    if (shadow_uv.x <= 0.0 || shadow_uv.x >= 1.0 || shadow_uv.y <= 0.0 || shadow_uv.y >= 1.0) {
        return 1.0;
    }

    let shadow_depth = shadow_ndc.z;
    if (shadow_depth <= 0.0 || shadow_depth >= 1.0) {
        return 1.0;
    }

    // 3x3 PCF (Percentage Closer Filtering)
    let dim = textureDimensions(t_shadow_map_2d_array);
    let texel_size = vec2<f32>(1.0 / f32(dim.x), 1.0 / f32(dim.y));
    let biased_depth = shadow_depth - bias;
    var shadow_sum = 0.0;

    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow_sum += textureSampleCompare(
                t_shadow_map_2d_array,
                s_shadow_map_compare,
                shadow_uv + offset,
                shadow_layer_index,
                biased_depth
            );
        }
    }
    return shadow_sum / 9.0;
}
$$ endif


fn BRDF_Lambert(diffuse_color: vec3<f32>) -> vec3<f32> {
    return RECIPROCAL_PI * diffuse_color;
}

fn F_Schlick(f0: vec3<f32>, f90: f32, dot_vh: f32,) -> vec3<f32> {
    // Optimized variant (presented by Epic at SIGGRAPH '13)
    // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );
    return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}

fn F_Schlick_f(f0: f32, f90: f32, dot_vh: f32,) -> f32 {
    // Optimized variant (presented by Epic at SIGGRAPH '13)
    // https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    let fresnel = exp2( ( - 5.55473 * dot_vh - 6.98316 ) * dot_vh );
    return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}

$$ if HAS_NORMAL_MAP is defined or HAS_CLEARCOAT_NORMAL_MAP is defined or USE_ANISOTROPY is defined
fn getTangentFrame( eye_pos: vec3<f32>, surf_norm: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let q0 = dpdx( eye_pos.xyz );
    let q1 = dpdy( eye_pos.xyz );
    let st0 = dpdx( uv.xy );
    let st1 = dpdy( uv.xy );
    let N = surf_norm; //  normalized
    let q1perp = cross( q1, N );
    let q0perp = cross( N, q0 );
    let T = q1perp * st0.x + q0perp * st1.x;
    let B = q1perp * st0.y + q0perp * st1.y;
    let det = max( dot( T, T ), dot( B, B ) );
    let scale = select(inverseSqrt(det), 0.0, det == 0.0);

    // We flip the Y when we compute the tangent from screen space.
    // See: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
    return mat3x3f(T * scale, -B * scale, N);
}

fn perturbNormal2Arb( eye_pos: vec3<f32>, surf_norm: vec3<f32>, map_n: vec3<f32>, uv: vec2<f32>, is_front: bool) -> vec3<f32> {
    let tbn = getTangentFrame( eye_pos, surf_norm, uv );
    let face_direction = f32(is_front) * 2.0 - 1.0;
    let scaled_map_n = vec3f(map_n.xy * face_direction, map_n.z);
    let perturb_normal = tbn * scaled_map_n;
    return normalize( perturb_normal );
}

$$ endif