{$ include 'full_screen_vertex.wgsl' $}

{$ include 'tone_mapping_pars' $}

struct Uniforms{
    exposure: f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    lut_contribution: f32,
};

// bindings
@group(0) @binding(0)
var colorTex: texture_2d<f32>;
@group(0) @binding(1)
var texSampler: sampler;
@group(0) @binding(2)
var<uniform> u_effect: Uniforms;

$$ if USE_LUT is defined
@group(0) @binding(3)
var lutTexture: texture_3d<f32>;
@group(0) @binding(4)
var lutSampler: sampler;
$$ endif


@fragment
fn fs_main(varyings: VertexOutput) -> @location(0) vec4<f32> {
    let texCoord = varyings.uv;
    var color = textureSample(colorTex, texSampler, texCoord);

    // A. Apply tone mapping to RGB channels
    var rgb = toneMapping(color.rgb * u_effect.exposure);

    // B. Color Grading (3D LUT) - macro-guarded
$$ if USE_LUT is defined
    {
        // Assume LUT size is 32; compute half-texel offset to avoid boundary artifacts
        let lut_size = 32.0;
        let half_texel = 0.5 / lut_size;

        // Clamp to [0, 1] and remap to 3D texture coordinates with half-texel inset
        let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        let lut_uvw = clamped * ((lut_size - 1.0) / lut_size) + half_texel;

        // Trilinear-interpolated 3D texture sample
        let lut_color = textureSampleLevel(lutTexture, lutSampler, lut_uvw, 0.0).rgb;
        rgb = mix(rgb, lut_color, u_effect.lut_contribution);
    }
$$ endif

    // C. Vignette (edge darkening) - controlled via uniform, no macro needed
    if (u_effect.vignette_intensity > 0.0) {
        let center_dist = distance(texCoord, vec2<f32>(0.5, 0.5));
        // smoothstep produces a soft falloff mask from edges
        let vignette = smoothstep(0.8, u_effect.vignette_smoothness, center_dist * (1.0 + u_effect.vignette_intensity));
        rgb *= vignette;
    }

    return vec4<f32>(rgb, color.a);
}