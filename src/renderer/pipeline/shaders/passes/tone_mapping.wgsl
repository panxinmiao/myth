{$ include 'full_screen_vertex.wgsl' $}

{$ include 'tone_mapping_pars' $}

{{ struct_definitions }}

// Auto-injected binding code for global resources (e.g. frame-level uniforms, scene-level data, etc.)
{{ binding_code }}

// bindings
@group(1) @binding(0)
var color_tex: texture_2d<f32>;
@group(1) @binding(1)
var tex_sampler: sampler;
@group(1) @binding(2)
var<uniform> u_effect: Uniforms;

$$ if USE_LUT is defined
@group(1) @binding(3)
var lut_texture: texture_3d<f32>;
@group(1) @binding(4)
var lut_sampler: sampler;
$$ endif


@fragment
fn fs_main(varyings: VertexOutput) -> @location(0) vec4<f32> {
    let uv = varyings.uv;
    var color_sample: vec4<f32>;

    // 1、Chromatic Aberration
    // Accroding to the distance from the center, apply a UV offset that increases towards the edges, creating more chromatic aberration at the edges and none at the center.
    if (u_effect.chromatic_aberration > 0.001) {

        let offset = (uv - 0.5) * u_effect.chromatic_aberration * 0.05; 
        let r = textureSample(color_tex, tex_sampler, uv + offset).r;
        let g = textureSample(color_tex, tex_sampler, uv).g; // green channel is sampled without offset for sharper focus
        let b = textureSample(color_tex, tex_sampler, uv - offset).b;
        let a = textureSample(color_tex, tex_sampler, uv).a;

        color_sample = vec4<f32>(r, g, b, a);
    }else {
        color_sample = textureSample(color_tex, tex_sampler, uv);
    }


    // 2. Apply tone mapping to RGB channels
    var rgb = toneMapping(color_sample.rgb * u_effect.exposure);

    // 3. Color Grading (3D LUT) - macro-guarded
$$ if USE_LUT is defined
    {
        // Compute half-texel offset to avoid boundary artifacts
        // let lut_size = 32.0;
        let lut_size = textureDimensions(lut_texture);
        let half_texel = vec3<f32>(0.5) / lut_size;

        // Clamp to [0, 1] and remap to 3D texture coordinates with half-texel inset
        let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        let lut_uvw = clamped * ((lut_size - vec3<f32>(1.0)) / lut_size) + half_texel;

        // Trilinear-interpolated 3D texture sample
        let lut_color = textureSampleLevel(lut_texture, lut_sampler, lut_uvw, 0.0).rgb;
        rgb = mix(rgb, lut_color, u_effect.lut_contribution);
    }
$$ endif

    // 4. Contrast & Saturation
    // Contrast
    rgb = (rgb - 0.5) * u_effect.contrast + 0.5;
    
    // Saturation
    let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)); // Rec. 709 luma coefficients
    rgb = mix(vec3<f32>(luminance), rgb, u_effect.saturation);
    
    rgb = max(rgb, vec3<f32>(0.0));

    // 5. Vignette (edge darkening) - controlled via uniform, no macro needed
    if (u_effect.vignette_intensity > 0.001) {
        // compute a radial mask that peaks at the center and falls off towards edges
        var v = uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y) * 16.0;

        // map smoothness to parabola exponent, where higher smoothness means a wider, softer highlight area
        let power = mix(1.0, 0.1, u_effect.vignette_smoothness);
        v = pow(v, power);

        // invert to create a mask that is 1.0 at the center and falls to 0.0 at edges
        var vignette_mask = 1.0 - v;

        // apply intensity and clamp to [0, 1]
        vignette_mask = clamp(vignette_mask * u_effect.vignette_intensity, 0.0, 1.0);

        rgb = mix(rgb, u_effect.vignette_color.rgb, vignette_mask);

    }

    // 6. Film Grain
    if (u_effect.film_grain > 0.001) {
        let seed = u_render_state.time * 100.0;
        let noise = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233)) + seed) * 43758.5453);
        let grain = (noise - 0.5) * u_effect.film_grain;
        rgb = rgb + grain;
    }

    return vec4<f32>(rgb, color_sample.a);
}