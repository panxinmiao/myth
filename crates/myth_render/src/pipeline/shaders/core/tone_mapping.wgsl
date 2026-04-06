$$ if TONE_MAPPING_MODE == "LINEAR"
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        return saturate(color);
    }
$$ endif

$$ if TONE_MAPPING_MODE == "REINHARD"
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        return saturate(color / (vec3<f32>(1.0) + color));
    }
$$ endif

$$ if TONE_MAPPING_MODE == "CINEON"
    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        var mapped_color = max(vec3<f32>(0.0), color - vec3<f32>(0.004));
        return pow((mapped_color * (6.2 * mapped_color + 0.5)) / (mapped_color * (6.2 * mapped_color + 1.7) + 0.06), vec3<f32>(2.2));
    }
$$ endif

$$ if TONE_MAPPING_MODE == "ACES_FILMIC"
    fn RRTAndODTFit(v: vec3<f32>) -> vec3<f32> {
        let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
        let b = v * (vec3<f32>(0.983729) * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
        return a / b;
    }

    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        let ACESInputMat = mat3x3<f32>(
            vec3<f32>(0.59719, 0.07600, 0.02840),
            vec3<f32>(0.35458, 0.90834, 0.13383),
            vec3<f32>(0.04823, 0.01566, 0.83777)
        );
        let ACESOutputMat = mat3x3<f32>(
            vec3<f32>(1.60475, -0.10208, -0.00327),
            vec3<f32>(-0.53108, 1.10813, -0.07276),
            vec3<f32>(-0.07367, -0.00605, 1.07602)
        );
        var mapped_color = color / 0.6;
        mapped_color = ACESInputMat * mapped_color;
        mapped_color = RRTAndODTFit(mapped_color);
        mapped_color = ACESOutputMat * mapped_color;
        return saturate(mapped_color);
    }
$$ endif

$$ if TONE_MAPPING_MODE == "AGX"
    const LINEAR_REC2020_TO_LINEAR_SRGB = mat3x3f(
        vec3f( 1.6605, - 0.1246, - 0.0182 ),
        vec3f( - 0.5876, 1.1329, - 0.1006 ),
        vec3f( - 0.0728, - 0.0083, 1.1187 )
    );

    const LINEAR_SRGB_TO_LINEAR_REC2020 = mat3x3f(
        vec3f( 0.6274, 0.0691, 0.0164 ),
        vec3f( 0.3293, 0.9195, 0.0880 ),
        vec3f( 0.0433, 0.0113, 0.8956 )
    );

    fn agxDefaultContrastApprox( x: vec3<f32> ) -> vec3<f32> {

        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;

        return  - 17.86 * x6 * x
            + 78.01 * x6
            - 126.7 * x4 * x
            + 92.06 * x4
            - 28.72 * x2 * x
            + 4.361 * x2
            - 0.1718 * x
            + 0.002857;
    }

    fn agxLook( x: vec3<f32>) -> vec3<f32> {
        $$ if AGX_LOOK is not defined or AGX_LOOK == "NONE"
            return x;
        $$ else

            let lw = vec3f(0.2126, 0.7152, 0.0722);
            let luma = dot(x, lw);

            let offset = vec3f(0.0);

            $$ if AGX_LOOK == "GOLDEN"
                let slope = vec3f(1.0, 0.9, 0.5);
                let power = vec3f(0.8);
                let sat = 1.3;
            $$ elif AGX_LOOK == "PUNCHY"
                let slope = vec3f(1.0);
                let power = vec3f(1.35);
                let sat = 1.4;
            $$ endif

            let val = pow(x * slope + offset, power);
            return luma + sat * (val - luma);
        $$ endif
    }

    fn toneMapping( color: vec3<f32> ) -> vec3<f32> {
        let AgXInsetMatrix = mat3x3f(
            vec3f( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
            vec3f( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
            vec3f( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
        );
        let AgXOutsetMatrix = mat3x3f(
            vec3f( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
            vec3f( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
            vec3f( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
        );

        let AgxMinEv = - 12.47393;
        let AgxMaxEv = 4.026069;

        var mapped_color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
        mapped_color = AgXInsetMatrix * mapped_color;
        mapped_color = max( mapped_color, vec3f( 1e-10 ) );
        mapped_color = log2( mapped_color );
        mapped_color = ( mapped_color - vec3f( AgxMinEv ) ) / vec3f( AgxMaxEv - AgxMinEv );
        mapped_color = clamp( mapped_color, vec3f( 0.0 ), vec3f( 1.0 ) );
        mapped_color = agxDefaultContrastApprox( mapped_color );
        mapped_color = agxLook(mapped_color);
        mapped_color = AgXOutsetMatrix * mapped_color;
        mapped_color = pow( max( vec3f( 0.0 ), mapped_color), vec3f( 2.2 ) );
        mapped_color = LINEAR_REC2020_TO_LINEAR_SRGB * mapped_color;
        mapped_color = clamp( mapped_color, vec3f( 0.0 ), vec3f( 1.0 ) );
        return mapped_color;

    }
$$ endif

$$ if TONE_MAPPING_MODE == "NEUTRAL"

    fn toneMapping(color: vec3<f32>) -> vec3<f32> {
        let StartCompression = 0.8 - 0.04;
        let Desaturation = 0.15;

        let x = min(color.r, min(color.g, color.b));
        let offset = select(0.04, x - 6.25 * x * x, x < 0.08);

        var mapped_color = color - vec3<f32>(offset);

        let peak = max(mapped_color.r, max(mapped_color.g, mapped_color.b));
        
        if (peak < StartCompression) {
            return mapped_color;
        }
        
        let d = 1.0 - StartCompression;
        let new_peak = 1.0 - d * d / (peak + d - StartCompression);

        mapped_color *= (new_peak / peak);

        let g = 1.0 - 1.0 / (Desaturation * (peak - new_peak) + 1.0);
        
        return mix(mapped_color, vec3<f32>(new_peak), g);
    }

$$ endif
