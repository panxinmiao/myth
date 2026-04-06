// ── Thin-Film Iridescence ────────────────────────────────────────────────
//
// Physically-based thin-film interference model.
// Ref: https://belcour.github.io/blog/research/2017/05/01/brdf-thin-film.html
//
// Depends on: core/common.wgsl (PI, pow2, F_Schlick, F_Schlick_f)

$$ if USE_IRIDESCENCE is defined

const XYZ_TO_REC709 = mat3x3f(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

fn Fresnel0ToIor( fresnel0: vec3<f32> ) -> vec3<f32> {
    let sqrt_f0 = sqrt( fresnel0 );
    return ( vec3<f32>( 1.0 ) + sqrt_f0 ) / ( vec3<f32>( 1.0 ) - sqrt_f0 );
}

fn IorToFresnel0_v( transmitted_ior: vec3<f32>, incident_ior: vec3<f32> ) -> vec3<f32> {
    let p = ( transmitted_ior - incident_ior ) / ( transmitted_ior + incident_ior );
    return p * p;
}

fn IorToFresnel0( transmitted_ior: f32, incident_ior: f32 ) -> f32 {
    return pow2( ( transmitted_ior - incident_ior ) / ( transmitted_ior + incident_ior ) );
}

fn evalSensitivity( OPD: f32, shift: vec3<f32> ) -> vec3<f32> {
    let phase = 2.0 * PI * OPD * 1.0e-9;
    let val = vec3<f32>( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
    let pos = vec3<f32>( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
    let VAR = vec3<f32>( 4.3278e+09, 9.3046e+09, 6.6121e+09 );

    var xyz = val * sqrt( 2.0 * PI * VAR ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * VAR );
    xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift.x ) * exp( - 4.5282e+09 * pow2( phase ) );
    xyz /= 1.0685e-7;

    let rgb = XYZ_TO_REC709 * xyz;
    return rgb;
}

fn evalIridescence( outside_ior: f32, eta2: f32, cos_theta1: f32, thin_film_thickness: f32, base_f0: vec3<f32> ) -> vec3<f32> {
    let iridescence_ior = mix( outside_ior, eta2, smoothstep( 0.0, 0.03, thin_film_thickness ) );
    let sin_theta2_sq = pow2( outside_ior / iridescence_ior ) * ( 1.0 - pow2( cos_theta1 ) );

    let cos_theta2_sq = 1.0 - sin_theta2_sq;
    if ( cos_theta2_sq < 0.0 ) {
        return vec3<f32>( 1.0);
    }

    let cos_theta2 = sqrt( cos_theta2_sq );

    let R0 = IorToFresnel0( iridescence_ior, outside_ior );
    let R12 = F_Schlick_f( R0, 1.0, cos_theta1 );
    let T121 = 1.0 - R12;
    var phi12 = 0.0;
    if ( iridescence_ior < outside_ior ) { phi12 = PI; }
    let phi21 = PI - phi12;

    let base_ior = Fresnel0ToIor( clamp( base_f0, vec3f(0.0), vec3f(0.9999) ) );
    let R1 = IorToFresnel0_v( base_ior, vec3f(iridescence_ior) );
    let R23 = F_Schlick( R1, 1.0, cos_theta2 );
    var phi23 = vec3<f32>( 0.0 );
    if ( base_ior.x < iridescence_ior ) { phi23.x = PI; }
    if ( base_ior.y < iridescence_ior ) { phi23.y = PI; }
    if ( base_ior.z < iridescence_ior ) { phi23.z = PI; }

    let OPD = 2.0 * iridescence_ior * thin_film_thickness * cos_theta2;
    let phi = vec3<f32>( phi21 ) + phi23;

    let R123 = clamp( R12 * R23, vec3f(1e-5), vec3f(0.9999) );
    let r123 = sqrt( R123 );
    let Rs = pow2( T121 ) * R23 / ( vec3<f32>( 1.0 ) - R123 );

    let C0 = R12 + Rs;
    var I = C0;

    var Cm = Rs - T121;
    for ( var m = 1; m <= 2; m = m + 1 ) {
        Cm *= r123;
        let Sm = 2.0 * evalSensitivity( f32(m) * OPD, f32(m) * phi );
        I += Cm * Sm;
    }

    return max( I, vec3<f32>( 0.0 ) );
}

$$ endif
