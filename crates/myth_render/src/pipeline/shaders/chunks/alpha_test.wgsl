$$ if ALPHA_MODE == "MASK"

    $$ if ALPHA_TO_COVERAGE is defined

    let aa_width = max(fwidth(opacity) * 1.5, 0.0001);

	// opacity = smoothstep( u_material.alpha_test, u_material.alpha_test + fwidth( opacity ), opacity );
    opacity = smoothstep(u_material.alpha_test - aa_width * 0.5, u_material.alpha_test + aa_width * 0.5, opacity);
    opacity = clamp(opacity, 0.0, 1.0);

	if ( opacity <= 0.01 ) {
        discard;
    }

	$$ else

    if opacity < u_material.alpha_test {
        discard;
    }

    $$ endif
$$ endif