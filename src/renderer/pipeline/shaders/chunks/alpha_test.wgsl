$$ if ALPHA_MODE == "MASK"
    if diffuse_color.a < u_material.alpha_test {
        discard;
    }
$$ endif