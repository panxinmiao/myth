$$ if ALPHA_MODE == "MASK"
    if opacity < u_material.alpha_test {
        discard;
    }
$$ endif