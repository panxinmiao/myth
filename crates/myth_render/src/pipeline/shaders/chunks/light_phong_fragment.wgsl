    var material: BlinnPhongMaterial;
    material.diffuse_color = diffuse_color.rgb;
    material.specular_color = u_material.specular.rgb;
    material.specular_shininess = u_material.shininess;
    material.specular_strength = specular_strength;