$$ if map_uv is defined
    out.map_uv = (u_material.map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_normal_map is defined
    out.normal_map_uv = (u_material.normal_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_specular_map is defined
    out.specular_map_uv = (u_material.specular_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_specular_intensity_map is defined
    out.specular_intensity_map_uv = (u_material.specular_intensity_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_ao_map is defined
    out.ao_map_uv = (u_material.ao_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_light_map is defined
    out.light_map_uv = (u_material.light_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_emissive_map is defined
    out.emissive_map_uv = (u_material.emissive_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_metalness_map is defined
    out.metalness_map_uv = (u_material.metalness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_roughness_map is defined
    out.roughness_map_uv = (u_material.roughness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_clearcoat_map is defined
    out.clearcoat_map_uv = (u_material.clearcoat_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_clearcoat_normal_map is defined
    out.clearcoat_normal_map_uv = (u_material.clearcoat_normal_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_clearcoat_roughness_map is defined
    out.clearcoat_roughness_map_uv = (u_material.clearcoat_roughness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_iridescence_map is defined
    out.iridescence_map_uv = (u_material.iridescence_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_iridescence_thickness_map is defined
    out.iridescence_thickness_map_uv = (u_material.iridescence_thickness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if use_anisotropy_map is defined
    out.anisotropy_map_uv = (u_material.anisotropy_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

