$$ if HAS_MAP is defined
    out.map_uv = (u_material.map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_NORMAL_MAP is defined
    out.normal_map_uv = (u_material.normal_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_SPECULAR_MAP is defined
    out.specular_map_uv = (u_material.specular_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_SPECULAR_INTENSITY_MAP is defined
    out.specular_intensity_map_uv = (u_material.specular_intensity_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_AO_MAP is defined
    out.ao_map_uv = (u_material.ao_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_LIGHT_MAP is defined
    out.light_map_uv = (u_material.light_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_EMISSIVE_MAP is defined
    out.emissive_map_uv = (u_material.emissive_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_METALNESS_MAP is defined
    out.metalness_map_uv = (u_material.metalness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_ROUGHNESS_MAP is defined
    out.roughness_map_uv = (u_material.roughness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_CLEARCOAT_MAP is defined
    out.clearcoat_map_uv = (u_material.clearcoat_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_CLEARCOAT_NORMAL_MAP is defined
    out.clearcoat_normal_map_uv = (u_material.clearcoat_normal_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_CLEARCOAT_ROUGHNESS_MAP is defined
    out.clearcoat_roughness_map_uv = (u_material.clearcoat_roughness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_IRIDESCENCE_MAP is defined
    out.iridescence_map_uv = (u_material.iridescence_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_IRIDESCENCE_THICKNESS_MAP is defined
    out.iridescence_thickness_map_uv = (u_material.iridescence_thickness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_ANISOTROPY_MAP is defined
    out.anisotropy_map_uv = (u_material.anisotropy_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_SHEEN_COLOR_MAP is defined
    out.sheen_color_map_uv = (u_material.sheen_color_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif

$$ if HAS_SHEEN_ROUGHNESS_MAP is defined
    out.sheen_roughness_map_uv = (u_material.sheen_roughness_map_transform * vec3<f32>(in.uv, 1.0)).xy;
$$ endif


