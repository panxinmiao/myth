$$ if map_uv is defined
    @location({{ loc.next() }}) map_uv: vec2<f32>,
$$ endif

$$ if use_normal_map is defined
    @location({{ loc.next() }}) normal_map_uv: vec2<f32>,
$$ endif

$$ if use_specular_map is defined
    @location({{ loc.next() }}) specular_map_uv: vec2<f32>,
$$ endif

$$ if use_specular_intensity_map is defined
    @location({{ loc.next() }}) specular_intensity_map_uv: vec2<f32>,
$$ endif

$$ if use_ao_map is defined
    @location({{ loc.next() }}) ao_map_uv: vec2<f32>,
$$ endif

$$ if use_light_map is defined
    @location({{ loc.next() }}) light_map_uv: vec2<f32>,
$$ endif

$$ if use_emissive_map is defined
    @location({{ loc.next() }}) emissive_map_uv: vec2<f32>,
$$ endif

$$ if use_metalness_map is defined
    @location({{ loc.next() }}) metalness_map_uv: vec2<f32>,
$$ endif

$$ if use_roughness_map is defined
    @location({{ loc.next() }}) roughness_map_uv: vec2<f32>,
$$ endif

$$ if use_clearcoat_map is defined
    @location({{ loc.next() }}) clearcoat_map_uv: vec2<f32>,
$$ endif

$$ if use_clearcoat_normal_map is defined
    @location({{ loc.next() }}) clearcoat_normal_map_uv: vec2<f32>,
$$ endif

$$ if use_clearcoat_roughness_map is defined
    @location({{ loc.next() }}) clearcoat_roughness_map_uv: vec2<f32>,
$$ endif

$$ if use_iridescence_map is defined
    @location({{ loc.next() }}) iridescence_map_uv: vec2<f32>,
$$ endif

$$ if use_iridescence_thickness_map is defined
    @location({{ loc.next() }}) iridescence_thickness_map_uv: vec2<f32>,
$$ endif

$$ if use_anisotropy_map is defined
    @location({{ loc.next() }}) anisotropy_map_uv: vec2<f32>,
$$ endif

