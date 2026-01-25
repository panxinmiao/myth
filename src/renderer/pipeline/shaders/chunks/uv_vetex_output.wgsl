$$ if HAS_MAP is defined
    @location({{ loc.next() }}) map_uv: vec2<f32>,
$$ endif

$$ if HAS_NORMAL_MAP is defined
    @location({{ loc.next() }}) normal_map_uv: vec2<f32>,
$$ endif

$$ if HAS_SPECULAR_MAP is defined
    @location({{ loc.next() }}) specular_map_uv: vec2<f32>,
$$ endif

$$ if HAS_SPECULAR_INTENSITY_MAP is defined
    @location({{ loc.next() }}) specular_intensity_map_uv: vec2<f32>,
$$ endif

$$ if HAS_AO_MAP is defined
    @location({{ loc.next() }}) ao_map_uv: vec2<f32>,
$$ endif

$$ if HAS_LIGHT_MAP is defined
    @location({{ loc.next() }}) light_map_uv: vec2<f32>,
$$ endif

$$ if HAS_EMISSIVE_MAP is defined
    @location({{ loc.next() }}) emissive_map_uv: vec2<f32>,
$$ endif

$$ if HAS_METALNESS_MAP is defined
    @location({{ loc.next() }}) metalness_map_uv: vec2<f32>,
$$ endif

$$ if HAS_ROUGHNESS_MAP is defined
    @location({{ loc.next() }}) roughness_map_uv: vec2<f32>,
$$ endif

$$ if HAS_CLEARCOAT_MAP is defined
    @location({{ loc.next() }}) clearcoat_map_uv: vec2<f32>,
$$ endif

$$ if HAS_CLEARCOAT_NORMAL_MAP is defined
    @location({{ loc.next() }}) clearcoat_normal_map_uv: vec2<f32>,
$$ endif

$$ if HAS_CLEARCOAT_ROUGHNESS_MAP is defined
    @location({{ loc.next() }}) clearcoat_roughness_map_uv: vec2<f32>,
$$ endif

$$ if HAS_IRIDESCENCE_MAP is defined
    @location({{ loc.next() }}) iridescence_map_uv: vec2<f32>,
$$ endif

$$ if HAS_IRIDESCENCE_THICKNESS_MAP is defined
    @location({{ loc.next() }}) iridescence_thickness_map_uv: vec2<f32>,
$$ endif

$$ if HAS_ANISOTROPY_MAP is defined
    @location({{ loc.next() }}) anisotropy_map_uv: vec2<f32>,
$$ endif

$$ if HAS_SHEEN_COLOR_MAP is defined
    @location({{ loc.next() }}) sheen_color_map_uv: vec2<f32>,
$$ endif

$$ if HAS_SHEEN_ROUGHNESS_MAP is defined
    @location({{ loc.next() }}) sheen_roughness_map_uv: vec2<f32>,
$$ endif

