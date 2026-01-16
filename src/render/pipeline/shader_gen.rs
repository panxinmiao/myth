use crate::render::pipeline::vertex::GeneratedVertexLayout;
use serde::{Serialize, Serializer};
use serde::ser::SerializeMap;
use super::shader_manager::{get_env, LocationAllocator};
use crate::resources::material::MaterialFeatures;
use crate::resources::geometry::GeometryFeatures;
use crate::scene::scene::SceneFeatures;
use minijinja::value::Value;


#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShaderCompilationOptions {
    pub mat_features: MaterialFeatures,
    pub geo_features: GeometryFeatures,
    pub scene_features: SceneFeatures,
}

impl Serialize for ShaderCompilationOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;

        // 展开 features
        if self.mat_features.contains(MaterialFeatures::USE_MAP) {map.serialize_entry("use_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_NORMAL_MAP) { map.serialize_entry("use_normal_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_ROUGHNESS_MAP) { map.serialize_entry("use_roughness_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_METALNESS_MAP) { map.serialize_entry("use_metalness_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_EMISSIVE_MAP) { map.serialize_entry("use_emissive_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_AO_MAP) { map.serialize_entry("use_ao_map", &true)?; }

        if self.mat_features.contains(MaterialFeatures::USE_SPECULAR_MAP) { map.serialize_entry("use_specular_map", &true)?; }
        if self.mat_features.contains(MaterialFeatures::USE_IBL) { map.serialize_entry("USE_IBL", &true)?; }


        if self.geo_features.contains(GeometryFeatures::HAS_UV) { map.serialize_entry("has_uv", &true)?; }
        if self.geo_features.contains(GeometryFeatures::HAS_NORMAL) { map.serialize_entry("has_normal", &true)?; }
        if self.geo_features.contains(GeometryFeatures::USE_VERTEX_COLOR) { map.serialize_entry("use_vertex_color", &true)?; }
        if self.geo_features.contains(GeometryFeatures::USE_TANGENT) { map.serialize_entry("use_tangent", &true)?; }
        if self.geo_features.contains(GeometryFeatures::USE_MORPHING) { map.serialize_entry("use_morphing", &true)?; }
        if self.geo_features.contains(GeometryFeatures::USE_SKINNING) { map.serialize_entry("use_skinning", &true)?; }

        if self.scene_features.contains(SceneFeatures::USE_ENV_MAP) { map.serialize_entry("use_env_map", &true)?; }
        // other scene features can be added here
        
        map.end()
    }
}

#[derive(Serialize)]
struct ShaderContext<'a> {
    #[serde(flatten)]
    options: &'a ShaderCompilationOptions,

    vertex_input_code: Option<&'a str>,
    binding_code: &'a str,
    
    loc: Value, 
}

pub struct ShaderGenerator;

impl ShaderGenerator {
    pub fn generate_shader(
        geometry_layout: &GeneratedVertexLayout,
        global_binding_code: &str,
        object_binding_code: &str,
        material_binding_code: &str,
        template_name: &str,
        options: &ShaderCompilationOptions,
    ) -> String {
        let env = get_env();
        let allocator = LocationAllocator::new();
        let loc_value = Value::from_object(allocator);

        let binding_code = format!("{}\n{}\n{}", global_binding_code, material_binding_code, object_binding_code);

        // 构建合并的 Context
        let ctx = ShaderContext {
            options,
            vertex_input_code: Some(&geometry_layout.vertex_input_code),
            binding_code: &binding_code,
            loc: loc_value,
        };

        let template_name = format!("templates/{}", template_name);

        let template = env.get_template(&template_name)
            .expect("Shader template not found");
        
        let source = template.render(&ctx)
            .expect("Shader render failed");

        format!("// === Auto-generated Unified Shader ===\n{}", source)
    }


}