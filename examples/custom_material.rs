//! [gallery]
//! name = "Custom Material"
//! category = "Materials"
//! description = "Registers a custom WGSL shader template and visualises normals in world space."
//! order = 130
//!

//! Custom Material Example — Normal Visualization
//!
//! Demonstrates the custom shader registration API by defining a material
//! that visualises surface normals as RGB colours.
//!
//! Key steps shown:
//! 1. Define a material struct with `#[myth_material(shader = "custom_normal")]`
//! 2. Write a WGSL template that uses engine chunks (`vertex_output_def`,
//!    `fragment_output_def`, `pack_fragment_output`)
//! 3. Register the template at init time via `register_shader_template`

use glam::Vec4;
use myth::prelude::*;
use myth_resources::myth_material;

// ── Custom WGSL Template ───────────────────────────────────────────────
//
// Embedded at compile-time.  The template engine resolves `{% include %}`
// directives against the built-in chunk library, so `vertex_output`
// and `fragment_output` work out of the box.

const CUSTOM_NORMAL_SHADER: &str = r#"
{{ vertex_input_code }}
{{ binding_code }}
{$ include 'core/vertex_output' $}
{$ include 'core/fragment_output' $}

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);

    $$ if HAS_NORMAL is defined
    var local_normal = vec3<f32>(in.normal.xyz);
    $$ endif

    var local_pos = vec4<f32>(local_position, 1.0);

    let world_pos = u_model.world_matrix * local_pos;
    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_NORMAL is defined
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    $$ if HAS_UV is defined
    out.uv = in.uv;
    $$ endif

    {$ include 'mixins/uv_vertex' $}
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // Map world-space normal from [-1,1] to [0,1] for visualisation.
    $$ if HAS_NORMAL is defined
    let n = normalize(in.normal);
    let color = vec4<f32>(n * 0.5 + 0.5, u_material.opacity);
    $$ else
    let color = vec4<f32>(0.5, 0.5, 1.0, u_material.opacity);
    $$ endif

    return pack_fragment_output(color * u_material.tint);
}
"#;

// ── Material Definition ────────────────────────────────────────────────

#[myth_material(shader = "custom_normal")]
pub struct NormalMaterial {
    /// Tint color multiplied with the normal visualisation.
    #[uniform(default = "Vec4::ONE")]
    pub tint: Vec4,

    /// Opacity value.
    #[uniform(default = "1.0")]
    pub opacity: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,
}

impl NormalMaterial {
    #[must_use]
    pub fn new() -> Self {
        Self::from_uniforms(NormalUniforms::default())
    }

    #[must_use]
    pub fn with_tint(self, tint: Vec4) -> Self {
        self.uniforms.write().tint = tint;
        self
    }
}

impl Default for NormalMaterial {
    fn default() -> Self {
        Self::new()
    }
}

// ── Application ────────────────────────────────────────────────────────

struct CustomMaterialDemo {
    controls: OrbitControls,
}

impl AppHandler for CustomMaterialDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. Register the custom shader template
        engine
            .renderer
            .register_shader_template("custom_normal", CUSTOM_NORMAL_SHADER);

        // 2. Build the scene
        let scene = engine.scene_manager.create_active();

        // Spawn a sphere with the custom normal-visualisation material
        let mat = Material::new_custom(NormalMaterial::new());
        scene.spawn_sphere(2.0, mat, &engine.assets);

        // Camera
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam)
            .set_position(0.0, 2.0, 6.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.0, 6.0), Vec3::ZERO),
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<CustomMaterialDemo>()
}
