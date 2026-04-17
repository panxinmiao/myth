//! Scene, Object3D, MeshComponent, FrameState — high-level wrappers around
//! `myth_engine::Scene`.

use myth_engine::NodeHandle;
use myth_engine::math::{Quat, Vec3};
use myth_engine::resources::tone_mapping::AgxLook;
use pyo3::prelude::*;

use myth_engine::ToneMappingMode;
use myth_engine::resources::mesh::Mesh;
use myth_engine::scene::camera::Camera;

use crate::animation::PyAnimationMixer;
use crate::camera::{PyOrthographicCamera, PyPerspectiveCamera, get_camera_component};
use crate::light::{PyDirectionalLight, PyPointLight, PySpotLight, get_light_component};
use crate::texture::PyTextureHandle;
use crate::{extract_geometry_handle, extract_material_handle, with_active_scene, with_engine};

// ============================================================================
// PyScene — A thin proxy wrapping the *active* scene.
// ============================================================================

/// A scene object that holds all objects, lights, and cameras for rendering.
///
/// Obtained via `engine.create_scene()`.
#[pyclass(name = "Scene")]
pub struct PyScene {
    pub(crate) scene_id: u32,
}

impl PyScene {
    pub fn new() -> Self {
        Self { scene_id: 0 }
    }
}

#[pymethods]
impl PyScene {
    // ----------------------------------------------------------------
    // Mesh
    // ----------------------------------------------------------------

    /// Add a mesh to the scene from geometry + material.
    ///
    /// Returns an `Object3D` handle that can be used to move, rotate, scale the mesh.
    ///
    /// # Arguments
    /// * `geometry` — BoxGeometry, SphereGeometry, PlaneGeometry ...
    /// * `material` — BasicMaterial, PhongMaterial, PhysicalMaterial ...
    fn add_mesh(
        &self,
        geometry: &Bound<'_, PyAny>,
        material: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject3D> {
        let geo_h = extract_geometry_handle(geometry)?;
        let mat_h = extract_material_handle(material)?;
        let handle = with_active_scene(|scene| {
            let mesh = Mesh::new(geo_h, mat_h);
            scene.add_mesh(mesh)
        })?;
        Ok(PyObject3D { handle })
    }

    // ----------------------------------------------------------------
    // Camera
    // ----------------------------------------------------------------

    /// Add a camera to the scene.
    ///
    /// Args:
    ///     camera: A PerspectiveCamera or OrthographicCamera.
    ///
    /// Returns an Object3D node handle. Set `scene.active_camera = node` to render from it.
    fn add_camera(&self, camera: &Bound<'_, PyAny>) -> PyResult<PyObject3D> {
        if let Ok(persp) = camera.extract::<PyPerspectiveCamera>() {
            let handle = with_engine(|engine| {
                let aspect = if persp.aspect > 0.0 {
                    persp.aspect
                } else {
                    let (w, h) = engine.size();
                    if h > 0 { w as f32 / h as f32 } else { 1.0 }
                };
                let scene = engine.scene_manager.active_scene_mut().unwrap();
                let mut cam = Camera::new_perspective(persp.fov, aspect, persp.near);
                cam.set_aa_mode(persp.anti_aliasing.mode);
                let h = scene.add_camera(cam);
                if let Some(node) = scene.get_node_mut(h) {
                    node.transform.position =
                        Vec3::new(persp.position[0], persp.position[1], persp.position[2]);
                }
                h
            })?;
            return Ok(PyObject3D { handle });
        }
        if let Ok(ortho) = camera.extract::<PyOrthographicCamera>() {
            let handle = with_engine(|engine| {
                let (w, h) = engine.size();
                let aspect = if h > 0 { w as f32 / h as f32 } else { 1.0 };
                let scene = engine.scene_manager.active_scene_mut().unwrap();
                let mut cam = Camera::new_orthographic(ortho.size, aspect, ortho.near, ortho.far);
                cam.set_aa_mode(ortho.anti_aliasing.mode);
                let h = scene.add_camera(cam);
                if let Some(node) = scene.get_node_mut(h) {
                    node.transform.position =
                        Vec3::new(ortho.position[0], ortho.position[1], ortho.position[2]);
                }
                h
            })?;
            return Ok(PyObject3D { handle });
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected PerspectiveCamera or OrthographicCamera",
        ))
    }

    // ----------------------------------------------------------------
    // Light
    // ----------------------------------------------------------------

    /// Add a light to the scene.
    ///
    /// Args:
    ///     light: A DirectionalLight, PointLight, or SpotLight.
    fn add_light(&self, light: &Bound<'_, PyAny>) -> PyResult<PyObject3D> {
        if let Ok(dir) = light.extract::<PyDirectionalLight>() {
            let handle = with_active_scene(|scene| scene.add_light(dir.to_myth_light()))?;
            return Ok(PyObject3D { handle });
        }
        if let Ok(pt) = light.extract::<PyPointLight>() {
            let handle = with_active_scene(|scene| scene.add_light(pt.to_myth_light()))?;
            return Ok(PyObject3D { handle });
        }
        if let Ok(sp) = light.extract::<PySpotLight>() {
            let handle = with_active_scene(|scene| scene.add_light(sp.to_myth_light()))?;
            return Ok(PyObject3D { handle });
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected DirectionalLight, PointLight, or SpotLight",
        ))
    }

    // ----------------------------------------------------------------
    // Gaussian Splatting
    // ----------------------------------------------------------------

    /// Add a Gaussian splatting point cloud to the scene.
    ///
    /// Args:
    ///     name: Display name for the scene node.
    ///     cloud: A ``GaussianCloud`` loaded via ``engine.load_gaussian_ply()``.
    ///
    /// Returns an ``Object3D`` handle for positioning the cloud.
    fn add_gaussian_cloud(
        &self,
        name: &str,
        cloud: &crate::gaussian::PyGaussianCloud,
    ) -> PyResult<PyObject3D> {
        crate::gaussian::add_gaussian_cloud_impl(name, cloud)
    }

    // ----------------------------------------------------------------
    // Hierarchy
    // ----------------------------------------------------------------

    /// Attach a child node to a parent node.
    fn attach(&self, child: &PyObject3D, parent: &PyObject3D) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.attach(child.handle, parent.handle);
        })?;
        Ok(())
    }

    /// Find a node by name. Returns None if not found.
    fn find_node_by_name(&self, name: &str) -> PyResult<Option<PyObject3D>> {
        let result = with_active_scene(|scene| {
            scene
                .find_node_by_name(name)
                .map(|h| PyObject3D { handle: h })
        })?;
        Ok(result)
    }

    // ----------------------------------------------------------------
    // Active Camera
    // ----------------------------------------------------------------

    /// Set the active camera for rendering.
    #[setter]
    fn set_active_camera(&self, cam: &PyObject3D) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.active_camera = Some(cam.handle);
        })?;
        Ok(())
    }

    #[getter]
    fn get_active_camera(&self) -> PyResult<Option<PyObject3D>> {
        let result =
            with_active_scene(|scene| scene.active_camera.map(|h| PyObject3D { handle: h }))?;
        Ok(result)
    }

    // ----------------------------------------------------------------
    // Background
    // ----------------------------------------------------------------

    /// Set the background to a solid color (r, g, b in 0..1).
    fn set_background_color(&self, r: f32, g: f32, b: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.set_background_color(r, g, b);
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Environment (IBL / Skybox)
    // ----------------------------------------------------------------

    /// Set the environment map for IBL lighting.
    fn set_environment_map(&self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        with_active_scene(|scene| {
            scene.environment.set_env_map(Some(th));
        })?;
        Ok(())
    }

    /// Set the environment map intensity.
    fn set_environment_intensity(&self, intensity: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.environment.set_intensity(intensity);
        })?;
        Ok(())
    }

    /// Set the ambient light color for environment lighting.
    fn set_ambient_light(&self, r: f32, g: f32, b: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.environment.set_ambient_light(Vec3::new(r, g, b));
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Bloom
    // ----------------------------------------------------------------

    /// Convenience: enable/disable bloom with optional strength and radius.
    #[pyo3(signature = (enabled, strength=None, radius=None))]
    fn set_bloom(&self, enabled: bool, strength: Option<f32>, radius: Option<f32>) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.bloom.set_enabled(enabled);
            if let Some(s) = strength {
                scene.bloom.set_strength(s);
            }
            if let Some(r) = radius {
                scene.bloom.set_radius(r);
            }
        })?;
        Ok(())
    }

    fn set_bloom_enabled(&self, enabled: bool) -> PyResult<()> {
        with_active_scene(|scene| scene.bloom.set_enabled(enabled))?;
        Ok(())
    }

    fn set_bloom_strength(&self, strength: f32) -> PyResult<()> {
        with_active_scene(|scene| scene.bloom.set_strength(strength))?;
        Ok(())
    }

    fn set_bloom_radius(&self, radius: f32) -> PyResult<()> {
        with_active_scene(|scene| scene.bloom.set_radius(radius))?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // SSAO
    // ----------------------------------------------------------------

    fn set_ssao_enabled(&self, enabled: bool) -> PyResult<()> {
        with_active_scene(|scene| scene.ssao.set_enabled(enabled))?;
        Ok(())
    }

    fn set_ssao_radius(&self, radius: f32) -> PyResult<()> {
        with_active_scene(|scene| scene.ssao.set_radius(radius))?;
        Ok(())
    }

    fn set_ssao_bias(&self, bias: f32) -> PyResult<()> {
        with_active_scene(|scene| scene.ssao.set_bias(bias))?;
        Ok(())
    }

    fn set_ssao_intensity(&self, intensity: f32) -> PyResult<()> {
        with_active_scene(|scene| scene.ssao.set_intensity(intensity))?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Tone Mapping
    // ----------------------------------------------------------------

    /// Convenience: set tone mapping mode (and optionally exposure).
    ///
    /// Supported modes: "linear", "neutral", "reinhard", "cineon", "aces", "agx"
    #[pyo3(signature = (mode, exposure=None))]
    fn set_tone_mapping(&self, mode: &str, exposure: Option<f32>) -> PyResult<()> {
        let tm = match mode.to_lowercase().as_str() {
            "linear" => ToneMappingMode::Linear,
            "neutral" => ToneMappingMode::Neutral,
            "reinhard" => ToneMappingMode::Reinhard,
            "cineon" => ToneMappingMode::Cineon,
            "aces" | "aces_filmic" | "acesfilmic" => ToneMappingMode::ACESFilmic,
            "agx" => ToneMappingMode::AgX(AgxLook::None),
            "agx_punchy" => ToneMappingMode::AgX(AgxLook::Punchy),
            _ => {
                log::warn!("Unknown tone mapping mode '{mode}', using Neutral");
                ToneMappingMode::Neutral
            }
        };
        with_active_scene(|scene| {
            scene.tone_mapping.mode = tm;
            scene.tone_mapping.set_exposure(exposure.unwrap_or(1.0));
        })?;
        Ok(())
    }

    /// Set the tone mapping mode.
    ///
    /// Supported modes: "linear", "neutral", "reinhard", "cineon", "aces", "agx"
    fn set_tone_mapping_mode(&self, mode: &str) -> PyResult<()> {
        self.set_tone_mapping(mode, None)
    }

    // ----------------------------------------------------------------
    // Animation
    // ----------------------------------------------------------------

    /// Play a named animation clip on a node.
    fn play_animation(&self, node: &PyObject3D, name: &str) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.play_animation(node.handle, name);
        })?;
        Ok(())
    }

    /// Play any available animation on a node (simple convenience).
    fn play_if_any_animation(&self, node: &PyObject3D) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.play_if_any_animation(node.handle);
        })?;
        Ok(())
    }

    /// Alias: play any available animation.
    fn play_any_animation(&self, node: &PyObject3D) -> PyResult<()> {
        self.play_if_any_animation(node)
    }

    /// List animation clip names available on a node.
    fn list_animations(&self, node: &PyObject3D) -> PyResult<Vec<String>> {
        let result = with_active_scene(|scene| {
            scene
                .animation_mixers
                .get(node.handle)
                .map(|m| m.list_animations())
                .unwrap_or_default()
        })?;
        Ok(result)
    }

    /// Get the animation mixer for a node (for advanced control).
    fn get_animation_mixer(&self, node: &PyObject3D) -> PyResult<Option<PyAnimationMixer>> {
        let result = with_active_scene(|scene| {
            if scene.animation_mixers.get(node.handle).is_some() {
                Some(PyAnimationMixer {
                    node_handle: node.handle,
                })
            } else {
                None
            }
        })?;
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("Scene(id={})", self.scene_id)
    }
}

// ============================================================================
// PyObject3D — A node reference
// ============================================================================

/// A 3D object in the scene — references a scene node by handle.
///
/// Provides position, rotation, scale, look_at, and other common operations.
#[pyclass(name = "Object3D", from_py_object)]
#[derive(Clone)]
pub struct PyObject3D {
    pub(crate) handle: NodeHandle,
}

impl PyObject3D {
    pub fn from_handle(handle: NodeHandle) -> Self {
        Self { handle }
    }
}

#[pymethods]
impl PyObject3D {
    // ----------------------------------------------------------------
    // Position
    // ----------------------------------------------------------------

    #[getter]
    fn get_position(&self) -> PyResult<[f32; 3]> {
        let result = with_active_scene(|scene| {
            scene
                .get_node(self.handle)
                .map(|n| n.transform.position.to_array())
                .unwrap_or([0.0; 3])
        })?;
        Ok(result)
    }

    #[setter]
    fn set_position(&self, pos: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.position = Vec3::new(pos[0], pos[1], pos[2]);
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Rotation (Euler angles in radians)
    // ----------------------------------------------------------------

    #[getter]
    fn get_rotation(&self) -> PyResult<[f32; 3]> {
        let result = with_active_scene(|scene| {
            scene
                .get_node(self.handle)
                .map(|n| {
                    let (x, y, z) = n
                        .transform
                        .rotation
                        .to_euler(myth_engine::math::EulerRot::XYZ);
                    [x, y, z]
                })
                .unwrap_or([0.0; 3])
        })?;
        Ok(result)
    }

    #[setter]
    fn set_rotation(&self, rot: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation =
                    Quat::from_euler(myth_engine::math::EulerRot::XYZ, rot[0], rot[1], rot[2]);
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Scale
    // ----------------------------------------------------------------

    #[getter]
    fn get_scale(&self) -> PyResult<[f32; 3]> {
        let result = with_active_scene(|scene| {
            scene
                .get_node(self.handle)
                .map(|n| n.transform.scale.to_array())
                .unwrap_or([1.0; 3])
        })?;
        Ok(result)
    }

    #[setter]
    fn set_scale(&self, s: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.scale = Vec3::new(s[0], s[1], s[2]);
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Rotation convenience (Euler degrees)
    // ----------------------------------------------------------------

    /// Get rotation as Euler angles in degrees [x, y, z].
    #[getter]
    fn get_rotation_euler(&self) -> PyResult<[f32; 3]> {
        let r = self.get_rotation()?;
        Ok([r[0].to_degrees(), r[1].to_degrees(), r[2].to_degrees()])
    }

    /// Set rotation from Euler angles in degrees [x, y, z].
    #[setter]
    fn set_rotation_euler(&self, deg: [f32; 3]) -> PyResult<()> {
        self.set_rotation([
            deg[0].to_radians(),
            deg[1].to_radians(),
            deg[2].to_radians(),
        ])?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Incremental rotation helpers
    // ----------------------------------------------------------------

    /// Rotate around the local X axis by `angle` radians.
    fn rotate_x(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation *= Quat::from_rotation_x(angle);
            }
        })?;
        Ok(())
    }

    /// Rotate around the local Y axis by `angle` radians.
    fn rotate_y(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation *= Quat::from_rotation_y(angle);
            }
        })?;
        Ok(())
    }

    /// Rotate around the local Z axis by `angle` radians.
    fn rotate_z(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation *= Quat::from_rotation_z(angle);
            }
        })?;
        Ok(())
    }

    /// Rotate around the **world** X axis by `angle` radians.
    fn rotate_world_x(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation = Quat::from_rotation_x(angle) * node.transform.rotation;
            }
        })?;
        Ok(())
    }

    /// Rotate around the **world** Y axis by `angle` radians.
    fn rotate_world_y(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation = Quat::from_rotation_y(angle) * node.transform.rotation;
            }
        })?;
        Ok(())
    }

    /// Rotate around the **world** Z axis by `angle` radians.
    fn rotate_world_z(&self, angle: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.rotation = Quat::from_rotation_z(angle) * node.transform.rotation;
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Uniform Scale Helper
    // ----------------------------------------------------------------

    /// Set uniform scale (single float).
    fn set_uniform_scale(&self, s: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.transform.scale = Vec3::splat(s);
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Visibility
    // ----------------------------------------------------------------

    #[getter]
    fn get_visible(&self) -> PyResult<bool> {
        let result =
            with_active_scene(|scene| scene.get_node(self.handle).is_none_or(|n| n.visible))?;
        Ok(result)
    }

    #[setter]
    fn set_visible(&self, v: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                node.visible = v;
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Cast / Receive Shadows
    // ----------------------------------------------------------------

    #[getter]
    fn get_cast_shadows(&self) -> PyResult<bool> {
        let result = with_active_scene(|scene| {
            scene
                .meshes
                .get(self.handle)
                .is_some_and(|m| m.cast_shadows)
        })?;
        Ok(result)
    }

    #[setter]
    fn set_cast_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.meshes.get_mut(self.handle) {
                mesh.cast_shadows = val;
            }
        })?;
        Ok(())
    }

    #[getter]
    fn get_receive_shadows(&self) -> PyResult<bool> {
        let result = with_active_scene(|scene| {
            scene
                .meshes
                .get(self.handle)
                .is_some_and(|m| m.receive_shadows)
        })?;
        Ok(result)
    }

    #[setter]
    fn set_receive_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.meshes.get_mut(self.handle) {
                mesh.receive_shadows = val;
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Look At
    // ----------------------------------------------------------------

    /// Rotate this node to look at a world-space target position.
    fn look_at(&self, target: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(node) = scene.get_node_mut(self.handle) {
                let pos = node.transform.position;
                let tgt = Vec3::new(target[0], target[1], target[2]);
                let dir = (tgt - pos).normalize_or_zero();
                if dir.length_squared() > 0.0 {
                    node.transform.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, dir);
                }
            }
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Name
    // ----------------------------------------------------------------

    #[getter]
    fn get_name(&self) -> PyResult<Option<String>> {
        let result = with_active_scene(|scene| scene.get_name(self.handle).map(String::from))?;
        Ok(result)
    }

    #[setter]
    fn set_name(&self, name: &str) -> PyResult<()> {
        with_active_scene(|scene| {
            scene.set_name(self.handle, name);
        })?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Component Proxies
    // ----------------------------------------------------------------

    /// Access the light component on this node, if any.
    ///
    /// Returns a typed proxy (`DirectionalLightComponent`,
    /// `PointLightComponent`, or `SpotLightComponent`) depending on the
    /// light kind, or `None` if this node has no light.
    #[getter]
    fn get_light(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        get_light_component(py, self.handle)
    }

    /// Access the camera component on this node, if any.
    ///
    /// Returns a typed proxy (`PerspectiveCameraComponent` or
    /// `OrthographicCameraComponent`) depending on the projection type,
    /// or `None` if this node has no camera.
    #[getter]
    fn get_camera(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        get_camera_component(py, self.handle)
    }

    /// Access the mesh component on this node, if any.
    ///
    /// Returns a `MeshComponent` proxy, or `None` if this node has no mesh.
    #[getter]
    fn get_mesh(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        get_mesh_component(py, self.handle)
    }

    fn __repr__(&self) -> PyResult<String> {
        let name = with_active_scene(|scene| {
            scene
                .get_name(self.handle)
                .map(String::from)
                .unwrap_or_else(|| format!("{:?}", self.handle))
        })?;
        Ok(format!("Object3D(name='{name}')"))
    }
}

// ============================================================================
// PyFrameState — Read-only frame timing information
// ============================================================================

/// Per-frame state information (delta_time, elapsed, dimensions).
#[pyclass(name = "FrameState")]
pub struct PyFrameState {
    /// Time elapsed since last frame (seconds).
    #[pyo3(get)]
    pub delta_time: f32,

    /// Total time since application start (seconds).
    #[pyo3(get)]
    pub elapsed: f32,

    /// Total frame count since start.
    #[pyo3(get)]
    pub frame_count: u64,
}

impl PyFrameState {
    pub fn new(elapsed: f32, dt: f32, frame_count: u64) -> Self {
        Self {
            delta_time: dt,
            elapsed,
            frame_count,
        }
    }
}

#[pymethods]
impl PyFrameState {
    /// Alias for `delta_time`.
    #[getter]
    fn get_dt(&self) -> f32 {
        self.delta_time
    }

    /// Alias for `elapsed`.
    #[getter]
    fn get_time(&self) -> f32 {
        self.elapsed
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameState(dt={:.4}, elapsed={:.2}, frame={})",
            self.delta_time, self.elapsed, self.frame_count
        )
    }
}

// ============================================================================
// MeshComponent — live proxy for mesh data on a scene node
// ============================================================================

/// Runtime proxy for a mesh component attached to a scene node.
///
/// Obtained via ``node.mesh``. Allows reading/writing per-instance mesh
/// properties such as shadow flags and render order.
#[pyclass(name = "MeshComponent")]
pub struct PyMeshComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PyMeshComponent {
    #[getter]
    fn get_visible(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_mesh(self.handle)
                .map(|m| m.visible)
                .unwrap_or(true)
        })
    }

    #[setter]
    fn set_visible(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.get_mesh_mut(self.handle) {
                mesh.visible = val;
            }
        })
    }

    #[getter]
    fn get_cast_shadows(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_mesh(self.handle)
                .map(|m| m.cast_shadows)
                .unwrap_or(true)
        })
    }

    #[setter]
    fn set_cast_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.get_mesh_mut(self.handle) {
                mesh.cast_shadows = val;
            }
        })
    }

    #[getter]
    fn get_receive_shadows(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_mesh(self.handle)
                .map(|m| m.receive_shadows)
                .unwrap_or(true)
        })
    }

    #[setter]
    fn set_receive_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.get_mesh_mut(self.handle) {
                mesh.receive_shadows = val;
            }
        })
    }

    #[getter]
    fn get_render_order(&self) -> PyResult<i32> {
        with_active_scene(|scene| {
            scene
                .get_mesh(self.handle)
                .map(|m| m.render_order)
                .unwrap_or(0)
        })
    }

    #[setter]
    fn set_render_order(&self, val: i32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mesh) = scene.get_mesh_mut(self.handle) {
                mesh.render_order = val;
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("MeshComponent(handle={:?})", self.handle)
    }
}

/// Return a `MeshComponent` proxy if the node has a mesh, else `None`.
pub(crate) fn get_mesh_component(
    py: Python<'_>,
    handle: NodeHandle,
) -> PyResult<Option<Py<PyAny>>> {
    let has_mesh = with_active_scene(|scene| scene.get_mesh(handle).is_some())?;
    if has_mesh {
        Ok(Some(
            PyMeshComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        ))
    } else {
        Ok(None)
    }
}
