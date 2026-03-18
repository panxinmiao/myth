//! Camera descriptors and runtime component proxies.
//!
//! * **Descriptor types** (`PerspectiveCamera`, `OrthographicCamera`) —
//!   plain-data objects used to *create* cameras via `scene.add_camera(desc)`.
//! * **Component proxies** (`PerspectiveCameraComponent`,
//!   `OrthographicCameraComponent`) — lightweight handles that read/write the
//!   *live* camera component attached to a scene node via `node.camera`.

use myth_engine::scene::camera::ProjectionType;
use myth_engine::{AntiAliasingMode, FxaaQuality, FxaaSettings, NodeHandle, TaaSettings};
use pyo3::prelude::*;

use crate::with_active_scene;

fn parse_fxaa_quality(quality: Option<&str>) -> FxaaQuality {
    match quality.unwrap_or("medium").to_lowercase().as_str() {
        "low" => FxaaQuality::Low,
        "medium" => FxaaQuality::Medium,
        "high" => FxaaQuality::High,
        _ => FxaaQuality::Medium,
    }
}

/// Anti-aliasing configuration wrapper.
#[pyclass(name = "AntiAliasing", from_py_object)]
#[derive(Clone)]
pub struct PyAntiAliasing {
    pub(crate) mode: AntiAliasingMode,
}

#[pymethods]
impl PyAntiAliasing {
    /// No anti-aliasing. Maximum performance.
    #[staticmethod]
    pub fn none() -> Self {
        Self {
            mode: AntiAliasingMode::None,
        }
    }

    /// Hardware multi-sampling.
    #[staticmethod]
    #[pyo3(signature = (samples=4))]
    pub fn msaa(samples: u32) -> Self {
        Self {
            mode: AntiAliasingMode::MSAA(samples),
        }
    }

    /// FXAA only.
    #[staticmethod]
    #[pyo3(signature = (quality=None))]
    pub fn fxaa(quality: Option<&str>) -> Self {
        let q = parse_fxaa_quality(quality);
        Self {
            mode: AntiAliasingMode::FXAA(FxaaSettings { quality: q }),
        }
    }

    /// MSAA + FXAA.
    #[staticmethod]
    #[pyo3(signature = (samples=4, quality=None))]
    pub fn msaa_fxaa(samples: u32, quality: Option<&str>) -> Self {
        let q = parse_fxaa_quality(quality);
        Self {
            mode: AntiAliasingMode::MSAA_FXAA(samples, FxaaSettings { quality: q }),
        }
    }

    /// Temporal Anti-Aliasing (Recommended for PBR).
    #[staticmethod]
    #[pyo3(signature = (feedback_weight=0.9, sharpen_intensity=0.5))]
    pub fn taa(feedback_weight: f32, sharpen_intensity: f32) -> Self {
        Self {
            mode: AntiAliasingMode::TAA(TaaSettings {
                feedback_weight,
                sharpen_intensity,
            }),
        }
    }

    /// TAA + FXAA.
    #[staticmethod]
    #[pyo3(signature = (feedback_weight=0.9, sharpen_intensity=0.5, quality=None))]
    pub fn taa_fxaa(feedback_weight: f32, sharpen_intensity: f32, quality: Option<&str>) -> Self {
        let q = parse_fxaa_quality(quality);
        Self {
            mode: AntiAliasingMode::TAA_FXAA(
                TaaSettings {
                    feedback_weight,
                    sharpen_intensity,
                },
                FxaaSettings { quality: q },
            ),
        }
    }

    fn __repr__(&self) -> String {
        match &self.mode {
            AntiAliasingMode::None => "AntiAliasing.none()".to_string(),
            AntiAliasingMode::FXAA(s) => {
                format!("AntiAliasing.fxaa(quality='{}')", s.quality.name())
            }
            AntiAliasingMode::MSAA(s) => format!("AntiAliasing.msaa(samples={})", s),
            AntiAliasingMode::MSAA_FXAA(s, _) => format!("AntiAliasing.msaa_fxaa(samples={})", s),
            AntiAliasingMode::TAA(s) => format!(
                "AntiAliasing.taa(feedback_weight={:.2}, sharpen_intensity={:.2})",
                s.feedback_weight, s.sharpen_intensity
            ),
            AntiAliasingMode::TAA_FXAA(t, _) => format!(
                "AntiAliasing.taa_fxaa(feedback_weight={:.2}, sharpen_intensity={:.2})",
                t.feedback_weight, t.sharpen_intensity
            ),
        }
    }
}

/// A perspective projection camera.
///
/// Example:
/// ```python
/// cam = myth.PerspectiveCamera(fov=60.0, near=0.1, far=1000.0)
/// node = scene.add_camera(cam)
/// node.position = [0, 2, 5]
/// scene.active_camera = node
/// ```
#[pyclass(name = "PerspectiveCamera", from_py_object)]
#[derive(Clone)]
pub struct PyPerspectiveCamera {
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub position: [f32; 3],
    pub anti_aliasing: PyAntiAliasing,
}

#[pymethods]
impl PyPerspectiveCamera {
    #[new]
    #[pyo3(signature = (fov=60.0, near=0.1, far=1000.0, aspect=0.0, position=[0.0, 0.0, 0.0], anti_aliasing=None))]
    fn new(
        fov: f32,
        near: f32,
        far: f32,
        aspect: f32,
        position: [f32; 3],
        anti_aliasing: Option<PyAntiAliasing>,
    ) -> Self {
        Self {
            fov,
            aspect,
            near,
            far,
            position,
            anti_aliasing: anti_aliasing.unwrap_or_else(PyAntiAliasing::none),
        }
    }

    #[getter]
    fn get_antialiasing(&self) -> PyAntiAliasing {
        self.anti_aliasing.clone()
    }

    #[setter]
    fn set_antialiasing(&mut self, aa: PyAntiAliasing) {
        self.anti_aliasing = aa;
    }

    #[getter]
    fn get_fov(&self) -> f32 {
        self.fov
    }
    #[setter]
    fn set_fov(&mut self, val: f32) {
        self.fov = val;
    }

    #[getter]
    fn get_aspect(&self) -> f32 {
        self.aspect
    }
    #[setter]
    fn set_aspect(&mut self, val: f32) {
        self.aspect = val;
    }

    #[getter]
    fn get_near(&self) -> f32 {
        self.near
    }
    #[setter]
    fn set_near(&mut self, val: f32) {
        self.near = val;
    }

    #[getter]
    fn get_far(&self) -> f32 {
        self.far
    }
    #[setter]
    fn set_far(&mut self, val: f32) {
        self.far = val;
    }

    fn __repr__(&self) -> String {
        format!(
            "PerspectiveCamera(fov={}, aspect={}, near={}, far={})",
            self.fov, self.aspect, self.near, self.far
        )
    }
}

/// An orthographic projection camera.
#[pyclass(name = "OrthographicCamera", from_py_object)]
#[derive(Clone)]
pub struct PyOrthographicCamera {
    pub size: f32,
    pub near: f32,
    pub far: f32,
    pub position: [f32; 3],
    pub anti_aliasing: PyAntiAliasing,
}

#[pymethods]
impl PyOrthographicCamera {
    #[new]
    #[pyo3(signature = (size=10.0, near=0.1, far=1000.0, position=[0.0, 0.0, 0.0], anti_aliasing=None))]
    fn new(
        size: f32,
        near: f32,
        far: f32,
        position: [f32; 3],
        anti_aliasing: Option<PyAntiAliasing>,
    ) -> Self {
        Self {
            size,
            near,
            far,
            position,
            anti_aliasing: anti_aliasing.unwrap_or_else(PyAntiAliasing::none),
        }
    }

    #[getter]
    fn get_antialiasing(&self) -> PyAntiAliasing {
        self.anti_aliasing.clone()
    }

    #[setter]
    fn set_antialiasing(&mut self, aa: PyAntiAliasing) {
        self.anti_aliasing = aa;
    }

    fn __repr__(&self) -> String {
        format!(
            "OrthographicCamera(size={}, near={}, far={})",
            self.size, self.near, self.far
        )
    }
}

// ============================================================================
// Component Proxies — live access to a camera on a scene node
// ============================================================================

/// Runtime proxy for a perspective camera component attached to a scene node.
///
/// Obtained via ``node.camera`` when the node carries a perspective camera.
/// Property reads/writes go directly to the engine's ECS storage.
#[pyclass(name = "PerspectiveCameraComponent")]
pub struct PyPerspectiveCameraComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PyPerspectiveCameraComponent {
    /// Vertical field of view in degrees.
    #[getter]
    fn get_fov(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.fov().to_degrees())
                .unwrap_or(60.0)
        })
    }

    #[setter]
    fn set_fov(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_fov_degrees(val);
            }
        })
    }

    #[getter]
    fn get_aspect(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.aspect())
                .unwrap_or(1.0)
        })
    }

    #[setter]
    fn set_aspect(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_aspect(val);
            }
        })
    }

    #[getter]
    fn get_near(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.near())
                .unwrap_or(0.1)
        })
    }

    #[setter]
    fn set_near(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_near(val);
            }
        })
    }

    #[getter]
    fn get_far(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.far())
                .unwrap_or(1000.0)
        })
    }

    #[setter]
    fn set_far(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_far(val);
            }
        })
    }

    /// Anti-aliasing configuration for this camera.
    #[getter]
    fn get_antialiasing(&self) -> PyResult<PyAntiAliasing> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| PyAntiAliasing {
                    mode: c.aa_mode.clone(),
                })
                .unwrap_or_else(|| PyAntiAliasing::none())
        })
    }

    #[setter]
    fn set_antialiasing(&self, aa: PyAntiAliasing) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_aa_mode(aa.mode);
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("PerspectiveCameraComponent(handle={:?})", self.handle)
    }
}

/// Runtime proxy for an orthographic camera component attached to a scene node.
///
/// Obtained via ``node.camera`` when the node carries an orthographic camera.
#[pyclass(name = "OrthographicCameraComponent")]
pub struct PyOrthographicCameraComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PyOrthographicCameraComponent {
    /// Orthographic view half-height.
    #[getter]
    fn get_size(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.ortho_size())
                .unwrap_or(10.0)
        })
    }

    #[setter]
    fn set_size(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_ortho_size(val);
            }
        })
    }

    #[getter]
    fn get_near(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.near())
                .unwrap_or(0.1)
        })
    }

    #[setter]
    fn set_near(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_near(val);
            }
        })
    }

    #[getter]
    fn get_far(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| c.far())
                .unwrap_or(1000.0)
        })
    }

    #[setter]
    fn set_far(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_far(val);
            }
        })
    }

    /// Anti-aliasing configuration for this camera.
    #[getter]
    fn get_antialiasing(&self) -> PyResult<PyAntiAliasing> {
        with_active_scene(|scene| {
            scene
                .get_camera(self.handle)
                .map(|c| PyAntiAliasing {
                    mode: c.aa_mode.clone(),
                })
                .unwrap_or_else(|| PyAntiAliasing::none())
        })
    }

    #[setter]
    fn set_antialiasing(&self, aa: PyAntiAliasing) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(cam) = scene.get_camera_mut(self.handle) {
                cam.set_aa_mode(aa.mode);
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("OrthographicCameraComponent(handle={:?})", self.handle)
    }
}

/// Inspect the camera component on `handle` and return the appropriate typed
/// proxy as a `PyAny`, or `None` if the node has no camera.
pub(crate) fn get_camera_component(
    py: Python<'_>,
    handle: NodeHandle,
) -> PyResult<Option<Py<PyAny>>> {
    let proj = with_active_scene(|scene| scene.get_camera(handle).map(|c| c.projection_type()))?;

    match proj {
        Some(ProjectionType::Perspective) => Ok(Some(
            PyPerspectiveCameraComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        Some(ProjectionType::Orthographic) => Ok(Some(
            PyOrthographicCameraComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        None => Ok(None),
    }
}
