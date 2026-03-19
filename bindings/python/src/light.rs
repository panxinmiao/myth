//! Light descriptors and runtime component proxies.
//!
//! * **Descriptor types** (`DirectionalLight`, `PointLight`, `SpotLight`) —
//!   plain-data objects used to *create* lights via `scene.add_light(desc)`.
//! * **Component proxies** (`DirectionalLightComponent`, `PointLightComponent`,
//!   `SpotLightComponent`) — lightweight handles that read/write the *live*
//!   light component attached to a scene node via `node.light`.

use myth_engine::NodeHandle;
use myth_engine::math::Vec3;
use myth_engine::scene::light::LightKind;
use pyo3::prelude::*;

use crate::with_active_scene;

// ============================================================================
// Descriptors — used to create lights
// ============================================================================

/// A directional light descriptor (like the sun).
///
/// Illuminates the entire scene from a direction. The node's position
/// determines the direction the light comes FROM.
///
/// Args:
///     color: Light color as [r, g, b] (default: white).
///     intensity: Light intensity in lux (default: 1.0).
///     cast_shadows: Whether this light casts shadows (default: False).
#[pyclass(name = "DirectionalLight", from_py_object)]
#[derive(Clone)]
pub struct PyDirectionalLight {
    #[pyo3(get, set)]
    pub color: [f32; 3],
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub cast_shadows: bool,
}

#[pymethods]
impl PyDirectionalLight {
    #[new]
    #[pyo3(signature = (color=[1.0, 1.0, 1.0], intensity=1.0, cast_shadows=false))]
    fn new(color: [f32; 3], intensity: f32, cast_shadows: bool) -> Self {
        Self {
            color,
            intensity,
            cast_shadows,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DirectionalLight(color=[{:.2}, {:.2}, {:.2}], intensity={})",
            self.color[0], self.color[1], self.color[2], self.intensity
        )
    }
}

impl PyDirectionalLight {
    pub fn to_myth_light(&self) -> myth_engine::Light {
        let mut light = myth_engine::Light::new_directional(Vec3::from(self.color), self.intensity);
        light.cast_shadows = self.cast_shadows;
        light
    }
}

/// A point light descriptor that emits in all directions.
///
/// Args:
///     color: Light color as [r, g, b] (default: white).
///     intensity: Light intensity in candela (default: 1.0).
///     range: Maximum effective range (default: 10.0, 0 = infinite).
///     cast_shadows: Whether this light casts shadows (default: False).
#[pyclass(name = "PointLight", from_py_object)]
#[derive(Clone)]
pub struct PyPointLight {
    #[pyo3(get, set)]
    pub color: [f32; 3],
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub range: f32,
    #[pyo3(get, set)]
    pub cast_shadows: bool,
}

#[pymethods]
impl PyPointLight {
    #[new]
    #[pyo3(signature = (color=[1.0, 1.0, 1.0], intensity=1.0, range=10.0, cast_shadows=false))]
    fn new(color: [f32; 3], intensity: f32, range: f32, cast_shadows: bool) -> Self {
        Self {
            color,
            intensity,
            range,
            cast_shadows,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PointLight(color=[{:.2}, {:.2}, {:.2}], intensity={}, range={})",
            self.color[0], self.color[1], self.color[2], self.intensity, self.range
        )
    }
}

impl PyPointLight {
    pub fn to_myth_light(&self) -> myth_engine::Light {
        let mut light =
            myth_engine::Light::new_point(Vec3::from(self.color), self.intensity, self.range);
        light.cast_shadows = self.cast_shadows;
        light
    }
}

/// A spotlight descriptor that emits in a cone shape.
///
/// Args:
///     color: Light color as [r, g, b] (default: white).
///     intensity: Light intensity in candela (default: 1.0).
///     range: Maximum range (default: 10.0).
///     inner_cone: Inner cone angle in radians (default: 0.3).
///     outer_cone: Outer cone angle in radians (default: 0.5).
///     cast_shadows: Whether this light casts shadows (default: False).
#[pyclass(name = "SpotLight", from_py_object)]
#[derive(Clone)]
pub struct PySpotLight {
    #[pyo3(get, set)]
    pub color: [f32; 3],
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub range: f32,
    #[pyo3(get, set)]
    pub inner_cone: f32,
    #[pyo3(get, set)]
    pub outer_cone: f32,
    #[pyo3(get, set)]
    pub cast_shadows: bool,
}

#[pymethods]
impl PySpotLight {
    #[new]
    #[pyo3(signature = (color=[1.0, 1.0, 1.0], intensity=1.0, range=10.0, inner_cone=0.3, outer_cone=0.5, cast_shadows=false))]
    fn new(
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_cone: f32,
        outer_cone: f32,
        cast_shadows: bool,
    ) -> Self {
        Self {
            color,
            intensity,
            range,
            inner_cone,
            outer_cone,
            cast_shadows,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpotLight(intensity={}, range={}, cone=[{}, {}])",
            self.intensity, self.range, self.inner_cone, self.outer_cone
        )
    }
}

impl PySpotLight {
    pub fn to_myth_light(&self) -> myth_engine::Light {
        let mut light = myth_engine::Light::new_spot(
            Vec3::from(self.color),
            self.intensity,
            self.range,
            self.inner_cone,
            self.outer_cone,
        );
        light.cast_shadows = self.cast_shadows;
        light
    }
}

// ============================================================================
// Component Proxies — live access to a light on a scene node
// ============================================================================

/// Runtime proxy for a directional light component attached to a scene node.
///
/// Obtained via ``node.light`` when the node carries a directional light.
/// Property reads/writes go directly to the engine's ECS storage.
#[pyclass(name = "DirectionalLightComponent")]
pub struct PyDirectionalLightComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PyDirectionalLightComponent {
    #[getter]
    fn get_color(&self) -> PyResult<[f32; 3]> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.color.to_array())
                .unwrap_or([1.0; 3])
        })
    }

    #[setter]
    fn set_color(&self, color: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.color = Vec3::from(color);
            }
        })
    }

    #[getter]
    fn get_intensity(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.intensity)
                .unwrap_or(1.0)
        })
    }

    #[setter]
    fn set_intensity(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.intensity = val;
            }
        })
    }

    #[getter]
    fn get_cast_shadows(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.cast_shadows)
                .unwrap_or(false)
        })
    }

    #[setter]
    fn set_cast_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.cast_shadows = val;
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("DirectionalLightComponent(handle={:?})", self.handle)
    }
}

/// Runtime proxy for a point light component attached to a scene node.
///
/// Obtained via ``node.light`` when the node carries a point light.
#[pyclass(name = "PointLightComponent")]
pub struct PyPointLightComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PyPointLightComponent {
    #[getter]
    fn get_color(&self) -> PyResult<[f32; 3]> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.color.to_array())
                .unwrap_or([1.0; 3])
        })
    }

    #[setter]
    fn set_color(&self, color: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.color = Vec3::from(color);
            }
        })
    }

    #[getter]
    fn get_intensity(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.intensity)
                .unwrap_or(1.0)
        })
    }

    #[setter]
    fn set_intensity(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.intensity = val;
            }
        })
    }

    #[getter]
    fn get_range(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .and_then(|l| match &l.kind {
                    LightKind::Point(p) => Some(p.range),
                    _ => None,
                })
                .unwrap_or(10.0)
        })
    }

    #[setter]
    fn set_range(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle)
                && let LightKind::Point(ref mut p) = light.kind
            {
                p.range = val;
            }
        })
    }

    #[getter]
    fn get_cast_shadows(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.cast_shadows)
                .unwrap_or(false)
        })
    }

    #[setter]
    fn set_cast_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.cast_shadows = val;
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("PointLightComponent(handle={:?})", self.handle)
    }
}

/// Runtime proxy for a spot light component attached to a scene node.
///
/// Obtained via ``node.light`` when the node carries a spot light.
#[pyclass(name = "SpotLightComponent")]
pub struct PySpotLightComponent {
    pub(crate) handle: NodeHandle,
}

#[pymethods]
impl PySpotLightComponent {
    #[getter]
    fn get_color(&self) -> PyResult<[f32; 3]> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.color.to_array())
                .unwrap_or([1.0; 3])
        })
    }

    #[setter]
    fn set_color(&self, color: [f32; 3]) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.color = Vec3::from(color);
            }
        })
    }

    #[getter]
    fn get_intensity(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.intensity)
                .unwrap_or(1.0)
        })
    }

    #[setter]
    fn set_intensity(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.intensity = val;
            }
        })
    }

    #[getter]
    fn get_range(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .and_then(|l| match &l.kind {
                    LightKind::Spot(s) => Some(s.range),
                    _ => None,
                })
                .unwrap_or(10.0)
        })
    }

    #[setter]
    fn set_range(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle)
                && let LightKind::Spot(ref mut s) = light.kind
            {
                s.range = val;
            }
        })
    }

    #[getter]
    fn get_inner_cone(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .and_then(|l| match &l.kind {
                    LightKind::Spot(s) => Some(s.inner_cone),
                    _ => None,
                })
                .unwrap_or(0.3)
        })
    }

    #[setter]
    fn set_inner_cone(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle)
                && let LightKind::Spot(ref mut s) = light.kind
            {
                s.inner_cone = val;
            }
        })
    }

    #[getter]
    fn get_outer_cone(&self) -> PyResult<f32> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .and_then(|l| match &l.kind {
                    LightKind::Spot(s) => Some(s.outer_cone),
                    _ => None,
                })
                .unwrap_or(0.5)
        })
    }

    #[setter]
    fn set_outer_cone(&self, val: f32) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle)
                && let LightKind::Spot(ref mut s) = light.kind
            {
                s.outer_cone = val;
            }
        })
    }

    #[getter]
    fn get_cast_shadows(&self) -> PyResult<bool> {
        with_active_scene(|scene| {
            scene
                .get_light(self.handle)
                .map(|l| l.cast_shadows)
                .unwrap_or(false)
        })
    }

    #[setter]
    fn set_cast_shadows(&self, val: bool) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(light) = scene.get_light_mut(self.handle) {
                light.cast_shadows = val;
            }
        })
    }

    fn __repr__(&self) -> String {
        format!("SpotLightComponent(handle={:?})", self.handle)
    }
}

/// Inspect the light component on `handle` and return the appropriate typed
/// proxy as a `PyAny`, or `None` if the node has no light.
pub(crate) fn get_light_component(
    py: Python<'_>,
    handle: NodeHandle,
) -> PyResult<Option<Py<PyAny>>> {
    let kind = with_active_scene(|scene| {
        scene.get_light(handle).map(|l| match &l.kind {
            LightKind::Directional(_) => 0u8,
            LightKind::Point(_) => 1,
            LightKind::Spot(_) => 2,
        })
    })?;

    match kind {
        Some(0) => Ok(Some(
            PyDirectionalLightComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        Some(1) => Ok(Some(
            PyPointLightComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        Some(2) => Ok(Some(
            PySpotLightComponent { handle }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        _ => Ok(None),
    }
}
