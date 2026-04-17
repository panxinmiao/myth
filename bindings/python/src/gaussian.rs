//! Python bindings for 3D Gaussian Splatting — cloud loading, scene integration,
//! and per-cloud rendering settings.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::scene::PyObject3D;
use crate::with_active_scene;

// ============================================================================
// PyGaussianCloud — opaque wrapper around a loaded GaussianCloud
// ============================================================================

/// A loaded Gaussian Splatting point cloud.
///
/// Create one via ``engine.load_gaussian_ply(path)`` and add it to a scene
/// with ``scene.add_gaussian_cloud(name, cloud)``.
#[pyclass(name = "GaussianCloud")]
pub struct PyGaussianCloud {
    pub(crate) inner: Arc<myth_engine::GaussianCloud>,
}

#[pymethods]
impl PyGaussianCloud {
    /// Number of Gaussian primitives in the cloud.
    #[getter]
    fn num_points(&self) -> usize {
        self.inner.num_points
    }

    /// Spherical-harmonics degree (0–3).
    #[getter]
    fn sh_degree(&self) -> u32 {
        self.inner.sh_degree
    }

    /// Axis-aligned bounding box minimum corner ``[x, y, z]``.
    #[getter]
    fn aabb_min(&self) -> [f32; 3] {
        self.inner.aabb_min.to_array()
    }

    /// Axis-aligned bounding box maximum corner ``[x, y, z]``.
    #[getter]
    fn aabb_max(&self) -> [f32; 3] {
        self.inner.aabb_max.to_array()
    }

    /// Centroid of the point cloud ``[x, y, z]``.
    #[getter]
    fn center(&self) -> [f32; 3] {
        self.inner.center.to_array()
    }

    /// Scene extent (half-diagonal of the bounding box).
    #[getter]
    fn scene_extent(&self) -> f32 {
        self.inner.scene_extent()
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianCloud(num_points={}, sh_degree={})",
            self.inner.num_points, self.inner.sh_degree
        )
    }
}

// ============================================================================
// PyEngine extensions (load_gaussian_ply)
// ============================================================================

/// Load a ``.ply`` file containing 3D Gaussian Splatting data.
///
/// Returns a ``GaussianCloud`` object.
pub fn load_gaussian_ply_impl(path: &str) -> PyResult<PyGaussianCloud> {
    let file = std::fs::File::open(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open PLY file '{path}': {e}"))
    })?;
    let reader = std::io::BufReader::new(file);
    let cloud = myth_engine::load_gaussian_ply(reader).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to parse Gaussian PLY '{path}': {e}"
        ))
    })?;
    Ok(PyGaussianCloud {
        inner: Arc::new(cloud),
    })
}

// ============================================================================
// PyScene extensions (add_gaussian_cloud)
// ============================================================================

/// Add a Gaussian splatting cloud to the active scene.
///
/// Returns an ``Object3D`` handle for positioning the cloud.
pub fn add_gaussian_cloud_impl(name: &str, cloud: &PyGaussianCloud) -> PyResult<PyObject3D> {
    let arc = cloud.inner.clone();
    let handle = with_active_scene(|scene| scene.add_gaussian_cloud(name, arc))?;
    Ok(PyObject3D::from_handle(handle))
}
