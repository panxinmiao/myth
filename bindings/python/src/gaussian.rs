//! Python bindings for 3D Gaussian Splatting — cloud loading, scene integration,
//! and per-cloud rendering settings.

use pyo3::prelude::*;

use crate::scene::PyObject3D;
use crate::{with_active_scene, with_engine};

// ============================================================================
// PyGaussianCloud — opaque wrapper around a GaussianCloudHandle
// ============================================================================

/// A loaded Gaussian Splatting point cloud.
///
/// Create one via ``engine.load_gaussian_ply(path)`` (or ``engine.load_gaussian_npz(path)``)
/// and add it to a scene with ``scene.add_gaussian_cloud(name, cloud)``.
#[pyclass(name = "GaussianCloud")]
pub struct PyGaussianCloud {
    pub(crate) handle: myth_engine::GaussianCloudHandle,
}

#[pymethods]
impl PyGaussianCloud {
    /// Number of Gaussian primitives in the cloud.
    #[getter]
    fn num_points(&self) -> PyResult<usize> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.num_points)
                .unwrap_or(0)
        })
    }

    /// Spherical-harmonics degree (0–3).
    #[getter]
    fn sh_degree(&self) -> PyResult<u32> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.sh_degree)
                .unwrap_or(0)
        })
    }

    /// Axis-aligned bounding box minimum corner ``[x, y, z]``.
    #[getter]
    fn aabb_min(&self) -> PyResult<[f32; 3]> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.aabb_min.to_array())
                .unwrap_or([0.0; 3])
        })
    }

    /// Axis-aligned bounding box maximum corner ``[x, y, z]``.
    #[getter]
    fn aabb_max(&self) -> PyResult<[f32; 3]> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.aabb_max.to_array())
                .unwrap_or([0.0; 3])
        })
    }

    /// Centroid of the point cloud ``[x, y, z]``.
    #[getter]
    fn center(&self) -> PyResult<[f32; 3]> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.center.to_array())
                .unwrap_or([0.0; 3])
        })
    }

    /// Scene extent (half-diagonal of the bounding box).
    #[getter]
    fn scene_extent(&self) -> PyResult<f32> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| c.scene_extent())
                .unwrap_or(0.0)
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        with_engine(|engine| {
            if let Some(c) = engine.assets.gaussian_clouds.get(self.handle) {
                format!(
                    "GaussianCloud(num_points={}, sh_degree={})",
                    c.num_points, c.sh_degree
                )
            } else {
                "GaussianCloud(<released>)".to_string()
            }
        })
    }
}

// ============================================================================
// PyEngine extensions (load_gaussian_ply / load_gaussian_npz)
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
    let handle = with_engine(|engine| engine.assets.gaussian_clouds.add(cloud))?;
    Ok(PyGaussianCloud { handle })
}

/// Load a ``.npz`` file containing compressed 3D Gaussian Splatting data.
///
/// Returns a ``GaussianCloud`` object.
#[cfg(feature = "gaussian-npz")]
pub fn load_gaussian_npz_impl(path: &str) -> PyResult<PyGaussianCloud> {
    let file = std::fs::File::open(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open NPZ file '{path}': {e}"))
    })?;
    let reader = std::io::BufReader::new(file);
    let cloud = myth_engine::load_gaussian_npz(reader).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to parse Gaussian NPZ '{path}': {e}"
        ))
    })?;
    let handle = with_engine(|engine| engine.assets.gaussian_clouds.add(cloud))?;
    Ok(PyGaussianCloud { handle })
}

// ============================================================================
// PyScene extensions (add_gaussian_cloud)
// ============================================================================

/// Add a Gaussian splatting cloud to the active scene.
///
/// Returns an ``Object3D`` handle for positioning the cloud.
pub fn add_gaussian_cloud_impl(name: &str, cloud: &PyGaussianCloud) -> PyResult<PyObject3D> {
    let handle = cloud.handle;
    let node = with_active_scene(|scene| scene.add_gaussian_cloud(name, handle))?;
    Ok(PyObject3D::from_handle(node))
}
