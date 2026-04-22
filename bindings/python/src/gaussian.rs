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
    fn count(&self) -> PyResult<usize> {
        self.num_points()
    }

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

    /// Source color space for the SH-fitted color coefficients.
    #[getter]
    fn color_space(&self) -> PyResult<&'static str> {
        with_engine(|engine| {
            engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .map(|c| match c.color_space {
                    myth_engine::ColorSpace::Linear => "linear",
                    myth_engine::ColorSpace::Srgb => "srgb",
                })
                .unwrap_or("srgb")
        })
    }

    #[setter]
    fn set_color_space(&self, value: &str) -> PyResult<()> {
        let color_space = match value.to_lowercase().as_str() {
            "linear" => myth_engine::ColorSpace::Linear,
            "srgb" => myth_engine::ColorSpace::Srgb,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "color_space must be 'srgb' or 'linear'",
                ));
            }
        };

        with_engine(|engine| {
            let cloud = engine
                .assets
                .gaussian_clouds
                .get(self.handle)
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("GaussianCloud asset not found")
                })?;

            let updated_cloud = myth_engine::GaussianCloud {
                gaussians: cloud.gaussians.clone(),
                sh_coefficients: cloud.sh_coefficients.clone(),
                sh_degree: cloud.sh_degree,
                num_points: cloud.num_points,
                aabb_min: cloud.aabb_min,
                aabb_max: cloud.aabb_max,
                center: cloud.center,
                mip_splatting: cloud.mip_splatting,
                kernel_size: cloud.kernel_size,
                color_space,
                opacity_compensation: cloud.opacity_compensation,
            };

            engine
                .assets
                .gaussian_clouds
                .update(self.handle, updated_cloud)
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("GaussianCloud asset not loaded")
                })?;
            Ok(())
        })?
    }

    fn __repr__(&self) -> PyResult<String> {
        with_engine(|engine| {
            if let Some(c) = engine.assets.gaussian_clouds.get(self.handle) {
                format!(
                    "GaussianCloud(count={}, sh_degree={})",
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
    let cloud = myth_engine::load_gaussian_ply_from_source(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to load Gaussian PLY '{path}': {e}"
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
    let cloud = myth_engine::load_gaussian_npz_from_source(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to load Gaussian NPZ '{path}': {e}"
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
