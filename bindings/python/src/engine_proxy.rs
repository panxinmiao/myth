//! PyEngine — The engine context proxy passed to Python callbacks.

use pyo3::prelude::*;

use crate::scene::PyScene;
use crate::texture::PyTextureHandle;
use crate::with_engine;
use myth_engine::SceneExt;

/// Engine context, available inside `@app.init` and `@app.update` callbacks.
#[pyclass(name = "Engine")]
pub struct PyEngine {
    _private: (),
}

impl PyEngine {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

#[pymethods]
impl PyEngine {
    // ---- Scene Management ----

    /// Create a new scene and set it as the active scene.
    fn create_scene(&self) -> PyResult<PyScene> {
        with_engine(|engine| {
            engine.scene_manager.create_active();
        })?;
        Ok(PyScene::new())
    }

    /// Get the currently active scene.
    fn active_scene(&self) -> PyResult<Option<PyScene>> {
        with_engine(|engine| engine.scene_manager.active_scene().map(|_| PyScene::new()))
    }

    // ---- Asset Loading ----

    /// Load a 2D texture from a file path.
    #[pyo3(signature = (path, color_space="srgb", generate_mipmaps=true))]
    fn load_texture(
        &self,
        path: &str,
        color_space: &str,
        generate_mipmaps: bool,
    ) -> PyResult<PyTextureHandle> {
        let cs = match color_space {
            "linear" | "Linear" => myth_engine::ColorSpace::Linear,
            _ => myth_engine::ColorSpace::Srgb,
        };

        with_engine(|engine| {
            engine
                .assets
                .load_texture_blocking(path, cs, generate_mipmaps)
                .map(PyTextureHandle::from_handle)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to load texture '{path}': {e}"
                    ))
                })
        })?
    }

    /// Load an HDR environment texture.
    fn load_hdr_texture(&self, path: &str) -> PyResult<PyTextureHandle> {
        with_engine(|engine| {
            engine
                .assets
                .load_hdr_texture_blocking(path)
                .map(PyTextureHandle::from_handle)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to load HDR texture '{path}': {e}"
                    ))
                })
        })?
    }

    /// Load a glTF/GLB model and add it to the active scene.
    fn load_gltf(&self, path: &str) -> PyResult<crate::scene::PyObject3D> {
        with_engine(|engine| {
            let assets = engine.assets.clone();
            let prefab = myth_engine::assets::GltfLoader::load(path, assets).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to load glTF '{path}': {e}"
                ))
            })?;

            let scene = engine
                .scene_manager
                .active_scene_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No active scene"))?;

            let handle = scene.instantiate(&prefab);
            Ok(crate::scene::PyObject3D::from_handle(handle))
        })?
    }

    // ---- Timing ----

    #[getter]
    fn get_time(&self) -> PyResult<f32> {
        with_engine(|engine| engine.time())
    }

    #[getter]
    fn get_frame_count(&self) -> PyResult<u64> {
        with_engine(|engine| engine.frame_count())
    }

    // ---- Input ----

    #[getter]
    fn get_input(&self) -> crate::input::PyInput {
        crate::input::PyInput::new()
    }

    // ---- Window ----

    /// Set the window title.
    ///
    /// Only works inside `@app.init` / `@app.update` callbacks when using
    /// the built-in `App` window.  Silently ignored when no window is
    /// available (e.g. `Renderer` mode).
    fn set_title(&self, title: &str) {
        crate::with_window(|window| {
            window.set_title(title);
        });
    }

    // ---- Gaussian Splatting ----

    /// Load a ``.ply`` file containing 3D Gaussian Splatting data.
    ///
    /// Returns a ``GaussianCloud`` object that can be added to a scene.
    fn load_gaussian_ply(&self, path: &str) -> PyResult<crate::gaussian::PyGaussianCloud> {
        crate::gaussian::load_gaussian_ply_impl(path)
    }

    /// Load a compressed ``.npz`` file containing 3D Gaussian Splatting data.
    ///
    /// Returns a ``GaussianCloud`` object that can be added to a scene.
    #[cfg(feature = "gaussian-npz")]
    fn load_gaussian_npz(&self, path: &str) -> PyResult<crate::gaussian::PyGaussianCloud> {
        crate::gaussian::load_gaussian_npz_impl(path)
    }
}
