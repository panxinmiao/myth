//! OrbitControls — Three.js-style orbit camera controller.

use glam::Vec3;
use pyo3::prelude::*;

use crate::scene::PyObject3D;
use crate::with_engine;

/// Three.js-style orbit camera controls.
#[pyclass(name = "OrbitControls")]
pub struct PyOrbitControls {
    inner: myth_engine::OrbitControls,
}

#[pymethods]
impl PyOrbitControls {
    #[new]
    #[pyo3(signature = (position=vec![0.0, 2.0, 5.0], target=vec![0.0, 0.0, 0.0]))]
    fn new(position: Vec<f32>, target: Vec<f32>) -> Self {
        let pos = if position.len() >= 3 {
            Vec3::new(position[0], position[1], position[2])
        } else {
            Vec3::new(0.0, 2.0, 5.0)
        };
        let tgt = if target.len() >= 3 {
            Vec3::new(target[0], target[1], target[2])
        } else {
            Vec3::ZERO
        };
        Self {
            inner: myth_engine::OrbitControls::new(pos, tgt),
        }
    }

    /// Update the orbit controls. Call every frame.
    fn update(&mut self, camera: &PyObject3D, dt: f32) -> PyResult<()> {
        with_engine(|engine| {
            let scene = engine.scene_manager.active_scene_mut().unwrap();
            let handle = camera.handle;

            // Read FOV from camera component
            let fov = scene
                .cameras
                .get(handle)
                .map_or(60.0_f32.to_radians(), |c| c.fov());

            // Update orbit controls with the node's transform
            if let Some(node) = scene.get_node_mut(handle) {
                self.inner
                    .update(&mut node.transform, &engine.input, fov, dt);
            }
        })?;
        Ok(())
    }

    /// Set the orbit target point.
    fn set_target(&mut self, target: [f32; 3]) {
        self.inner.set_target(Vec3::from(target));
    }

    /// Adjust orbit position and target to frame (fit) a given node.
    fn fit(&mut self, node: &PyObject3D) -> PyResult<()> {
        with_engine(|engine| {
            let scene = engine.scene_manager.active_scene_mut().unwrap();
            scene.update_subtree(node.handle);
            if let Some(bbox) = scene.get_bbox_of_node(node.handle, &engine.assets) {
                self.inner.fit(&bbox);
            }
        })?;
        Ok(())
    }

    #[getter]
    fn get_enable_damping(&self) -> bool {
        self.inner.enable_damping
    }
    #[setter]
    fn set_enable_damping(&mut self, v: bool) {
        self.inner.enable_damping = v;
    }

    #[getter]
    fn get_damping_factor(&self) -> f32 {
        self.inner.damping_factor
    }
    #[setter]
    fn set_damping_factor(&mut self, v: f32) {
        self.inner.damping_factor = v;
    }

    #[getter]
    fn get_rotate_speed(&self) -> f32 {
        self.inner.rotate_speed
    }
    #[setter]
    fn set_rotate_speed(&mut self, v: f32) {
        self.inner.rotate_speed = v;
    }

    #[getter]
    fn get_zoom_speed(&self) -> f32 {
        self.inner.zoom_speed
    }
    #[setter]
    fn set_zoom_speed(&mut self, v: f32) {
        self.inner.zoom_speed = v;
    }

    #[getter]
    fn get_pan_speed(&self) -> f32 {
        self.inner.pan_speed
    }
    #[setter]
    fn set_pan_speed(&mut self, v: f32) {
        self.inner.pan_speed = v;
    }

    #[getter]
    fn get_min_distance(&self) -> f32 {
        self.inner.min_distance
    }
    #[setter]
    fn set_min_distance(&mut self, v: f32) {
        self.inner.min_distance = v;
    }

    #[getter]
    fn get_max_distance(&self) -> f32 {
        self.inner.max_distance
    }
    #[setter]
    fn set_max_distance(&mut self, v: f32) {
        self.inner.max_distance = v;
    }

    fn __repr__(&self) -> String {
        "OrbitControls(...)".to_string()
    }
}
