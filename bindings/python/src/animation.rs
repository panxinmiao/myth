//! Animation bindings -- mixer, clips, actions.

use pyo3::prelude::*;

use crate::scene::PyObject3D;
use crate::with_active_scene;

/// Animation mixer attached to a node.
///
/// Use `scene.play_animation(node, "clip_name")` for simple playback,
/// or use AnimationMixer for advanced control.
#[pyclass(name = "AnimationMixer")]
pub struct PyAnimationMixer {
    pub(crate) node_handle: myth_engine::NodeHandle,
}

#[pymethods]
impl PyAnimationMixer {
    /// List all animation clip names available on this mixer.
    fn list_animations(&self) -> PyResult<Vec<String>> {
        let result = with_active_scene(|scene| {
            scene
                .animation_mixers
                .get(self.node_handle)
                .map(|m| m.list_animations())
                .unwrap_or_default()
        })?;
        Ok(result)
    }

    /// Play an animation by name.
    fn play(&self, name: &str) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mixer) = scene.animation_mixers.get_mut(self.node_handle) {
                mixer.play(name);
            }
        })?;
        Ok(())
    }

    /// Stop an animation by name.
    fn stop(&self, name: &str) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mixer) = scene.animation_mixers.get_mut(self.node_handle) {
                mixer.stop(name);
            }
        })?;
        Ok(())
    }

    /// Stop all animations.
    fn stop_all(&self) -> PyResult<()> {
        with_active_scene(|scene| {
            if let Some(mixer) = scene.animation_mixers.get_mut(self.node_handle) {
                mixer.stop_all();
            }
        })?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("AnimationMixer(node={:?})", self.node_handle)
    }
}

impl PyAnimationMixer {
    pub fn from_node(node: &PyObject3D) -> Self {
        Self {
            node_handle: node.handle,
        }
    }
}
