//! TextureHandle wrapper -- opaque handle for loaded textures.

use pyo3::prelude::*;

/// An opaque handle to a loaded texture.
///
/// Obtain via `ctx.load_texture()`, `ctx.load_hdr_texture()`, etc.
/// Pass to material methods like `mat.set_map(handle)`.
#[pyclass(name = "TextureHandle", skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTextureHandle {
    handle: myth_engine::TextureHandle,
}

impl PyTextureHandle {
    pub fn from_handle(handle: myth_engine::TextureHandle) -> Self {
        Self { handle }
    }

    pub fn inner(&self) -> myth_engine::TextureHandle {
        self.handle
    }
}

#[pymethods]
impl PyTextureHandle {
    fn __repr__(&self) -> String {
        format!("TextureHandle({:?})", self.handle)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}
