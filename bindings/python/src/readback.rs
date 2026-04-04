//! Python wrapper for [`ReadbackStream`].
//!
//! Exposes a non-blocking ring-buffer readback pipeline to Python.
//! See [`myth_render::core::ReadbackStream`] for the underlying
//! architecture.

use pyo3::prelude::*;

use myth_engine::render::core::ReadbackStream;

/// High-throughput, non-blocking GPU→CPU readback stream.
///
/// Created via :meth:`Renderer.create_readback_stream`. Each call to
/// :meth:`submit` records a GPU copy from the headless render target into
/// a ring-buffer slot. When the GPU completes the copy, the frame becomes
/// available via :meth:`try_recv`.
///
/// At the end of a recording session, call :meth:`flush` to block until
/// all remaining in-flight frames have been received.
///
/// Example::
///
///     renderer = myth.Renderer()
///     renderer.init_headless(800, 600)
///     scene = renderer.create_scene()
///     # … setup scene …
///
///     stream = renderer.create_readback_stream(buffer_count=3)
///
///     for i in range(100):
///         renderer.update(1.0 / 60.0)
///         renderer.render()
///         stream.submit(renderer)
///         renderer.poll_device()
///
///         frame = stream.try_recv()
///         if frame is not None:
///             process(frame)
///
///     for frame in stream.flush(renderer):
///         process(frame)
#[pyclass(unsendable, name = "ReadbackStream")]
pub struct PyReadbackStream {
    stream: ReadbackStream,
}

impl PyReadbackStream {
    pub fn new(stream: ReadbackStream) -> Self {
        Self { stream }
    }
}

#[pymethods]
impl PyReadbackStream {
    /// Submit a non-blocking copy from the headless texture to the next
    /// ring-buffer slot.
    ///
    /// Args:
    ///     renderer: The :class:`Renderer` that owns the headless texture.
    ///
    /// Raises:
    ///     RuntimeError: If the ring buffer is full (all slots in-flight).
    ///         Drain frames with :meth:`try_recv` or :meth:`flush` first.
    fn submit(&mut self, renderer: &crate::renderer::PyMythRenderer) -> PyResult<()> {
        let engine = renderer.engine_ref_pub()?;
        let device = engine
            .renderer
            .device()
            .ok_or_else(|| rt_err("renderer not initialised"))?;
        let queue = engine
            .renderer
            .queue()
            .ok_or_else(|| rt_err("renderer not initialised"))?;
        let texture = engine
            .renderer
            .headless_texture()
            .ok_or_else(|| rt_err("no headless texture — call init_headless() first"))?;

        self.stream
            .submit(device, queue, texture)
            .map_err(|e| rt_err(&e.to_string()))
    }

    /// Return the next ready frame as ``bytes``, or ``None`` if no frame
    /// is available yet.
    fn try_recv<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        match self.stream.try_recv() {
            Ok(Some(frame)) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("pixels", pyo3::types::PyBytes::new(py, &frame.pixels))?;
                dict.set_item("frame_index", frame.frame_index)?;
                Ok(Some(dict))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(rt_err(&e.to_string())),
        }
    }

    /// Block until all in-flight frames are returned.
    ///
    /// Args:
    ///     renderer: The :class:`Renderer` that owns the GPU device.
    ///
    /// Returns:
    ///     ``list[dict]``: Each dict has ``"pixels"`` (``bytes``) and
    ///     ``"frame_index"`` (``int``).
    fn flush<'py>(&mut self, py: Python<'py>, renderer: &crate::renderer::PyMythRenderer) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let engine = renderer.engine_ref_pub()?;
        let device = engine
            .renderer
            .device()
            .ok_or_else(|| rt_err("renderer not initialised"))?;

        let result = pyo3::types::PyList::empty(py);

        let flush_result = self.stream
            .flush(device)
            .map_err(|e| rt_err(&e.to_string()))?;

        for frame in flush_result {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pixels", pyo3::types::PyBytes::new(py, &frame.pixels))?;
            dict.set_item("frame_index", frame.frame_index)?;
            result.append(dict)?;
        }

        Ok(result)
    }

    /// Number of ring-buffer slots.
    #[getter]
    fn buffer_count(&self) -> usize {
        self.stream.buffer_count()
    }

    /// Total frames submitted so far.
    #[getter]
    fn frames_submitted(&self) -> u64 {
        self.stream.frames_submitted()
    }

    /// Render target dimensions as ``(width, height)``.
    #[getter]
    fn dimensions(&self) -> (u32, u32) {
        self.stream.dimensions()
    }

    fn __repr__(&self) -> String {
        let (w, h) = self.stream.dimensions();
        format!(
            "ReadbackStream({}×{}, slots={}, submitted={})",
            w,
            h,
            self.stream.buffer_count(),
            self.stream.frames_submitted(),
        )
    }
}

fn rt_err(msg: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg.to_string())
}
