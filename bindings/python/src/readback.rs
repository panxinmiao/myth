//! Python wrapper for [`ReadbackStream`].
//!
//! Provides both a non-blocking ([`try_submit`](PyReadbackStream::try_submit))
//! and blocking ([`submit_blocking`](PyReadbackStream::submit_blocking))
//! submission API. The blocking path releases the GIL via
//! [`Python::detach`] so other Python threads can run while waiting
//! for GPU completion.
//!
//! Two receive modes are available:
//!
//! - [`try_recv`](PyReadbackStream::try_recv) — returns a `dict` with
//!   freshly-allocated `bytes`.
//! - [`try_recv_into`](PyReadbackStream::try_recv_into) — writes directly
//!   into a caller-supplied `bytearray`, eliminating Python-side
//!   allocations in steady state.

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};

use myth_engine::render::core::ReadbackStream;

/// High-throughput GPU→CPU readback stream.
///
/// Created via :meth:`Renderer.create_readback_stream`. Use
/// :meth:`try_submit` for real-time streaming (frame drops OK) or
/// :meth:`submit_blocking` for offline recording (zero frame loss).
///
/// When the GPU completes a copy, the frame becomes available via
/// :meth:`try_recv` (allocating) or :meth:`try_recv_into` (zero-copy into
/// a reusable ``bytearray``).
///
/// At the end of a session, call :meth:`flush` to drain remaining
/// in-flight frames.
///
/// Example (real-time)::
///
///     stream = renderer.create_readback_stream(buffer_count=3)
///     buf = bytearray(stream.frame_byte_size)
///
///     for i in range(100):
///         renderer.update(1.0 / 60.0)
///         renderer.render()
///         stream.try_submit(renderer)
///         renderer.poll_device()
///
///         idx = stream.try_recv_into(buf)
///         if idx is not None:
///             process(idx, buf)
///
///     for frame in stream.flush(renderer):
///         process(frame["frame_index"], frame["pixels"])
///
/// Example (offline — zero frame loss)::
///
///     stream = renderer.create_readback_stream(buffer_count=3)
///     buf = bytearray(stream.frame_byte_size)
///
///     for i in range(100):
///         renderer.update(1.0 / 60.0)
///         renderer.render()
///         stream.submit_blocking(renderer)
///         renderer.poll_device()
///
///         idx = stream.try_recv_into(buf)
///         if idx is not None:
///             process(idx, buf)
///
///     for frame in stream.flush(renderer):
///         process(frame["frame_index"], frame["pixels"])
#[pyclass(unsendable, name = "ReadbackStream")]
pub struct PyReadbackStream {
    stream: ReadbackStream,
    /// Persistent buffer for zero-allocation `try_recv_into` calls.
    recv_buf: Vec<u8>,
}

impl PyReadbackStream {
    pub fn new(stream: ReadbackStream) -> Self {
        Self {
            stream,
            recv_buf: Vec::new(),
        }
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
    fn try_submit(&mut self, renderer: &crate::renderer::PyMythRenderer) -> PyResult<()> {
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
            .try_submit(device, queue, texture)
            .map_err(|e| rt_err(&e.to_string()))
    }

    /// Submit a copy, blocking when the ring buffer is full.
    ///
    /// The GIL is released during the blocking wait so that other Python
    /// threads can proceed. Completed frames are stashed internally and
    /// can be retrieved via :meth:`try_recv` or :meth:`try_recv_into`.
    ///
    /// Args:
    ///     renderer: The :class:`Renderer` that owns the headless texture.
    ///
    /// Raises:
    ///     RuntimeError: If the stash exceeds *max_stash_size*.
    #[pyo3(signature = (renderer))]
    fn submit_blocking(
        &mut self,
        py: Python<'_>,
        renderer: &crate::renderer::PyMythRenderer,
    ) -> PyResult<()> {
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

        let stream = &mut self.stream;
        py.detach(move || {
            stream.submit_blocking(device, queue, texture)
        })
        .map_err(|e: myth_engine::render::core::ReadbackError| rt_err(&e.to_string()))
    }

    /// Return the next ready frame as ``dict``, or ``None`` if no frame
    /// is available yet.
    ///
    /// The returned dict contains:
    ///
    /// - ``"pixels"``: ``bytes`` — tightly-packed pixel data.
    /// - ``"frame_index"``: ``int`` — zero-based submission index.
    ///
    /// This is the *allocating* receive path. For steady-state zero
    /// allocation, prefer :meth:`try_recv_into`.
    fn try_recv<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.stream.try_recv() {
            Ok(Some(frame)) => {
                let dict = PyDict::new(py);
                dict.set_item("pixels", PyBytes::new(py, &frame.pixels))?;
                dict.set_item("frame_index", frame.frame_index)?;
                Ok(Some(dict))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(rt_err(&e.to_string())),
        }
    }

    /// Zero-allocation receive: writes pixel data into a ``bytearray``.
    ///
    /// The buffer is automatically resized to :attr:`frame_byte_size` on
    /// the first successful receive. Subsequent calls reuse the same
    /// allocation, achieving steady-state zero allocation on both Rust
    /// and Python sides.
    ///
    /// Args:
    ///     buffer: A writable ``bytearray`` to receive pixel data.
    ///
    /// Returns:
    ///     The zero-based frame index, or ``None`` if no frame is ready.
    ///
    /// Example::
    ///
    ///     buf = bytearray(stream.frame_byte_size)
    ///     idx = stream.try_recv_into(buf)
    ///     if idx is not None:
    ///         arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    fn try_recv_into(
        &mut self,
        buffer: &Bound<'_, PyByteArray>,
    ) -> PyResult<Option<u64>> {
        match self.stream.try_recv_into(&mut self.recv_buf) {
            Ok(Some(frame_index)) => {
                let expected = self.recv_buf.len();
                if buffer.len() != expected {
                    buffer.resize(expected)?;
                }
                // SAFETY: We hold the GIL and just ensured the buffer size
                // matches. No other Python code can mutate the bytearray
                // concurrently.
                unsafe {
                    buffer
                        .as_bytes_mut()
                        .copy_from_slice(&self.recv_buf);
                }
                Ok(Some(frame_index))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(rt_err(&e.to_string())),
        }
    }

    /// Block until all in-flight frames are returned.
    ///
    /// The GIL is released during the blocking GPU poll.
    ///
    /// Args:
    ///     renderer: The :class:`Renderer` that owns the GPU device.
    ///
    /// Returns:
    ///     ``list[dict]``: Each dict has ``"pixels"`` (``bytes``) and
    ///     ``"frame_index"`` (``int``).
    fn flush<'py>(
        &mut self,
        py: Python<'py>,
        renderer: &crate::renderer::PyMythRenderer,
    ) -> PyResult<Bound<'py, PyList>> {
        let engine = renderer.engine_ref_pub()?;
        let device = engine
            .renderer
            .device()
            .ok_or_else(|| rt_err("renderer not initialised"))?;

        let stream = &mut self.stream;
        let frames = py
            .detach(move || stream.flush(device))
            .map_err(|e: myth_engine::render::core::ReadbackError| rt_err(&e.to_string()))?;

        let result = PyList::empty(py);
        for frame in frames {
            let dict = PyDict::new(py);
            dict.set_item("pixels", PyBytes::new(py, &frame.pixels))?;
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

    /// Expected byte size of one tightly-packed frame.
    ///
    /// Use this to pre-allocate a ``bytearray`` for :meth:`try_recv_into`.
    #[getter]
    fn frame_byte_size(&self) -> usize {
        self.stream.frame_byte_size()
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
