//! MythRenderer — GUI-agnostic engine wrapper.
//!
//! `MythRenderer` exposes the Myth engine without coupling to winit. Users
//! can drive the render-loop themselves from *any* Python windowing library
//! (glfw, PySide, wxPython, SDL2, …) by providing a raw platform window handle.
//!
//! # Window-handle protocol
//!
//! Most Python GUI toolkits expose the native window identifier:
//!
//! | Library   | API                                   |
//! |-----------|---------------------------------------|
//! | glfw      | `glfw.get_win32_window(win)` (Win32)  |
//! | PySide6   | `int(widget.winId())`                 |
//! | wxPython  | `win.GetHandle()`                     |
//! | SDL2      | `SDL_GetWindowWMInfo` → hwnd          |
//! | Tk        | `root.winfo_id()`                     |
//!
//! Pass the integer handle to [`MythRenderer.init_with_handle`].

#[cfg(target_os = "windows")]
use std::num::NonZero;

use pyo3::prelude::*;

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle,
};

use myth_engine::{Engine, RendererInitConfig, RendererSettings, SceneExt};

use crate::{clear_engine_ptr, set_engine_ptr};

// ---------------------------------------------------------------------------
// RawWindow — platform wrapper that implements raw-window-handle traits
// ---------------------------------------------------------------------------

/// A thin wrapper holding raw platform handles so that wgpu can create a
/// surface for an externally-owned window.
struct RawWindow {
    raw_window: RawWindowHandle,
    raw_display: RawDisplayHandle,
}

// SAFETY: The handles are plain integer/pointer values identifying
// OS-level resources that persist as long as the host window is alive.
// The Python caller is responsible for keeping the window alive.
unsafe impl Send for RawWindow {}
unsafe impl Sync for RawWindow {}

impl HasWindowHandle for RawWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        // SAFETY: the handle was constructed from a valid platform window
        Ok(unsafe { WindowHandle::borrow_raw(self.raw_window) })
    }
}

impl HasDisplayHandle for RawWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // SAFETY: the display handle was constructed from valid platform data
        Ok(unsafe { DisplayHandle::borrow_raw(self.raw_display) })
    }
}

// ---------------------------------------------------------------------------
// Platform-specific constructors
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn build_raw_window(window_handle: isize) -> PyResult<RawWindow> {
    use raw_window_handle::{Win32WindowHandle, WindowsDisplayHandle};

    let hwnd = NonZero::new(window_handle).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("window_handle must be a non-zero HWND")
    })?;
    let mut wh = Win32WindowHandle::new(hwnd);

    // Attempt to fill hinstance via GetModuleHandleW(NULL)
    let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) };
    if !hinstance.is_null() {
        wh.hinstance = NonZero::new(hinstance as isize);
    }

    Ok(RawWindow {
        raw_window: RawWindowHandle::Win32(wh),
        raw_display: RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
    })
}

#[cfg(target_os = "windows")]
unsafe extern "system" {
    fn GetModuleHandleW(lpModuleName: *const u16) -> *mut std::ffi::c_void;
}

#[cfg(target_os = "macos")]
fn build_raw_window(window_handle: isize) -> PyResult<RawWindow> {
    use raw_window_handle::{AppKitDisplayHandle, AppKitWindowHandle};

    // On macOS the handle is an NSView pointer
    let ns_view =
        std::ptr::NonNull::new(window_handle as *mut std::ffi::c_void).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "window_handle must be a non-null NSView pointer",
            )
        })?;
    let wh = AppKitWindowHandle::new(ns_view);

    Ok(RawWindow {
        raw_window: RawWindowHandle::AppKit(wh),
        raw_display: RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
    })
}

#[cfg(target_os = "linux")]
fn build_raw_window(window_handle: isize) -> PyResult<RawWindow> {
    use raw_window_handle::{XlibDisplayHandle, XlibWindowHandle};

    let xlib_window = window_handle as u64;
    let wh = XlibWindowHandle::new(xlib_window.into());

    // NULL display → wgpu will open its own X connection
    let dh = XlibDisplayHandle::new(None, 0);

    Ok(RawWindow {
        raw_window: RawWindowHandle::Xlib(wh),
        raw_display: RawDisplayHandle::Xlib(dh),
    })
}

// Fallback for unsupported platforms (compile-time error avoidance)
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn build_raw_window(_window_handle: isize) -> PyResult<RawWindow> {
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Raw window handle init is not supported on this platform",
    ))
}

// ---------------------------------------------------------------------------
// RenderPath enum (Python-facing)
// ---------------------------------------------------------------------------

/// Render pipeline path.
///
/// - ``RenderPath.BASIC``: Forward LDR + MSAA.
/// - ``RenderPath.HIGH_FIDELITY``: HDR + post-processing (bloom, SSAO, tone mapping, …).
#[pyclass(name = "RenderPath", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRenderPath {
    /// Forward LDR + MSAA
    #[pyo3(name = "BASIC")]
    Basic = 0,
    /// HDR + post-processing (bloom, SSAO, tone mapping, etc.)
    #[pyo3(name = "HIGH_FIDELITY")]
    HighFidelity = 1,
}

#[pymethods]
impl PyRenderPath {
    fn __repr__(&self) -> &'static str {
        match self {
            Self::Basic => "RenderPath.BASIC",
            Self::HighFidelity => "RenderPath.HIGH_FIDELITY",
        }
    }
}

/// Parse a render path from either a `PyRenderPath` enum or a legacy string.
pub(crate) fn parse_render_path(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<String> {
    // Try enum first
    if let Ok(e) = obj.extract::<PyRenderPath>() {
        return Ok(match e {
            PyRenderPath::Basic => "basic".to_string(),
            PyRenderPath::HighFidelity => "high_fidelity".to_string(),
        });
    }
    // Fall back to string
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "render_path must be a RenderPath enum (e.g. myth.RenderPath.HIGH_FIDELITY) or a string",
    ))
}

// ---------------------------------------------------------------------------
// build_settings (shared with app.rs)
// ---------------------------------------------------------------------------

pub(crate) fn build_settings(render_path: &str, vsync: bool) -> RendererSettings {
    let path = match render_path {
        "hdr" | "high" | "high_fidelity" | "HighFidelity" => myth_engine::RenderPath::HighFidelity,
        _ => myth_engine::RenderPath::BasicForward,
    };
    RendererSettings {
        path,
        vsync,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// MythRenderer Python class
// ---------------------------------------------------------------------------

/// A low-level, GUI-agnostic renderer that can be driven from *any*
/// Python event loop (glfw, PySide, wxPython, SDL2, …).
///
/// Typical usage:
///
/// ```python
/// import myth
/// import glfw
///
/// renderer = myth.Renderer(render_path="high_fidelity")
///
/// glfw.init()
/// window = glfw.create_window(1280, 720, "Hello", None, None)
/// hwnd = glfw.get_win32_window(window)  # platform-specific
///
/// renderer.init_with_handle(hwnd, 1280, 720)
///
/// scene = renderer.create_scene()
/// # … setup scene …
///
/// while not glfw.window_should_close(window):
///     glfw.poll_events()
///     renderer.update(1.0 / 60.0)
///     renderer.render()
///
/// renderer.dispose()
/// ```
#[pyclass(unsendable, name = "Renderer")]
pub struct PyMythRenderer {
    engine: Option<Box<Engine>>,
    render_path: String,
    vsync: bool,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    /// Internal stream for the simple recording API.
    active_stream: Option<myth_engine::render::core::ReadbackStream>,
}

#[pymethods]
impl PyMythRenderer {
    #[new]
    #[pyo3(signature = (
        render_path = None,
        vsync = true,
    ))]
    fn new(render_path: Option<&Bound<'_, PyAny>>, vsync: bool) -> PyResult<Self> {
        let rp = match render_path {
            Some(obj) => parse_render_path(obj)?,
            None => "basic".to_string(),
        };
        Ok(Self {
            engine: None,
            render_path: rp,
            vsync,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            active_stream: None,
        })
    }

    /// Initialize the renderer with a raw platform window handle.
    ///
    /// Args:
    ///     window_handle: Platform-specific integer handle
    ///         - **Windows**: HWND (from ``glfw.get_win32_window()``, ``int(widget.winId())``, etc.)
    ///         - **macOS**: NSView pointer
    ///         - **Linux/X11**: X11 Window ID
    ///     width: Initial framebuffer width in pixels.
    ///     height: Initial framebuffer height in pixels.
    fn init_with_handle(&mut self, window_handle: isize, width: u32, height: u32) -> PyResult<()> {
        if self.engine.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Renderer already initialized",
            ));
        }

        let raw_window = build_raw_window(window_handle)?;
        let settings = build_settings(&self.render_path, self.vsync);

        let mut engine = Engine::new(RendererInitConfig::default(), settings);
        pollster::block_on(engine.init(raw_window, width, height)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Renderer initialization failed: {e}"
            ))
        })?;

        // Set the initial screen size so that OrbitControls (and anything else
        // that queries Input::screen_size) works correctly from the first frame.
        // Engine::init does NOT call inject_resize, so we do it here.
        engine.resize(width, height);

        self.engine = Some(Box::new(engine));

        // Set ENGINE_PTR so that all proxy objects (Scene, Object3D, Material, …)
        // can access the engine through the thread-local. The pointer is stable
        // because Engine lives on the heap via Box, and we only clear it on dispose().
        set_engine_ptr(self.engine.as_mut().unwrap());

        let now = std::time::Instant::now();
        self.start_time = now;
        self.last_frame_time = now;

        Ok(())
    }

    /// Notify the renderer that the window has been resized.
    ///
    /// Call this from the host GUI library's resize callback.
    #[pyo3(signature = (width, height))]
    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        let engine = self.engine_mut()?;
        engine.resize(width, height);
        Ok(())
    }

    /// Advance the engine state by one frame.
    ///
    /// If ``dt`` is ``None``, the renderer calculates delta time automatically
    /// from wall-clock time.
    #[pyo3(signature = (dt=None))]
    fn update(&mut self, dt: Option<f32>) -> PyResult<()> {
        let now = std::time::Instant::now();
        let delta = dt.unwrap_or_else(|| now.duration_since(self.last_frame_time).as_secs_f32());
        self.last_frame_time = now;

        let engine = self.engine_mut()?;
        engine.update(delta);
        Ok(())
    }

    /// Render one frame.
    ///
    /// This calls the engine's full render pipeline (extract → prepare →
    /// queue → render) and presents to the surface.
    fn render(&mut self) -> PyResult<()> {
        let engine = self.engine_mut()?;
        engine.render_active_scene();
        engine.renderer.maybe_prune();
        Ok(())
    }

    /// Convenience: call ``update()`` + ``render()`` in one shot.
    #[pyo3(signature = (dt=None))]
    fn frame(&mut self, dt: Option<f32>) -> PyResult<()> {
        self.update(dt)?;
        self.render()
    }

    // ----------------------------------------------------------------
    // Scene / Asset management (mirrors PyEngine API)
    // ----------------------------------------------------------------

    /// Create a new scene and set it as the active scene.
    fn create_scene(&mut self) -> PyResult<crate::scene::PyScene> {
        let engine = self.engine_mut()?;
        engine.scene_manager.create_active();
        Ok(crate::scene::PyScene::new())
    }

    /// Get the currently active scene.
    fn active_scene(&mut self) -> PyResult<Option<crate::scene::PyScene>> {
        let engine = self.engine_mut()?;
        Ok(engine
            .scene_manager
            .active_scene()
            .map(|_| crate::scene::PyScene::new()))
    }

    /// Load a 2D texture from a file path.
    #[pyo3(signature = (path, color_space="srgb", generate_mipmaps=true))]
    fn load_texture(
        &mut self,
        path: &str,
        color_space: &str,
        generate_mipmaps: bool,
    ) -> PyResult<crate::texture::PyTextureHandle> {
        let cs = match color_space {
            "linear" | "Linear" => myth_engine::ColorSpace::Linear,
            _ => myth_engine::ColorSpace::Srgb,
        };
        let engine = self.engine_mut()?;
        engine
            .assets
            .load_texture_blocking(path, cs, generate_mipmaps)
            .map(crate::texture::PyTextureHandle::from_handle)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load texture '{path}': {e}"
                ))
            })
    }

    /// Load an HDR environment texture.
    fn load_hdr_texture(&mut self, path: &str) -> PyResult<crate::texture::PyTextureHandle> {
        let engine = self.engine_mut()?;
        engine
            .assets
            .load_hdr_texture_blocking(path)
            .map(crate::texture::PyTextureHandle::from_handle)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load HDR texture '{path}': {e}"
                ))
            })
    }

    /// Load a glTF/GLB model and add it to the active scene.
    fn load_gltf(&mut self, path: &str) -> PyResult<crate::scene::PyObject3D> {
        let engine = self.engine_mut()?;
        let assets = engine.assets.clone();
        let prefab = myth_engine::assets::GltfLoader::load(path, assets).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load glTF '{path}': {e}"
            ))
        })?;
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No active scene"))?;
        let handle = scene.instantiate(&prefab);
        Ok(crate::scene::PyObject3D::from_handle(handle))
    }

    // ----------------------------------------------------------------
    // Input forwarding
    // ----------------------------------------------------------------

    /// Inject a key-down event.
    fn inject_key_down(&mut self, key: &str) -> PyResult<()> {
        let engine = self.engine_mut()?;
        if let Some(k) = crate::input::parse_key(key) {
            engine
                .input
                .inject_key(k, myth_engine::resources::input::ButtonState::Pressed);
        }
        Ok(())
    }

    /// Inject a key-up event.
    fn inject_key_up(&mut self, key: &str) -> PyResult<()> {
        let engine = self.engine_mut()?;
        if let Some(k) = crate::input::parse_key(key) {
            engine
                .input
                .inject_key(k, myth_engine::resources::input::ButtonState::Released);
        }
        Ok(())
    }

    /// Inject mouse movement (position in pixels).
    fn inject_mouse_move(&mut self, x: f32, y: f32) -> PyResult<()> {
        let engine = self.engine_mut()?;
        engine.input.inject_mouse_position(x, y);
        Ok(())
    }

    /// Inject a mouse button press.
    ///
    /// Args:
    ///     button: 0 = left, 1 = middle, 2 = right
    fn inject_mouse_down(&mut self, button: u32) -> PyResult<()> {
        let engine = self.engine_mut()?;
        let btn = match button {
            0 => myth_engine::resources::input::MouseButton::Left,
            1 => myth_engine::resources::input::MouseButton::Middle,
            2 => myth_engine::resources::input::MouseButton::Right,
            _ => myth_engine::resources::input::MouseButton::Other(button as u16),
        };
        engine
            .input
            .inject_mouse_button(btn, myth_engine::resources::input::ButtonState::Pressed);
        Ok(())
    }

    /// Inject a mouse button release.
    ///
    /// Args:
    ///     button: 0 = left, 1 = middle, 2 = right
    fn inject_mouse_up(&mut self, button: u32) -> PyResult<()> {
        let engine = self.engine_mut()?;
        let btn = match button {
            0 => myth_engine::resources::input::MouseButton::Left,
            1 => myth_engine::resources::input::MouseButton::Middle,
            2 => myth_engine::resources::input::MouseButton::Right,
            _ => myth_engine::resources::input::MouseButton::Other(button as u16),
        };
        engine
            .input
            .inject_mouse_button(btn, myth_engine::resources::input::ButtonState::Released);
        Ok(())
    }

    /// Inject scroll wheel input.
    fn inject_scroll(&mut self, dx: f32, dy: f32) -> PyResult<()> {
        let engine = self.engine_mut()?;
        engine.input.inject_scroll(dx, dy);
        Ok(())
    }

    // ----------------------------------------------------------------
    // Headless / Readback
    // ----------------------------------------------------------------

    /// Initialize the renderer in headless (offscreen) mode.
    ///
    /// No window or surface is created. An offscreen render target of the
    /// specified dimensions is allocated, suitable for server-side rendering
    /// and GPU readback.
    ///
    /// Args:
    ///     width: Render target width in pixels.
    ///     height: Render target height in pixels.
    ///     format: Pixel format string (``"rgba8"`` or ``"rgba16float"``).
    ///         Defaults to ``"rgba8"``.
    #[pyo3(signature = (width, height, format=None))]
    fn init_headless(
        &mut self,
        width: u32,
        height: u32,
        format: Option<&str>,
    ) -> PyResult<()> {
        if self.engine.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Renderer already initialized",
            ));
        }

        let target_format = match format {
            Some("rgba16float" | "rgba16" | "hdr") => {
                Some(myth_engine::resources::PixelFormat::Rgba16Float)
            }
            Some("rgba8") | None => None,
            Some(other) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unsupported format: {other} (use 'rgba8' or 'rgba16float')"
                )));
            }
        };

        let settings = build_settings(&self.render_path, self.vsync);
        let mut engine = Engine::new(RendererInitConfig::default(), settings);
        pollster::block_on(engine.init_headless(width, height, target_format)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Headless initialization failed: {e}"
            ))
        })?;

        self.engine = Some(Box::new(engine));
        set_engine_ptr(self.engine.as_mut().unwrap());

        let now = std::time::Instant::now();
        self.start_time = now;
        self.last_frame_time = now;

        Ok(())
    }

    /// Read back the current render target as raw bytes.
    ///
    /// Returns:
    ///     ``bytes``: Tightly-packed pixel data (RGBA8 = 4 bytes/px,
    ///     RGBA16Float = 8 bytes/px). Row order is top-to-bottom.
    fn readback_pixels<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let engine = self.engine_mut()?;
        let pixels = engine.readback_pixels().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("readback failed: {e}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &pixels))
    }

    /// Create a :class:`ReadbackStream` for non-blocking, high-throughput
    /// GPU→CPU readback.
    ///
    /// Args:
    ///     buffer_count: Number of ring-buffer slots (default 3).
    ///
    /// Returns:
    ///     A :class:`ReadbackStream` instance.
    #[pyo3(signature = (buffer_count=3))]
    fn create_readback_stream(
        &self,
        buffer_count: usize,
    ) -> PyResult<crate::readback::PyReadbackStream> {
        let engine = self.engine_ref()?;
        let stream = engine.renderer.create_readback_stream(buffer_count).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to create readback stream: {e}"
            ))
        })?;
        Ok(crate::readback::PyReadbackStream::new(stream))
    }

    /// Drive pending GPU callbacks without blocking.
    ///
    /// Call this once per frame in a readback-stream loop so that
    /// ``map_async`` callbacks fire and frames become available via
    /// :meth:`ReadbackStream.try_recv`.
    fn poll_device(&self) -> PyResult<()> {
        let engine = self.engine_ref()?;
        engine.renderer.poll_device();
        Ok(())
    }

    // ----------------------------------------------------------------
    // Simple Recording API
    // ----------------------------------------------------------------

    /// Begin a recording session with the given number of ring-buffer slots.
    ///
    /// This is the **simple mode** counterpart to
    /// :meth:`create_readback_stream`. It creates an internal stream and
    /// enables :meth:`render_and_record` / :meth:`try_pull_frame` /
    /// :meth:`flush_recording`.
    ///
    /// Args:
    ///     buffer_count: Number of ring-buffer slots (default 3).
    ///
    /// Raises:
    ///     RuntimeError: If a recording session is already active.
    #[pyo3(signature = (buffer_count=3))]
    fn start_recording(&mut self, buffer_count: usize) -> PyResult<()> {
        if self.active_stream.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "recording session already active — call flush_recording() first",
            ));
        }
        let engine = self.engine_ref()?;
        let stream = engine.renderer.create_readback_stream(buffer_count).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to create readback stream: {e}"
            ))
        })?;
        self.active_stream = Some(stream);
        Ok(())
    }

    /// Update, render, and record one frame — all in a single call.
    ///
    /// Equivalent to ``update(dt) → render() → submit → poll_device``.
    /// Pull completed frames with :meth:`try_pull_frame`.
    ///
    /// Args:
    ///     dt: Delta time in seconds. If ``None``, computed from wall-clock.
    ///
    /// Raises:
    ///     RuntimeError: If no recording session is active.
    #[pyo3(signature = (dt=None))]
    fn render_and_record(&mut self, dt: Option<f32>) -> PyResult<()> {
        if self.active_stream.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "no active recording — call start_recording() first",
            ));
        }

        // ---- update + render ----
        let now = std::time::Instant::now();
        let delta = dt.unwrap_or_else(|| now.duration_since(self.last_frame_time).as_secs_f32());
        self.last_frame_time = now;

        let engine = self.engine.as_deref_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Renderer not initialized")
        })?;
        engine.update(delta);
        engine.render_active_scene();

        // ---- submit ----
        let device = engine.renderer.device().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("device not available")
        })?;
        let queue = engine.renderer.queue().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("queue not available")
        })?;
        let texture = engine.renderer.headless_texture().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("no headless texture")
        })?;

        self.active_stream
            .as_mut()
            .unwrap()
            .submit(device, queue, texture)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("submit failed: {e}"))
            })?;

        // ---- poll ----
        engine.renderer.poll_device();

        Ok(())
    }

    /// Return the next completed frame as ``dict``, or ``None``.
    ///
    /// The returned dict has:
    ///   - ``"pixels"``: ``bytes`` — tightly-packed pixel data.
    ///   - ``"frame_index"``: ``int`` — zero-based index.
    ///
    /// Raises:
    ///     RuntimeError: If no recording session is active.
    fn try_pull_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let stream = self.active_stream.as_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "no active recording — call start_recording() first",
            )
        })?;
        match stream.try_recv() {
            Ok(Some(frame)) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("pixels", pyo3::types::PyBytes::new(py, &frame.pixels))?;
                dict.set_item("frame_index", frame.frame_index)?;
                Ok(Some(dict))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    /// Block until all in-flight frames are received, then end the session.
    ///
    /// Returns:
    ///     ``list[dict]``: All remaining frames. Each dict has ``"pixels"``
    ///     (``bytes``) and ``"frame_index"`` (``int``).
    fn flush_recording<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let mut stream = self.active_stream.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "no active recording — call start_recording() first",
            )
        })?;
        let engine = self.engine_ref()?;
        let device = engine.renderer.device().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("device not available")
        })?;

        let result = pyo3::types::PyList::empty(py);

        let flush_result = stream
            .flush(device)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("flush failed: {e}"))
            })?;

        for frame in flush_result {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("pixels", pyo3::types::PyBytes::new(py, &frame.pixels))?;
            dict.set_item("frame_index", frame.frame_index)?;
            result.append(dict)?;
        }

        Ok(result)
    }

    // ----------------------------------------------------------------
    // Timing
    // ----------------------------------------------------------------

    #[getter]
    fn get_time(&self) -> PyResult<f32> {
        Ok(self.engine_ref()?.time())
    }

    #[getter]
    fn get_frame_count(&self) -> PyResult<u64> {
        Ok(self.engine_ref()?.frame_count())
    }

    #[getter]
    fn get_input(&self) -> PyResult<crate::input::PyInput> {
        self.engine_ref()?;
        Ok(crate::input::PyInput::new())
    }

    // ----------------------------------------------------------------
    // Context manager & cleanup
    // ----------------------------------------------------------------

    /// Release all GPU resources.
    fn dispose(&mut self) {
        if self.engine.is_some() {
            clear_engine_ptr();
        }
        self.engine = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.dispose();
        false
    }

    // ----------------------------------------------------------------
    // Properties
    // ----------------------------------------------------------------

    #[getter]
    fn get_render_path(&self) -> &str {
        &self.render_path
    }

    #[getter]
    fn get_vsync(&self) -> bool {
        self.vsync
    }

    fn __repr__(&self) -> String {
        let init = if self.engine.is_some() {
            "initialized"
        } else {
            "not initialized"
        };
        format!(
            "Renderer(render_path='{}', vsync={}, {})",
            self.render_path, self.vsync, init
        )
    }
}

// ---------------------------------------------------------------------------
// Engine access helpers (for use within Renderer methods)
// ---------------------------------------------------------------------------

impl PyMythRenderer {
    fn engine_ref(&self) -> PyResult<&Engine> {
        self.engine.as_deref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Renderer not initialized — call init_with_handle() first",
            )
        })
    }

    fn engine_mut(&mut self) -> PyResult<&mut Engine> {
        self.engine.as_deref_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Renderer not initialized — call init_with_handle() first",
            )
        })
    }

    /// Public accessor for sibling modules (e.g. `readback`).
    pub(crate) fn engine_ref_pub(&self) -> PyResult<&Engine> {
        self.engine_ref()
    }
}

impl Drop for PyMythRenderer {
    fn drop(&mut self) {
        if self.engine.is_some() {
            clear_engine_ptr();
        }
    }
}
