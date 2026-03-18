//! PyApp — Application entry point and Python ↔ Engine bridge.

use std::cell::RefCell;
use std::ffi::c_int;

use pyo3::prelude::*;

use myth_engine::Engine;
use myth_engine::app::{AppHandler, Window};
use myth_engine::engine::FrameState;

use crate::engine_proxy::PyEngine;
use crate::scene::PyFrameState;
use crate::{clear_engine_ptr, set_engine_ptr};

// ---------------------------------------------------------------------------
// Raw FFI for GIL management (PyO3 0.28 removed with_gil / allow_threads)
// ---------------------------------------------------------------------------
unsafe extern "C" {
    fn PyEval_SaveThread() -> *mut std::ffi::c_void;
    fn PyEval_RestoreThread(state: *mut std::ffi::c_void);
    fn PyGILState_Ensure() -> c_int;
    fn PyGILState_Release(state: c_int);
}

// ---------------------------------------------------------------------------
// Pending callbacks (collected during compose_frame, called with GIL)
// ---------------------------------------------------------------------------

type BoxCb = Box<dyn FnOnce() + Send>;

thread_local! {
    static PENDING_CALLBACKS: RefCell<Vec<BoxCb>> = RefCell::new(Vec::new());
}

#[allow(dead_code)]
pub fn push_callback(cb: BoxCb) {
    PENDING_CALLBACKS.with(|v| v.borrow_mut().push(cb));
}

fn drain_callbacks() {
    PENDING_CALLBACKS.with(|v| {
        let cbs: Vec<BoxCb> = v.borrow_mut().drain(..).collect();
        for cb in cbs {
            cb();
        }
    });
}

// ---------------------------------------------------------------------------
// Stored Python callbacks (set before run, used by PythonHandler)
// ---------------------------------------------------------------------------

thread_local! {
    static INIT_FN: RefCell<Option<Py<PyAny>>> = RefCell::new(None);
    static UPDATE_FN: RefCell<Option<Py<PyAny>>> = RefCell::new(None);
}

// ---------------------------------------------------------------------------
// PythonHandler — implements myth_engine::AppHandler
// ---------------------------------------------------------------------------

struct PythonHandler;

impl AppHandler for PythonHandler {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        INIT_FN.with(|cell| {
            let borrow = cell.borrow();
            if let Some(ref init_fn) = *borrow {
                set_engine_ptr(engine);
                crate::set_window_context(window);

                let gstate = unsafe { PyGILState_Ensure() };

                let py = unsafe { Python::assume_attached() };
                let ctx = Py::new(py, PyEngine::new()).unwrap();
                let result = init_fn.call1(py, (ctx,));

                if let Err(e) = result {
                    e.print(py);
                }
                drain_callbacks();

                unsafe {
                    PyGILState_Release(gstate);
                }
                crate::clear_window_context();
                clear_engine_ptr();
            }
        });

        PythonHandler
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        UPDATE_FN.with(|cell| {
            let borrow = cell.borrow();
            if let Some(ref update_fn) = *borrow {
                set_engine_ptr(engine);
                crate::set_window_context(window);

                let gstate = unsafe { PyGILState_Ensure() };

                let py = unsafe { Python::assume_attached() };
                let ctx = Py::new(py, PyEngine::new()).unwrap();
                let py_frame = Py::new(
                    py,
                    PyFrameState::new(frame.time, frame.dt, frame.frame_count),
                )
                .unwrap();
                let result = update_fn.call1(py, (ctx, py_frame));

                if let Err(e) = result {
                    e.print(py);
                }
                drain_callbacks();

                unsafe {
                    PyGILState_Release(gstate);
                }
                crate::clear_window_context();
                clear_engine_ptr();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// build_settings (delegated to renderer module)
// ---------------------------------------------------------------------------

use crate::renderer::build_settings;

// ---------------------------------------------------------------------------
// PyApp — the Python-facing application class
// ---------------------------------------------------------------------------

/// The main myth application.
///
/// Register `@app.init` and `@app.update` callbacks, then call `app.run()`.
///
/// Example:
/// ```python
/// import myth
///
/// app = myth.App(title="My App", render_path="hdr")
///
/// @app.init
/// def init(ctx):
///     scene = ctx.create_scene()
///     ...
///
/// @app.update
/// def update(ctx, frame):
///     ...
///
/// app.run()
/// ```
#[pyclass(name = "App")]
pub struct PyApp {
    title: String,
    render_path: String,
    vsync: bool,
    clear_color: [f32; 4],
    init_fn: Option<Py<PyAny>>,
    update_fn: Option<Py<PyAny>>,
}

#[pymethods]
impl PyApp {
    #[new]
    #[pyo3(signature = (
        title = "Myth Engine",
        render_path = None,
        vsync = true,
        clear_color = [0.1, 0.1, 0.1, 1.0],
    ))]
    fn new(
        title: &str,
        render_path: Option<&Bound<'_, PyAny>>,
        vsync: bool,
        clear_color: [f32; 4],
    ) -> PyResult<Self> {
        let rp = match render_path {
            Some(obj) => crate::renderer::parse_render_path(obj)?,
            None => "basic".to_string(),
        };
        Ok(Self {
            title: title.to_string(),
            render_path: rp,
            vsync,
            clear_color,
            init_fn: None,
            update_fn: None,
        })
    }

    /// Register an init callback. Used as a decorator: `@app.init`.
    fn init(&mut self, py: Python<'_>, func: Py<PyAny>) -> Py<PyAny> {
        self.init_fn = Some(func.clone_ref(py));
        func
    }

    /// Register a per-frame update callback. Used as a decorator: `@app.update`.
    fn update(&mut self, py: Python<'_>, func: Py<PyAny>) -> Py<PyAny> {
        self.update_fn = Some(func.clone_ref(py));
        func
    }

    /// Run the application (blocking).
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        let settings = build_settings(&self.render_path, self.vsync, self.clear_color);

        // Store callbacks in thread-locals so PythonHandler can access them.
        if let Some(ref f) = self.init_fn {
            INIT_FN.with(|cell| *cell.borrow_mut() = Some(f.clone_ref(py)));
        }
        if let Some(ref f) = self.update_fn {
            UPDATE_FN.with(|cell| *cell.borrow_mut() = Some(f.clone_ref(py)));
        }

        let title = self.title.clone();

        // Release the GIL before entering the blocking event loop.
        let saved = unsafe { PyEval_SaveThread() };

        let result = myth_engine::App::new()
            .with_title(title)
            .with_settings(settings)
            .run::<PythonHandler>();

        // Restore the GIL.
        unsafe {
            PyEval_RestoreThread(saved);
        }

        // Clean up thread-locals.
        INIT_FN.with(|cell| *cell.borrow_mut() = None);
        UPDATE_FN.with(|cell| *cell.borrow_mut() = None);

        result.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Engine error: {e}"))
        })
    }

    #[getter]
    fn get_title(&self) -> &str {
        &self.title
    }
    #[setter]
    fn set_title(&mut self, val: String) {
        self.title = val;
    }

    #[getter]
    fn get_render_path(&self) -> &str {
        &self.render_path
    }
    #[setter]
    fn set_render_path(&mut self, val: String) {
        self.render_path = val;
    }

    #[getter]
    fn get_vsync(&self) -> bool {
        self.vsync
    }
    #[setter]
    fn set_vsync(&mut self, val: bool) {
        self.vsync = val;
    }

    #[getter]
    fn get_clear_color(&self) -> [f32; 4] {
        self.clear_color
    }
    #[setter]
    fn set_clear_color(&mut self, val: [f32; 4]) {
        self.clear_color = val;
    }

    fn __repr__(&self) -> String {
        format!(
            "App(title='{}', render_path='{}', vsync={})",
            self.title, self.render_path, self.vsync
        )
    }
}
