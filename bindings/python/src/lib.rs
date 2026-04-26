//! # Myth Engine — Python Bindings
//!
//! Provides a friendly, Three.js-style Python API for the Myth 3D rendering
//! engine.  All engine access is funnelled through thread-local pointers that
//! are valid only during `@app.init` / `@app.update` callbacks (or while a
//! [`PyMythRenderer`] is alive).  Every public helper returns
//! [`PyResult`] so that misuse surfaces as a clean Python exception rather
//! than a process-level crash.

use pyo3::prelude::*;
use std::cell::Cell;

mod animation;
mod app;
mod camera;
mod controls;
mod engine_proxy;
mod gaussian;
mod geometry;
mod input;
mod light;
mod material;
mod readback;
mod renderer;
mod scene;
mod texture;

// ============================================================================
// Thread-Local Engine Context
// ============================================================================
//
// During init/update callbacks we set a thread-local raw pointer to the
// Engine.  All proxy objects (PyScene, PyObject3D, …) access the engine
// through `with_engine`.
//
// Safety invariants:
//   - The engine lives on the main thread for the entire app lifetime.
//   - The pointer is set before callbacks and cleared after.
//   - All access happens on the same thread (winit event loop is
//     single-threaded).

thread_local! {
    static ENGINE_PTR: Cell<*mut myth_engine::Engine> = const { Cell::new(std::ptr::null_mut()) };
}

pub(crate) fn set_engine_ptr(engine: &mut myth_engine::Engine) {
    ENGINE_PTR.with(|cell| cell.set(engine as *mut _));
}

pub(crate) fn clear_engine_ptr() {
    ENGINE_PTR.with(|cell| cell.set(std::ptr::null_mut()));
}

// ---------------------------------------------------------------------------
// Window pointer
// ---------------------------------------------------------------------------
//
// We store a raw `*const dyn Window` in a `RefCell`.  The pointer is set
// before each callback and cleared after, so it is only dereferenced while
// the referent is live on the callstack.  No `transmute` is involved.

thread_local! {
    static WINDOW_PTR: std::cell::RefCell<Option<*const dyn myth_engine::app::Window>>
        = const { std::cell::RefCell::new(None) };
}

pub(crate) fn set_window_context(window: &dyn myth_engine::app::Window) {
    let ptr = window as *const (dyn myth_engine::app::Window + '_);
    // SAFETY: We erase the borrow lifetime from the fat pointer.  The pointer
    // is only stored for the duration of the callback (set before, cleared
    // after), so it never outlives the referent.
    let ptr: *const (dyn myth_engine::app::Window + 'static) = unsafe { std::mem::transmute(ptr) };
    WINDOW_PTR.with(|cell| {
        *cell.borrow_mut() = Some(ptr);
    });
}

pub(crate) fn clear_window_context() {
    WINDOW_PTR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Execute a closure with access to the Window trait object.
/// Returns `None` if called outside an init/update callback.
pub(crate) fn with_window<R>(f: impl FnOnce(&dyn myth_engine::app::Window) -> R) -> Option<R> {
    WINDOW_PTR.with(|cell| {
        let borrow = cell.borrow();
        (*borrow).map(|ptr| unsafe { f(&*ptr) })
    })
}

// ============================================================================
// Safe engine / scene accessors (never panic — return PyResult)
// ============================================================================

const ENGINE_UNAVAILABLE: &str = "Engine not available: this method can only be called inside @app.init / @app.update callbacks or while a Renderer is active";

/// Execute a closure with mutable access to the engine.
///
/// Returns `Err(PyRuntimeError)` when called outside a valid context instead
/// of panicking.
pub(crate) fn with_engine<R>(f: impl FnOnce(&mut myth_engine::Engine) -> R) -> PyResult<R> {
    ENGINE_PTR.with(|cell| {
        let ptr = cell.get();
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                ENGINE_UNAVAILABLE,
            ));
        }
        // SAFETY: pointer is non-null, set by `set_engine_ptr`, and only
        // accessed on the same thread during a bounded callback window.
        Ok(unsafe { f(&mut *ptr) })
    })
}

/// Execute a closure with mutable access to the active scene.
///
/// Returns `Err(PyRuntimeError)` when there is no engine context or no
/// active scene.
pub(crate) fn with_active_scene<R>(f: impl FnOnce(&mut myth_engine::Scene) -> R) -> PyResult<R> {
    with_engine(|engine| {
        let scene = engine
            .scene_manager
            .active_scene_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No active scene"))?;
        Ok(f(scene))
    })?
}

// ============================================================================
// Helper: extract geometry / material handles via duck-typing
// ============================================================================
//
// Instead of a rigid if-let downcast chain for every concrete type, we use
// PyO3's dynamic attribute access: any Python object that exposes a
// `_geo_handle` / `_mat_handle` property (returning an integer index) is
// accepted.  The built-in geometry/material classes provide this attribute
// automatically.

pub(crate) fn extract_geometry_handle(
    geo: &Bound<'_, PyAny>,
) -> PyResult<myth_engine::GeometryHandle> {
    let bits: u64 = geo
        .call_method0("_get_handle")
        .and_then(|v| v.extract::<u64>())
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "Expected a Geometry object (BoxGeometry, SphereGeometry, PlaneGeometry, Geometry)",
            )
        })?;
    Ok(myth_engine::GeometryHandle::from(
        slotmap::KeyData::from_ffi(bits),
    ))
}

pub(crate) fn extract_material_handle(
    mat: &Bound<'_, PyAny>,
) -> PyResult<myth_engine::MaterialHandle> {
    let bits: u64 = mat
        .call_method0("_get_handle")
        .and_then(|v| v.extract::<u64>())
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "Expected a Material object (UnlitMaterial, PhongMaterial, PhysicalMaterial)",
            )
        })?;
    Ok(myth_engine::MaterialHandle::from(
        slotmap::KeyData::from_ffi(bits),
    ))
}

// ============================================================================
// Python Module Registration
// ============================================================================

/// The `myth` Python module.
#[pymodule]
fn myth_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_class::<app::PyApp>()?;
    m.add_class::<renderer::PyMythRenderer>()?;
    m.add_class::<engine_proxy::PyEngine>()?;
    m.add_class::<scene::PyScene>()?;
    m.add_class::<scene::PyObject3D>()?;
    m.add_class::<scene::PyFrameState>()?;

    // Geometry
    m.add_class::<geometry::PyBoxGeometry>()?;
    m.add_class::<geometry::PySphereGeometry>()?;
    m.add_class::<geometry::PyPlaneGeometry>()?;
    m.add_class::<geometry::PyCustomGeometry>()?;

    // Material
    m.add_class::<material::PyUnlitMaterial>()?;
    m.add_class::<material::PyPhongMaterial>()?;
    m.add_class::<material::PyPhysicalMaterial>()?;

    // Camera & Light
    m.add_class::<camera::PyAntiAliasing>()?;
    m.add_class::<camera::PyPerspectiveCamera>()?;
    m.add_class::<camera::PyOrthographicCamera>()?;
    m.add_class::<light::PyDirectionalLight>()?;
    m.add_class::<light::PyPointLight>()?;
    m.add_class::<light::PySpotLight>()?;

    // Component Proxies
    m.add_class::<camera::PyPerspectiveCameraComponent>()?;
    m.add_class::<camera::PyOrthographicCameraComponent>()?;
    m.add_class::<light::PyDirectionalLightComponent>()?;
    m.add_class::<light::PyPointLightComponent>()?;
    m.add_class::<light::PySpotLightComponent>()?;
    m.add_class::<scene::PyMeshComponent>()?;

    // Texture
    m.add_class::<texture::PyTextureHandle>()?;

    // Controls
    m.add_class::<controls::PyOrbitControls>()?;

    // Input
    m.add_class::<input::PyInput>()?;

    // Animation
    m.add_class::<animation::PyAnimationMixer>()?;

    // Enums
    m.add_class::<renderer::PyRenderPath>()?;

    // Readback
    m.add_class::<readback::PyReadbackStream>()?;

    // Gaussian Splatting
    m.add_class::<gaussian::PyGaussianCloud>()?;

    Ok(())
}
