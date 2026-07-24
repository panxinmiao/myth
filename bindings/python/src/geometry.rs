//! Geometry types: BoxGeometry, SphereGeometry, PlaneGeometry,
//! CylinderGeometry, ConeGeometry, TorusGeometry, Geometry (custom).

use pyo3::prelude::*;
use slotmap::Key;

use crate::with_engine;

/// A box (cuboid) geometry.
///
/// Args:
///     width: Width along X axis (default: 1.0)
///     height: Height along Y axis (default: 1.0)
///     depth: Depth along Z axis (default: 1.0)
#[pyclass(name = "BoxGeometry")]
pub struct PyBoxGeometry {
    #[pyo3(get)]
    width: f32,
    #[pyo3(get)]
    height: f32,
    #[pyo3(get)]
    depth: f32,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyBoxGeometry {
    #[new]
    #[pyo3(signature = (width=1.0, height=1.0, depth=1.0))]
    fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width,
            height,
            depth,
            handle: None,
        }
    }

    /// Internal: returns the raw slotmap key as u64 for duck-typed handle extraction.
    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "BoxGeometry(width={}, height={}, depth={})",
            self.width, self.height, self.depth
        )
    }
}

impl PyBoxGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::Geometry::new_box(self.width, self.height, self.depth);
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A sphere geometry.
///
/// Args:
///     radius: Sphere radius (default: 1.0)
///     width_segments: Horizontal segments (default: 32)
///     height_segments: Vertical segments (default: 16)
#[pyclass(name = "SphereGeometry")]
pub struct PySphereGeometry {
    #[pyo3(get)]
    radius: f32,
    #[pyo3(get)]
    width_segments: u32,
    #[pyo3(get)]
    height_segments: u32,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PySphereGeometry {
    #[new]
    #[pyo3(signature = (radius=1.0, width_segments=32, height_segments=16))]
    fn new(radius: f32, width_segments: u32, height_segments: u32) -> Self {
        Self {
            radius,
            width_segments,
            height_segments,
            handle: None,
        }
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "SphereGeometry(radius={}, segments={}x{})",
            self.radius, self.width_segments, self.height_segments
        )
    }
}

impl PySphereGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::Geometry::new_sphere(self.radius);
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A plane geometry.
///
/// Args:
///     width: Width along X axis (default: 1.0)
///     height: Height along Z axis (default: 1.0)
#[pyclass(name = "PlaneGeometry")]
pub struct PyPlaneGeometry {
    #[pyo3(get)]
    width: f32,
    #[pyo3(get)]
    height: f32,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyPlaneGeometry {
    #[new]
    #[pyo3(signature = (width=1.0, height=1.0))]
    fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            handle: None,
        }
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "PlaneGeometry(width={}, height={})",
            self.width, self.height
        )
    }
}

impl PyPlaneGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::Geometry::new_plane(self.width, self.height);
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A cylinder geometry.
///
/// Args:
///     radius: Radius for the top and bottom caps (default: 1.0)
///     height: Height along Y axis (default: 1.0)
///     radial_segments: Number of radial segments (default: 32)
///     height_segments: Number of vertical segments (default: 1)
///     open_ended: Whether to omit the caps (default: False)
#[pyclass(name = "CylinderGeometry")]
pub struct PyCylinderGeometry {
    #[pyo3(get)]
    radius: f32,
    #[pyo3(get)]
    height: f32,
    #[pyo3(get)]
    radial_segments: u32,
    #[pyo3(get)]
    height_segments: u32,
    #[pyo3(get)]
    open_ended: bool,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyCylinderGeometry {
    #[new]
    #[pyo3(signature = (radius=1.0, height=1.0, radial_segments=32, height_segments=1, open_ended=false))]
    fn new(
        radius: f32,
        height: f32,
        radial_segments: u32,
        height_segments: u32,
        open_ended: bool,
    ) -> Self {
        Self {
            radius,
            height,
            radial_segments,
            height_segments,
            open_ended,
            handle: None,
        }
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "CylinderGeometry(radius={}, height={}, radial_segments={}, height_segments={}, open_ended={})",
            self.radius, self.height, self.radial_segments, self.height_segments, self.open_ended
        )
    }
}

impl PyCylinderGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::create_cylinder(&myth_engine::CylinderOptions {
                radius_top: self.radius,
                radius_bottom: self.radius,
                height: self.height,
                radial_segments: self.radial_segments,
                height_segments: self.height_segments,
                open_ended: self.open_ended,
            });
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A cone geometry.
///
/// Args:
///     radius: Base radius (default: 1.0)
///     height: Height along Y axis (default: 1.0)
///     radial_segments: Number of radial segments (default: 32)
///     height_segments: Number of vertical segments (default: 1)
///     open_ended: Whether to omit the bottom cap (default: False)
#[pyclass(name = "ConeGeometry")]
pub struct PyConeGeometry {
    #[pyo3(get)]
    radius: f32,
    #[pyo3(get)]
    height: f32,
    #[pyo3(get)]
    radial_segments: u32,
    #[pyo3(get)]
    height_segments: u32,
    #[pyo3(get)]
    open_ended: bool,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyConeGeometry {
    #[new]
    #[pyo3(signature = (radius=1.0, height=1.0, radial_segments=32, height_segments=1, open_ended=false))]
    fn new(
        radius: f32,
        height: f32,
        radial_segments: u32,
        height_segments: u32,
        open_ended: bool,
    ) -> Self {
        Self {
            radius,
            height,
            radial_segments,
            height_segments,
            open_ended,
            handle: None,
        }
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "ConeGeometry(radius={}, height={}, radial_segments={}, height_segments={}, open_ended={})",
            self.radius, self.height, self.radial_segments, self.height_segments, self.open_ended
        )
    }
}

impl PyConeGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::create_cone(&myth_engine::ConeOptions {
                radius: self.radius,
                height: self.height,
                radial_segments: self.radial_segments,
                height_segments: self.height_segments,
                open_ended: self.open_ended,
            });
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A torus geometry.
///
/// Args:
///     radius: Major radius from torus center to tube center (default: 1.0)
///     tube: Tube radius (default: 0.4)
///     radial_segments: Segments around the tube cross-section (default: 16)
///     tubular_segments: Segments around the main ring (default: 32)
#[pyclass(name = "TorusGeometry")]
pub struct PyTorusGeometry {
    #[pyo3(get)]
    radius: f32,
    #[pyo3(get)]
    tube: f32,
    #[pyo3(get)]
    radial_segments: u32,
    #[pyo3(get)]
    tubular_segments: u32,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyTorusGeometry {
    #[new]
    #[pyo3(signature = (radius=1.0, tube=0.4, radial_segments=16, tubular_segments=32))]
    fn new(radius: f32, tube: f32, radial_segments: u32, tubular_segments: u32) -> Self {
        Self {
            radius,
            tube,
            radial_segments,
            tubular_segments,
            handle: None,
        }
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "TorusGeometry(radius={}, tube={}, radial_segments={}, tubular_segments={})",
            self.radius, self.tube, self.radial_segments, self.tubular_segments
        )
    }
}

impl PyTorusGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let geo = myth_engine::create_torus(&myth_engine::TorusOptions {
                radius: self.radius,
                tube: self.tube,
                radial_segments: self.radial_segments,
                tubular_segments: self.tubular_segments,
            });
            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

// ============================================================================

/// A custom geometry that you can build by setting attributes manually.
///
/// Example:
/// ```python
/// geo = myth.Geometry()
/// geo.set_positions([0,0,0, 1,0,0, 0,1,0])
/// geo.set_indices([0, 1, 2])
/// ```
#[pyclass(name = "Geometry")]
pub struct PyCustomGeometry {
    positions: Option<Vec<f32>>,
    normals: Option<Vec<f32>>,
    uvs: Option<Vec<f32>>,
    colors: Option<Vec<f32>>,
    indices: Option<Vec<u32>>,
    handle: Option<myth_engine::GeometryHandle>,
}

#[pymethods]
impl PyCustomGeometry {
    #[new]
    fn new() -> Self {
        Self {
            positions: None,
            normals: None,
            uvs: None,
            colors: None,
            indices: None,
            handle: None,
        }
    }

    /// Set vertex positions as a flat list [x0,y0,z0, x1,y1,z1, ...].
    fn set_positions(&mut self, data: Vec<f32>) {
        self.positions = Some(data);
        self.handle = None;
    }

    /// Set vertex normals as a flat list [nx0,ny0,nz0, ...].
    fn set_normals(&mut self, data: Vec<f32>) {
        self.normals = Some(data);
        self.handle = None;
    }

    /// Set UV coordinates as a flat list [u0,v0, u1,v1, ...].
    fn set_uvs(&mut self, data: Vec<f32>) {
        self.uvs = Some(data);
        self.handle = None;
    }

    /// Set vertex colors as a flat list [r0,g0,b0, r1,g1,b1, ...].
    fn set_colors(&mut self, data: Vec<f32>) {
        self.colors = Some(data);
        self.handle = None;
    }

    /// Set triangle indices.
    fn set_indices(&mut self, data: Vec<u32>) {
        self.indices = Some(data);
        self.handle = None;
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }
}

impl PyCustomGeometry {
    pub fn get_or_create_handle(&mut self) -> PyResult<myth_engine::GeometryHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let h = with_engine(|engine| {
            let mut geo = myth_engine::Geometry::new();

            if let Some(ref positions) = self.positions {
                geo.set_attribute(
                    "position",
                    myth_engine::Attribute::new_planar::<[f32; 3]>(
                        bytemuck::cast_slice(positions),
                        myth_engine::VertexFormat::Float32x3,
                    ),
                );
            }

            if let Some(ref normals) = self.normals {
                geo.set_attribute(
                    "normal",
                    myth_engine::Attribute::new_planar::<[f32; 3]>(
                        bytemuck::cast_slice(normals),
                        myth_engine::VertexFormat::Float32x3,
                    ),
                );
            }

            if let Some(ref uvs) = self.uvs {
                geo.set_attribute(
                    "uv",
                    myth_engine::Attribute::new_planar::<[f32; 2]>(
                        bytemuck::cast_slice(uvs),
                        myth_engine::VertexFormat::Float32x2,
                    ),
                );
            }

            if let Some(ref colors) = self.colors {
                geo.set_attribute(
                    "color",
                    myth_engine::Attribute::new_planar::<[f32; 4]>(
                        bytemuck::cast_slice(colors),
                        myth_engine::VertexFormat::Float32x4,
                    ),
                );
            }

            if let Some(ref indices) = self.indices {
                geo.set_indices_u32(indices);
            }

            geo.compute_bounding_volume();

            engine.assets.geometries.add(geo)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}
