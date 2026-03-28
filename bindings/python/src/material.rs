//! Material wrappers �?UnlitMaterial, PhongMaterial, PhysicalMaterial.

use pyo3::prelude::*;
use slotmap::Key;

use myth_engine::math::{Vec2, Vec3, Vec4};

use myth_engine::MaterialHandle;

use crate::texture::PyTextureHandle;
use crate::with_engine;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_color(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    if let Ok(s) = obj.extract::<String>() {
        return hex_to_rgb(&s);
    }
    if let Ok(list) = obj.extract::<Vec<f32>>()
        && list.len() >= 3
    {
        return Ok([list[0], list[1], list[2]]);
    }
    if let Ok((r, g, b)) = obj.extract::<(f32, f32, f32)>() {
        return Ok([r, g, b]);
    }
    let _ = py;
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "Color must be '#RRGGBB', '#RGB', [r,g,b], or (r,g,b)",
    ))
}

fn hex_to_rgb(s: &str) -> PyResult<[f32; 3]> {
    let s = s.trim_start_matches('#');
    match s.len() {
        6 => {
            let r = u8::from_str_radix(&s[0..2], 16).unwrap_or(0);
            let g = u8::from_str_radix(&s[2..4], 16).unwrap_or(0);
            let b = u8::from_str_radix(&s[4..6], 16).unwrap_or(0);
            Ok([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
        }
        3 => {
            let r = u8::from_str_radix(&s[0..1], 16).unwrap_or(0);
            let g = u8::from_str_radix(&s[1..2], 16).unwrap_or(0);
            let b = u8::from_str_radix(&s[2..3], 16).unwrap_or(0);
            Ok([r as f32 / 15.0, g as f32 / 15.0, b as f32 / 15.0])
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid hex color",
        )),
    }
}

fn parse_side(name: &str) -> myth_engine::Side {
    match name.to_lowercase().as_str() {
        "front" => myth_engine::Side::Front,
        "back" => myth_engine::Side::Back,
        "double" | "both" => myth_engine::Side::Double,
        _ => myth_engine::Side::Front,
    }
}

fn parse_alpha_mode(name: &str) -> myth_engine::AlphaMode {
    match name.to_lowercase().as_str() {
        "blend" => myth_engine::AlphaMode::Blend,
        "mask" => myth_engine::AlphaMode::Mask,
        "blend_mask" | "blendmask" => myth_engine::AlphaMode::BlendMask,
        _ => myth_engine::AlphaMode::Opaque,
    }
}

// ---------------------------------------------------------------------------
// UnlitMaterial
// ---------------------------------------------------------------------------

#[pyclass(name = "UnlitMaterial")]
pub struct PyUnlitMaterial {
    handle: Option<MaterialHandle>,
    color: [f32; 3],
    opacity: f32,
    side: String,
    map: Option<myth_engine::TextureHandle>,
}

impl PyUnlitMaterial {
    pub(crate) fn get_or_create_handle(&mut self) -> PyResult<MaterialHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let color = self.color;
        let opacity = self.opacity;
        let side = self.side.clone();
        let map = self.map;

        let h = with_engine(|engine| {
            let mat =
                myth_engine::UnlitMaterial::new(Vec4::new(color[0], color[1], color[2], opacity));
            mat.set_side(parse_side(&side));
            if let Some(tex) = map {
                mat.set_map(Some(tex));
            }
            let material = myth_engine::Material::new(myth_engine::MaterialType::Unlit(mat));
            engine.assets.materials.add(material)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

#[pymethods]
impl PyUnlitMaterial {
    #[new]
    #[pyo3(signature = (color=None, opacity=1.0, side="front"))]
    fn new(
        py: Python<'_>,
        color: Option<&Bound<'_, PyAny>>,
        opacity: f32,
        side: &str,
    ) -> PyResult<Self> {
        let c = match color {
            Some(obj) => parse_color(py, obj)?,
            None => [1.0, 1.0, 1.0],
        };
        Ok(Self {
            handle: None,
            color: c,
            opacity,
            side: side.to_string(),
            map: None,
        })
    }

    #[getter]
    fn get_color(&self) -> [f32; 3] {
        self.color
    }
    #[setter]
    fn set_color(&mut self, py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<()> {
        self.color = parse_color(py, val)?;
        if let Some(h) = self.handle {
            let c = self.color;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(unlit) = mat.as_unlit()
                {
                    unlit.set_color(Vec4::new(c[0], c[1], c[2], self.opacity));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_opacity(&self) -> f32 {
        self.opacity
    }
    #[setter]
    fn set_opacity(&mut self, val: f32) -> PyResult<()> {
        self.opacity = val;
        if let Some(h) = self.handle {
            let c = self.color;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(unlit) = mat.as_unlit()
                {
                    unlit.set_color(Vec4::new(c[0], c[1], c[2], val));
                }
            })?;
        }
        Ok(())
    }

    /// Set the color (diffuse) texture map.
    fn set_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(unlit) = mat.as_unlit()
                {
                    unlit.set_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "UnlitMaterial(color=[{:.2}, {:.2}, {:.2}])",
            self.color[0], self.color[1], self.color[2]
        )
    }
}

// ---------------------------------------------------------------------------
// PhongMaterial
// ---------------------------------------------------------------------------

#[pyclass(name = "PhongMaterial")]
pub struct PyPhongMaterial {
    handle: Option<MaterialHandle>,
    color: [f32; 3],
    specular: [f32; 3],
    emissive: [f32; 3],
    emissive_intensity: f32,
    shininess: f32,
    opacity: f32,
    side: String,
    alpha_mode: String,
    depth_write: bool,
    map: Option<myth_engine::TextureHandle>,
    normal_map: Option<myth_engine::TextureHandle>,
    normal_scale: [f32; 2],
    specular_map: Option<myth_engine::TextureHandle>,
    emissive_map: Option<myth_engine::TextureHandle>,
}

impl PyPhongMaterial {
    pub(crate) fn get_or_create_handle(&mut self) -> PyResult<MaterialHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let color = self.color;
        let opacity = self.opacity;
        let specular = self.specular;
        let emissive = self.emissive;
        let emissive_intensity = self.emissive_intensity;
        let shininess = self.shininess;
        let side = self.side.clone();
        let map = self.map;
        let normal_map = self.normal_map;
        let normal_scale = self.normal_scale;
        let specular_map = self.specular_map;
        let emissive_map = self.emissive_map;

        let h = with_engine(|engine| {
            let mat =
                myth_engine::PhongMaterial::new(Vec4::new(color[0], color[1], color[2], opacity));
            mat.set_specular(Vec3::from(specular));
            mat.set_emissive(Vec3::from(emissive));
            mat.set_emissive_intensity(emissive_intensity);
            mat.set_shininess(shininess);
            mat.set_side(parse_side(&side));
            mat.set_alpha_mode(parse_alpha_mode(&self.alpha_mode));
            mat.set_depth_write(self.depth_write);
            if let Some(tex) = map {
                mat.set_map(Some(tex));
            }
            if let Some(tex) = normal_map {
                mat.set_normal_map(Some(tex));
                mat.set_normal_scale(Vec2::new(normal_scale[0], normal_scale[1]));
            }
            if let Some(tex) = specular_map {
                mat.set_specular_map(Some(tex));
            }
            if let Some(tex) = emissive_map {
                mat.set_emissive_map(Some(tex));
            }
            let material = myth_engine::Material::new(myth_engine::MaterialType::Phong(mat));
            engine.assets.materials.add(material)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

#[pymethods]
impl PyPhongMaterial {
    #[new]
    #[pyo3(signature = (
        color = None,
        specular = None,
        emissive = None,
        shininess = 30.0,
        emissive_intensity = 1.0,
        opacity = 1.0,
        side = "front",
        alpha_mode = "opaque",
        depth_write = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        color: Option<&Bound<'_, PyAny>>,
        specular: Option<&Bound<'_, PyAny>>,
        emissive: Option<&Bound<'_, PyAny>>,
        shininess: f32,
        emissive_intensity: f32,
        opacity: f32,
        side: &str,
        alpha_mode: &str,
        depth_write: bool,
    ) -> PyResult<Self> {
        let color = match color {
            Some(obj) => parse_color(py, obj)?,
            None => [1.0, 1.0, 1.0], // #ffffff
        };
        let specular = match specular {
            Some(obj) => parse_color(py, obj)?,
            None => [0.067, 0.067, 0.067], // ~#111111
        };
        let emissive = match emissive {
            Some(obj) => parse_color(py, obj)?,
            None => [0.0, 0.0, 0.0], // #000000
        };
        Ok(Self {
            handle: None,
            color,
            specular,
            emissive,
            emissive_intensity,
            shininess,
            opacity,
            side: side.to_string(),
            alpha_mode: alpha_mode.to_string(),
            depth_write,
            map: None,
            normal_map: None,
            normal_scale: [1.0, 1.0],
            specular_map: None,
            emissive_map: None,
        })
    }

    #[getter]
    fn get_color(&self) -> [f32; 3] {
        self.color
    }
    #[setter]
    fn set_color(&mut self, py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<()> {
        self.color = parse_color(py, val)?;
        if let Some(h) = self.handle {
            let c = self.color;
            let o = self.opacity;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_color(Vec4::new(c[0], c[1], c[2], o));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_shininess(&self) -> f32 {
        self.shininess
    }
    #[setter]
    fn set_shininess(&mut self, val: f32) -> PyResult<()> {
        self.shininess = val;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_shininess(val);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_opacity(&self) -> f32 {
        self.opacity
    }
    #[setter]
    fn set_opacity(&mut self, val: f32) -> PyResult<()> {
        self.opacity = val;
        if let Some(h) = self.handle {
            let c = self.color;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_color(Vec4::new(c[0], c[1], c[2], val));
                }
            })?;
        }
        Ok(())
    }

    fn set_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_normal_map(&mut self, tex: &PyTextureHandle, scale: Option<Vec<f32>>) -> PyResult<()> {
        let th = tex.inner();
        let ns = match scale {
            Some(v) if v.len() >= 2 => [v[0], v[1]],
            Some(v) if v.len() == 1 => [v[0], v[0]],
            _ => [1.0, 1.0],
        };
        self.normal_map = Some(th);
        self.normal_scale = ns;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_normal_map(Some(th));
                    phong.set_normal_scale(Vec2::new(ns[0], ns[1]));
                }
            })?;
        }
        Ok(())
    }

    fn set_specular_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.specular_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_specular_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_emissive_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.emissive_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_emissive_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    // ---- Property getters/setters ----

    #[getter]
    fn get_specular(&self) -> [f32; 3] {
        self.specular
    }
    #[setter]
    fn set_specular(&mut self, py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<()> {
        self.specular = parse_color(py, val)?;
        if let Some(h) = self.handle {
            let s = self.specular;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_specular(Vec3::from(s));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_emissive(&self) -> [f32; 3] {
        self.emissive
    }
    #[setter]
    fn set_emissive(&mut self, py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<()> {
        self.emissive = parse_color(py, val)?;
        if let Some(h) = self.handle {
            let e = self.emissive;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_emissive(Vec3::from(e));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_emissive_intensity(&self) -> f32 {
        self.emissive_intensity
    }
    #[setter]
    fn set_emissive_intensity(&mut self, v: f32) -> PyResult<()> {
        self.emissive_intensity = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_emissive_intensity(v);
                }
            })?;
        }
        Ok(())
    }

    // ---- Render settings ----

    #[getter]
    fn get_alpha_mode(&self) -> String {
        self.alpha_mode.clone()
    }
    #[setter]
    fn set_alpha_mode(&mut self, mode: &str) -> PyResult<()> {
        self.alpha_mode = mode.to_string();
        if let Some(h) = self.handle {
            let am = parse_alpha_mode(mode);
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_alpha_mode(am);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_depth_write(&self) -> bool {
        self.depth_write
    }
    #[setter]
    fn set_depth_write(&mut self, v: bool) -> PyResult<()> {
        self.depth_write = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phong) = mat.as_phong()
                {
                    phong.set_depth_write(v);
                }
            })?;
        }
        Ok(())
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "PhongMaterial(color=[{:.2}, {:.2}, {:.2}], shininess={:.1})",
            self.color[0], self.color[1], self.color[2], self.shininess
        )
    }
}

// ---------------------------------------------------------------------------
// PhysicalMaterial �?PBR metallic-roughness
// ---------------------------------------------------------------------------

#[pyclass(name = "PhysicalMaterial")]
pub struct PyPhysicalMaterial {
    handle: Option<MaterialHandle>,
    color: [f32; 3],
    metalness: f32,
    roughness: f32,
    emissive: [f32; 3],
    emissive_intensity: f32,
    opacity: f32,
    side: String,
    map: Option<myth_engine::TextureHandle>,
    normal_map: Option<myth_engine::TextureHandle>,
    normal_scale: f32,
    roughness_map: Option<myth_engine::TextureHandle>,
    metalness_map: Option<myth_engine::TextureHandle>,
    emissive_map: Option<myth_engine::TextureHandle>,
    ao_map: Option<myth_engine::TextureHandle>,
    // Render settings
    alpha_mode: String,
    depth_write: bool,
    // Advanced PBR
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen_color: [f32; 3],
    sheen_roughness: f32,
    transmission: f32,
    ior: f32,
}

impl PyPhysicalMaterial {
    pub(crate) fn get_or_create_handle(&mut self) -> PyResult<MaterialHandle> {
        if let Some(h) = self.handle {
            return Ok(h);
        }
        let c = self.color;
        let o = self.opacity;

        let h = with_engine(|engine| {
            let mat = myth_engine::PhysicalMaterial::new(Vec4::new(c[0], c[1], c[2], o))
                .with_roughness(self.roughness)
                .with_metalness(self.metalness)
                .with_emissive(Vec3::from(self.emissive), self.emissive_intensity)
                .with_normal_scale(Vec2::splat(self.normal_scale));

            mat.set_side(parse_side(&self.side));
            mat.set_alpha_mode(parse_alpha_mode(&self.alpha_mode));
            mat.set_depth_write(self.depth_write);

            if self.clearcoat > 0.0 {
                mat.set_clearcoat(self.clearcoat);
                mat.set_clearcoat_roughness(self.clearcoat_roughness);
            }
            if self.transmission > 0.0 {
                mat.set_transmission(self.transmission);
                mat.set_ior(self.ior);
            }
            if self.sheen_color != [0.0; 3] {
                mat.set_sheen_color(Vec3::from(self.sheen_color));
                mat.set_sheen_roughness(self.sheen_roughness);
            }

            // Textures
            if let Some(tex) = self.map {
                mat.set_map(Some(tex));
            }
            if let Some(tex) = self.normal_map {
                mat.set_normal_map(Some(tex));
            }
            if let Some(tex) = self.roughness_map {
                mat.set_roughness_map(Some(tex));
            }
            if let Some(tex) = self.metalness_map {
                mat.set_metalness_map(Some(tex));
            }
            if let Some(tex) = self.emissive_map {
                mat.set_emissive_map(Some(tex));
            }
            if let Some(tex) = self.ao_map {
                mat.set_ao_map(Some(tex));
            }

            let material = myth_engine::Material::new(myth_engine::MaterialType::Physical(mat));
            engine.assets.materials.add(material)
        })?;
        self.handle = Some(h);
        Ok(h)
    }
}

#[pymethods]
impl PyPhysicalMaterial {
    #[new]
    #[pyo3(signature = (
        color = None,
        metalness = 0.0,
        roughness = 0.5,
        emissive = None,
        emissive_intensity = 1.0,
        opacity = 1.0,
        side = "front",
        alpha_mode = "opaque",
        depth_write = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        color: Option<&Bound<'_, PyAny>>,
        metalness: f32,
        roughness: f32,
        emissive: Option<&Bound<'_, PyAny>>,
        emissive_intensity: f32,
        opacity: f32,
        side: &str,
        alpha_mode: &str,
        depth_write: bool,
    ) -> PyResult<Self> {
        let color = match color {
            Some(obj) => parse_color(py, obj)?,
            None => [1.0, 1.0, 1.0],
        };
        let emissive = match emissive {
            Some(obj) => parse_color(py, obj)?,
            None => [0.0, 0.0, 0.0],
        };
        Ok(Self {
            handle: None,
            color,
            metalness,
            roughness,
            emissive,
            emissive_intensity,
            opacity,
            side: side.to_string(),
            alpha_mode: alpha_mode.to_string(),
            depth_write,
            map: None,
            normal_map: None,
            normal_scale: 1.0,
            roughness_map: None,
            metalness_map: None,
            emissive_map: None,
            ao_map: None,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            sheen_color: [0.0; 3],
            sheen_roughness: 0.0,
            transmission: 0.0,
            ior: 1.5,
        })
    }

    #[getter]
    fn get_color(&self) -> [f32; 3] {
        self.color
    }
    #[setter]
    fn set_color(&mut self, py: Python<'_>, val: &Bound<'_, PyAny>) -> PyResult<()> {
        self.color = parse_color(py, val)?;
        if let Some(h) = self.handle {
            let c = self.color;
            let o = self.opacity;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_color(Vec4::new(c[0], c[1], c[2], o));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_metalness(&self) -> f32 {
        self.metalness
    }
    #[setter]
    fn set_metalness(&mut self, v: f32) -> PyResult<()> {
        self.metalness = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_metalness(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_roughness(&self) -> f32 {
        self.roughness
    }
    #[setter]
    fn set_roughness(&mut self, v: f32) -> PyResult<()> {
        self.roughness = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_roughness(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_emissive_intensity(&self) -> f32 {
        self.emissive_intensity
    }
    #[setter]
    fn set_emissive_intensity(&mut self, v: f32) -> PyResult<()> {
        self.emissive_intensity = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_emissive_intensity(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_opacity(&self) -> f32 {
        self.opacity
    }
    #[setter]
    fn set_opacity(&mut self, v: f32) -> PyResult<()> {
        self.opacity = v;
        if let Some(h) = self.handle {
            let c = self.color;
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_color(Vec4::new(c[0], c[1], c[2], v));
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_clearcoat(&self) -> f32 {
        self.clearcoat
    }
    #[setter]
    fn set_clearcoat(&mut self, v: f32) -> PyResult<()> {
        self.clearcoat = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_clearcoat(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_clearcoat_roughness(&self) -> f32 {
        self.clearcoat_roughness
    }
    #[setter]
    fn set_clearcoat_roughness(&mut self, v: f32) -> PyResult<()> {
        self.clearcoat_roughness = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_clearcoat_roughness(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_transmission(&self) -> f32 {
        self.transmission
    }
    #[setter]
    fn set_transmission(&mut self, v: f32) -> PyResult<()> {
        self.transmission = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_transmission(v);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_ior(&self) -> f32 {
        self.ior
    }
    #[setter]
    fn set_ior(&mut self, v: f32) -> PyResult<()> {
        self.ior = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_ior(v);
                }
            })?;
        }
        Ok(())
    }

    // ---- Render settings ----

    #[getter]
    fn get_alpha_mode(&self) -> String {
        self.alpha_mode.clone()
    }
    #[setter]
    fn set_alpha_mode(&mut self, mode: &str) -> PyResult<()> {
        self.alpha_mode = mode.to_string();
        if let Some(h) = self.handle {
            let am = parse_alpha_mode(mode);
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_alpha_mode(am);
                }
            })?;
        }
        Ok(())
    }

    #[getter]
    fn get_depth_write(&self) -> bool {
        self.depth_write
    }
    #[setter]
    fn set_depth_write(&mut self, v: bool) -> PyResult<()> {
        self.depth_write = v;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_depth_write(v);
                }
            })?;
        }
        Ok(())
    }

    // ---- Texture Setters ----

    fn set_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_normal_map(&mut self, tex: &PyTextureHandle, scale: Option<f32>) -> PyResult<()> {
        let th = tex.inner();
        let s = scale.unwrap_or(1.0);
        self.normal_map = Some(th);
        self.normal_scale = s;
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_normal_map(Some(th));
                    phys.set_normal_scale(Vec2::splat(s));
                }
            })?;
        }
        Ok(())
    }

    fn set_roughness_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.roughness_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_roughness_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_metalness_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.metalness_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_metalness_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_emissive_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.emissive_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_emissive_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn set_ao_map(&mut self, tex: &PyTextureHandle) -> PyResult<()> {
        let th = tex.inner();
        self.ao_map = Some(th);
        if let Some(h) = self.handle {
            with_engine(|engine| {
                if let Some(mat) = engine.assets.materials.get(h)
                    && let Some(phys) = mat.as_physical()
                {
                    phys.set_ao_map(Some(th));
                }
            })?;
        }
        Ok(())
    }

    fn _get_handle(&mut self) -> PyResult<u64> {
        let h = self.get_or_create_handle()?;
        Ok(h.data().as_ffi())
    }

    fn __repr__(&self) -> String {
        format!(
            "PhysicalMaterial(color=[{:.2}, {:.2}, {:.2}], metalness={:.2}, roughness={:.2})",
            self.color[0], self.color[1], self.color[2], self.metalness, self.roughness
        )
    }
}
