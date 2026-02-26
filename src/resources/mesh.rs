use uuid::Uuid;

use crate::assets::{GeometryHandle, MaterialHandle};
use crate::resources::buffer::{BufferReadGuard, CpuBuffer};
use crate::resources::uniforms::MorphUniforms;

pub const MAX_MORPH_TARGETS: usize = 128;
pub const MORPH_WEIGHT_THRESHOLD: f32 = 0.0000;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct Mesh {
    pub uuid: Uuid,
    pub name: String,

    // === Rescources ===
    pub geometry: GeometryHandle,
    pub material: MaterialHandle,

    // === Instance-specific rendering settings ===
    pub visible: bool,

    pub cast_shadows: bool,
    pub receive_shadows: bool,

    // Render Order
    pub render_order: i32,

    /// Morph Target original influences
    pub morph_target_influences: Vec<f32>,

    /// Morph Uniform Buffer (used by GPU, updated every frame)
    pub(crate) morph_uniforms: CpuBuffer<MorphUniforms>,

    pub(crate) morph_dirty: bool,
}

impl Mesh {
    #[must_use]
    pub fn new(geometry: GeometryHandle, material: MaterialHandle) -> Self {
        let uuid = Uuid::new_v4();
        Self {
            uuid,
            name: format!("Mesh_{uuid}"),
            geometry,
            material,
            visible: true,
            cast_shadows: true,
            receive_shadows: true,
            render_order: 0,
            morph_target_influences: Vec::new(),
            morph_uniforms: CpuBuffer::new_uniform(None),
            morph_dirty: false,
        }
    }

    pub fn morph_uniforms(&self) -> BufferReadGuard<'_, MorphUniforms> {
        self.morph_uniforms.read()
    }

    pub fn morph_uniforms_mut(
        &mut self,
    ) -> crate::resources::buffer::BufferGuard<'_, MorphUniforms> {
        self.morph_uniforms.write()
    }

    /// initialize morph target influences
    /// should be called after loading geometry
    pub fn init_morph_targets(&mut self, target_count: u32, vertex_count: u32) {
        self.morph_target_influences = vec![0.0; target_count as usize];

        // initialize uniform buffer
        let mut uniforms = self.morph_uniforms.write();
        uniforms.vertex_count = vertex_count;
        uniforms.count = 0;
        uniforms.flags = 0;
    }

    /// set influence for a single morph target
    pub fn set_morph_target_influence(&mut self, index: usize, weight: f32) {
        if index < self.morph_target_influences.len() {
            self.morph_target_influences[index] = weight;
        }
    }

    /// batch set morph target influences
    pub fn set_morph_target_influences(&mut self, weights: &[f32]) {
        let len = weights.len().min(self.morph_target_influences.len());
        if self.morph_target_influences[..len] != weights[..len] {
            self.morph_target_influences[..len].copy_from_slice(&weights[..len]);
            self.morph_dirty = true;
        }
    }

    /// update Morph Uniforms (sort, cull, and fill GPU buffer)
    /// should be called before each frame rendering
    pub fn update_morph_uniforms(&mut self) {
        if self.morph_target_influences.is_empty() || !self.morph_dirty {
            return;
        }

        // 1. Collect active influences (filter out very small values)
        let mut active_targets: Vec<(usize, f32)> = self
            .morph_target_influences
            .iter()
            .enumerate()
            .filter(|(_, w)| w.abs() > MORPH_WEIGHT_THRESHOLD)
            .map(|(i, w)| (i, *w))
            .collect();

        // 2. Sort by weight descending
        active_targets.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Truncate to maximum capacity
        active_targets.truncate(MAX_MORPH_TARGETS);

        // 4. Fill Uniform Buffer (packed into Vec4/UVec4)
        let mut uniforms = self.morph_uniforms.write();
        uniforms.count = active_targets.len() as u32;

        // Clear arrays (8 Vec4s, each containing 4 values)
        for i in 0..8 {
            uniforms.weights[i] = glam::Vec4::ZERO;
            uniforms.indices[i] = glam::UVec4::ZERO;
        }

        // Fill active targets (map index i to Vec4[i/4][i%4])
        for (i, (target_idx, weight)) in active_targets.iter().enumerate() {
            let vec_idx = i / 4; // which Vec4
            let component = i % 4; // which component in Vec4
            uniforms.weights[vec_idx][component] = *weight;
            uniforms.indices[vec_idx][component] = *target_idx as u32;
        }
        self.morph_dirty = false;
    }
}
