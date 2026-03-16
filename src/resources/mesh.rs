use uuid::Uuid;

use crate::assets::{GeometryHandle, MaterialHandle};
use crate::resources::buffer::{BufferReadGuard, CpuBuffer};
use crate::resources::uniforms::MorphUniforms;

pub const MAX_MORPH_TARGETS: usize = 128;
pub const MORPH_WEIGHT_THRESHOLD: f32 = 0.0000;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct Mesh {
    uuid: Uuid,
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

    /// Morph Target original influences (use setter methods to modify)
    morph_target_influences: Vec<f32>,

    /// Cached influences from the previous frame (used for change detection)
    prev_morph_target_influences: Vec<f32>,

    /// Morph Uniform Buffer (used by GPU, updated every frame)
    pub(crate) morph_uniforms: CpuBuffer<MorphUniforms>,

    // pub(crate) morph_dirty: bool,
    pub(crate) morph_update_frames: u8,
}

impl Mesh {
    /// Returns the unique identifier for this mesh.
    #[inline]
    #[must_use]
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

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
            prev_morph_target_influences: Vec::new(),
            morph_uniforms: CpuBuffer::new_uniform(None),
            // morph_dirty: false,
            morph_update_frames: 0,
        }
    }

    /// Returns the morph target influences as a slice
    #[inline]
    pub fn morph_target_influences(&self) -> &[f32] {
        &self.morph_target_influences
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
        self.prev_morph_target_influences = vec![0.0; target_count as usize];

        // initialize uniform buffer
        let mut uniforms = self.morph_uniforms.write();
        uniforms.vertex_count = vertex_count;
        uniforms.count = 0;
        uniforms.flags = 0;
    }

    /// Calls this at the end of each frame to latch current morph target influences into the "previous" cache.
    pub fn latch_previous_morph_weights(&mut self) {
        self.prev_morph_target_influences
            .clone_from(&self.morph_target_influences);
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
            // self.morph_dirty = true;
            // Once changed, we must update the GPU buffer for 2 consecutive frames!
            // Frame 1: Push the new current weights to GPU.
            // Frame 2: Push the latched prev_weights to GPU (to zero out velocity).
            self.morph_update_frames = 2;
        }
    }

    /// update Morph Uniforms (sort, cull, and fill GPU buffer)
    /// should be called before each frame rendering
    pub fn update_morph_uniforms(&mut self) {
        if self.morph_target_influences.is_empty() || self.morph_update_frames == 0 {
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
        {
            let mut uniforms = self.morph_uniforms.write();
            uniforms.count = active_targets.len() as u32;

            // Clear arrays (8 Vec4s, each containing 4 values)
            for i in 0..8 {
                uniforms.weights[i] = glam::Vec4::ZERO;
                uniforms.prev_weights[i] = glam::Vec4::ZERO;
                uniforms.indices[i] = glam::UVec4::ZERO;
            }

            // Fill active targets (map index i to Vec4[i/4][i%4])
            for (i, (target_idx, weight)) in active_targets.iter().enumerate() {
                let vec_idx = i / 4; // which Vec4
                let component = i % 4; // which component in Vec4
                uniforms.weights[vec_idx][component] = *weight;
                uniforms.indices[vec_idx][component] = *target_idx as u32;

                let prev_weight = self
                    .prev_morph_target_influences
                    .get(*target_idx)
                    .copied()
                    .unwrap_or(0.0);
                uniforms.prev_weights[vec_idx][component] = prev_weight;
            }
            // self.morph_dirty = false;

            self.morph_update_frames -= 1;
        }

        self.latch_previous_morph_weights();
    }
}
