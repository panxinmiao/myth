use glam::{Affine3A, Mat4, Vec3};
use slotmap::SlotMap;
use uuid::Uuid;

use crate::resources::BoundingBox;
use crate::{
    resources::buffer::CpuBuffer,
    scene::{Node, NodeHandle, SkeletonKey},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BindMode {
    /// Bones follow node movement (most common, e.g., character skinning)
    /// In this case, `inverse_bind_matrix` = `node.world_matrix.inverse()`
    Attached,
    /// Bones are detached from nodes (special use cases)
    /// In this case, uses the static `inverse_matrix` recorded at bind time
    Detached,
}

#[derive(Debug, Clone)]
pub struct SkinBinding {
    pub skeleton: SkeletonKey,
    pub bind_mode: BindMode,
    /// Inverse matrix snapshot at bind time (used for Detached mode)
    pub bind_matrix_inv: Affine3A,
}

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub id: Uuid,
    pub name: String,

    // === Core Data ===
    // Bone list: ordered array, corresponds to joint index in shader
    // bones[i] corresponds to joints[i] in shader
    pub bones: Vec<NodeHandle>,

    // Inverse Bind Matrices
    // This is static data, typically unchanged after loading from glTF
    // Purpose: transforms vertices from Mesh space to bone local space
    pub(crate) inverse_bind_matrices: Vec<Affine3A>,

    // === Bounding Box Data (for frustum culling) ===
    /// Local space bounding box (relative to root bone)
    pub(crate) local_bounds: Option<BoundingBox>,
    /// Root bone index (usually bones[0])
    pub(crate) root_bone_index: usize,

    // === Runtime Data ===
    // Final computed matrix array, updated every frame
    // Data flow: data here -> copy to GPU Uniform Buffer
    pub(crate) joint_matrices: CpuBuffer<Vec<Mat4>>,
}

impl Skeleton {
    #[must_use]
    pub fn new(
        name: &str,
        bones: Vec<NodeHandle>,
        inverse_bind_matrices: Vec<Affine3A>,
        root_bone_index: usize,
    ) -> Self {
        let count = bones.len();

        let joint_matrices = CpuBuffer::new(
            vec![Mat4::IDENTITY; count],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some(&format!("SkeletonJointMatrices_{name}")),
        );

        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            bones,
            inverse_bind_matrices,
            local_bounds: None,
            root_bone_index,
            joint_matrices,
        }
    }

    /// Gets the local space bounding box (lazy computed)
    #[inline]
    pub fn local_bounds(&self) -> Option<&BoundingBox> {
        self.local_bounds.as_ref()
    }

    /// Gets the root bone node handle
    #[inline]
    pub fn root_bone(&self) -> Option<NodeHandle> {
        self.bones.get(self.root_bone_index).copied()
    }

    /// Computes the local space bounding box for bones
    /// Based on bone bind pose positions, calculates bounding box relative to root bone
    pub fn compute_local_bounds(&mut self, nodes: &SlotMap<NodeHandle, Node>) {
        if self.bones.is_empty() {
            return;
        }

        // Get the inverse of the root bone's world matrix
        let root_bone_handle = self.bones[self.root_bone_index];
        let Some(root_node) = nodes.get(root_bone_handle) else {
            return;
        };
        let root_world_inv = root_node.transform.world_matrix.inverse();

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut valid_count = 0;

        for &bone_handle in &self.bones {
            if let Some(bone_node) = nodes.get(bone_handle) {
                let world_pos: Vec3 = bone_node.transform.world_matrix.translation.into();
                let local_pos = root_world_inv.transform_point3(world_pos);

                min = min.min(local_pos);
                max = max.max(local_pos);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            let size = max - min;
            let padding = size * 0.15;

            self.local_bounds = Some(BoundingBox {
                min: min - padding,
                max: max + padding,
            });
        }
    }

    /// Computes the precise world bounding box for the current pose (no padding, iterates all bones in real-time)
    /// Used for camera framing (Frame Object)
    pub fn compute_tight_world_bounds(
        &self,
        nodes: &SlotMap<NodeHandle, Node>,
    ) -> Option<BoundingBox> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut valid = false;

        for &bone_handle in &self.bones {
            if let Some(bone_node) = nodes.get(bone_handle) {
                let pos = bone_node.transform.world_matrix.translation.into();
                min = min.min(pos);
                max = max.max(pos);
                valid = true;
            }
        }

        if valid {
            Some(BoundingBox { min, max })
        } else {
            None
        }
    }

    /// Gets the world space bounding box
    #[inline]
    pub fn world_bounds(&self, nodes: &SlotMap<NodeHandle, Node>) -> Option<BoundingBox> {
        let local_bounds = self.local_bounds.as_ref()?;
        let root_bone_handle = self.bones.get(self.root_bone_index)?;
        let root_node = nodes.get(*root_bone_handle)?;
        Some(local_bounds.transform(&root_node.transform.world_matrix))
    }

    /// Updates bone matrices
    ///
    /// # Arguments
    /// * `nodes`: Global node storage, from which we read each bone's `world_matrix`
    /// * `root_matrix_inv`: Inverse of the world matrix of the node containing the `SkinnedMesh`
    ///   (used to transform bone transforms back to Mesh local space)
    pub fn compute_joint_matrices(
        &mut self,
        nodes: &SlotMap<NodeHandle, Node>,
        root_matrix_inv: Affine3A,
    ) {
        for (i, &bone_handle) in self.bones.iter().enumerate() {
            // 1. Get current bone's world transform for this frame (computed by scene graph system)
            let Some(bone_node) = nodes.get(bone_handle) else {
                continue;
            };
            let bone_world_matrix = bone_node.transform.world_matrix;

            // 2. Get the corresponding inverse bind matrix
            let ibm = self.inverse_bind_matrices[i];

            // 3. Compute the final Joint Matrix
            // Order matters: first apply IBM (transform to bone local), then apply current bone world transform, finally (optionally) cancel Mesh's own transform
            self.joint_matrices.write()[i] = (root_matrix_inv * bone_world_matrix * ibm).into();
        }
    }
}
