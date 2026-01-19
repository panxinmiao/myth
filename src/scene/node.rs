use std::borrow::Cow;
use crate::scene::{NodeIndex, MeshKey, CameraKey, LightKey, SkeletonKey};
use crate::scene::transform::Transform;
use crate::scene::skeleton::{SkinBinding, BindMode};
use crate::animation::values::MorphWeightData;
use crate::resources::mesh::MAX_MORPH_TARGETS;

#[derive(Debug, Clone)]
pub struct Node {
    pub name: Cow<'static, str>,
    
    pub(crate) parent: Option<NodeIndex>,
    pub(crate) children: Vec<NodeIndex>,

    pub mesh: Option<MeshKey>, 
    pub camera: Option<CameraKey>,
    pub light: Option<LightKey>, 

    pub skin: Option<SkinBinding>,

    pub transform: Transform,
    
    pub visible: bool,
    
    pub morph_weights: Vec<f32>,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Self {
            name: Cow::Owned(name.to_string()),
            parent: None,
            children: Vec::new(),

            mesh: None,
            camera: None,
            light: None,

            skin: None,
            
            transform: Transform::new(),
            
            visible: true,
            
            morph_weights: Vec::new(),
        }
    }

    pub fn set_morph_weights_from_pod(&mut self, data: &MorphWeightData, count: usize) {
        if self.morph_weights.len() < count {
            self.morph_weights.resize(count, 0.0);
        }
        let valid_count = count.min(MAX_MORPH_TARGETS);
        self.morph_weights[..valid_count].copy_from_slice(&data.weights[..valid_count]);
    }

    pub fn bind_skeleton(&mut self, skeleton: SkeletonKey, bind_mode: BindMode) {

        let bind_matrix_inv = self.transform.world_matrix.inverse();

        self.skin = Some(SkinBinding {
            skeleton,
            bind_mode,
            bind_matrix_inv,
        });
    }
}