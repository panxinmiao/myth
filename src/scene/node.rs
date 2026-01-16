use uuid::Uuid;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};
use crate::scene::{NodeIndex, MeshKey, CameraKey, LightKey, SkeletonKey};
use crate::scene::transform::Transform;
use crate::scene::skeleton::{SkinBinding, BindMode};

#[derive(Debug, Clone)]
pub struct Node {
    pub id: Uuid,
    pub name: Cow<'static, str>,
    
    // === 场景图结构 ===
    pub parent: Option<NodeIndex>,
    pub children: Vec<NodeIndex>,

    // === 组件关联 ===
    pub mesh: Option<MeshKey>, 
    pub camera: Option<CameraKey>,
    pub light: Option<LightKey>, 

    pub skin: Option<SkinBinding>,

    // === Transform 组件 ===
    pub transform: Transform,
    
    pub visible: bool,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: Cow::Owned(name.to_string()),
            parent: None,
            children: Vec::new(),

            mesh: None,
            camera: None,
            light: None,

            skin: None,
            
            // 初始化 Transform
            transform: Transform::new(),
            
            visible: true,
        }
    }

    pub fn bind_skeleton(&mut self, skeleton: SkeletonKey, bind_mode: BindMode) {

        let bind_matrix_inv = self.world_matrix.inverse();

        self.skin = Some(SkinBinding {
            skeleton,
            bind_mode,
            bind_matrix_inv,
        });
    }
}



impl Deref for Node {
    type Target = Transform;

    fn deref(&self) -> &Self::Target {
        &self.transform
    }
}

impl DerefMut for Node {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.transform
    }
}