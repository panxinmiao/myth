//! Material 相关操作

use crate::assets::{AssetServer, MaterialHandle};
use crate::resources::material::Material;
use crate::resources::material::MaterialData;

use crate::renderer::core::binding::Bindings;
use crate::renderer::core::builder::ResourceBuilder;

use super::{ResourceManager, GpuMaterial};

impl ResourceManager {
    pub(crate) fn prepare_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let Some(material) = assets.get_material(handle) else {
            log::warn!("Material {:?} not found in AssetServer.", handle);
            return;
        };

        // let uniform_ver = material.data.uniform_version();
        let binding_ver = material.data.binding_version();
        let layout_ver = material.data.layout_version();

        self.ensure_material_resources(material);

        if !self.gpu_materials.contains_key(handle) {
            self.build_full_material(assets, handle, material);
        }

        let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");

        if layout_ver != gpu_mat.last_layout_version {
            self.build_full_material(assets, handle, material);
            return;
        }

        if binding_ver != gpu_mat.last_binding_version {
            self.rebuild_material_bind_group(assets, handle, material);
            return;
        }

        let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
        gpu_mat.last_used_frame = self.frame_index;
    }

    pub(crate) fn build_full_material(&mut self, assets: &AssetServer, handle: MaterialHandle, material: &Material) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GpuMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            last_data_version: material.data.uniform_version(),
            last_binding_version: material.data.binding_version(),
            last_layout_version: material.data.layout_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_materials.insert(handle, gpu_mat);
        self.gpu_materials.get(handle).expect("Just inserted")
    }

    pub(crate) fn rebuild_material_bind_group(&mut self, assets: &AssetServer, handle: MaterialHandle, material: &Material) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        self.prepare_binding_resources(assets, &builder.resources);
        let layout = {
            let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");
            gpu_mat.layout.clone()
        };

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);

        {
            let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
            gpu_mat.bind_group = bind_group;
            gpu_mat.bind_group_id = bg_id;
            gpu_mat.last_binding_version = material.data.binding_version();
            gpu_mat.last_used_frame = self.frame_index;
        }

        self.gpu_materials.get(handle).expect("gpu material should exist.")
    }

    pub(crate) fn ensure_material_resources(&mut self, material: &Material) {

        // todo 优化：要ensure所有资源，不光是主 uniform buffer
        match &material.data {  
            MaterialData::Basic(m) => {
                self.ensure_buffer(&m.uniforms);
            },
            MaterialData::Phong(m) => {
                self.ensure_buffer(&m.uniforms);
            },
            MaterialData::Standard(m) => {
                self.ensure_buffer(&m.uniforms);
            },
        }
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }
}
