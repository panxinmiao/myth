//! Geometry 相关操作

use std::sync::Arc;

use crate::assets::{AssetServer, GeometryHandle};
use crate::resources::geometry::Geometry;

use super::{ResourceManager, GpuGeometry};

impl ResourceManager {
    pub(crate) fn prepare_geometry(&mut self, assets: &AssetServer, handle: GeometryHandle) {
        let geometry = if let Some(geo) = assets.get_geometry(handle) { geo } else {
            log::warn!("Geometry {:?} not found in AssetServer.", handle);
            return;
        };

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            if geometry.structure_version() == gpu_geo.version && geometry.data_version() == gpu_geo.last_data_version {
                gpu_geo.last_used_frame = self.frame_index;
                return;
            }
        }

        let mut buffer_ids_changed = false;
        let needs_data_update = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.data_version() != gpu_geo.last_data_version
        } else { true };

        if needs_data_update {
            for attr in geometry.attributes().values() {
                let new_id = self.prepare_attribute_buffer(attr);
                if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                    if !gpu_geo.vertex_buffer_ids.contains(&new_id) {
                        buffer_ids_changed = true;
                    }
                }
            }
            if let Some(indices) = geometry.index_attribute() {
                let new_id = self.prepare_attribute_buffer(indices);
                if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                    if let Some((_, _, _, old_id)) = gpu_geo.index_buffer {
                        if old_id != new_id { buffer_ids_changed = true; }
                    }
                }
            }
        }

        let needs_rebuild = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.structure_version() > gpu_geo.version || buffer_ids_changed
        } else { true };

        if needs_rebuild {
            self.create_gpu_geometry(geometry, handle);
        } else if needs_data_update {
            if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
                gpu_geo.last_data_version = geometry.data_version();
            }
        }

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            gpu_geo.last_used_frame = self.frame_index;
        }
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry, handle: GeometryHandle) {
        let layout_info = Arc::new(crate::renderer::pipeline::vertex::generate_vertex_layout(geometry));

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();

        for layout_desc in &layout_info.buffers {
            let gpu_buf = self.gpu_buffers.get(&layout_desc.buffer.id()).expect("Vertex buffer should be prepared");
            vertex_buffers.push(gpu_buf.buffer.clone());
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = geometry.index_attribute() {
            let gpu_buf = self.gpu_buffers.get(&indices.buffer.id()).expect("Index buffer should be prepared");
            let format = match indices.format {
                wgpu::VertexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                wgpu::VertexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                _ => wgpu::IndexFormat::Uint16,
            };
            Some((gpu_buf.buffer.clone(), format, indices.count, gpu_buf.id))
        } else { None };

        let mut draw_range = geometry.draw_range.clone();
        if draw_range == (0..u32::MAX) {
            if let Some(attr) = geometry.attributes().get("position") {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else if let Some(attr) = geometry.attributes().values().next() {
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else {
                draw_range = 0..0;
            }
        }

        let gpu_geo = GpuGeometry {
            layout_info,
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            draw_range,
            instance_range: 0..1,
            version: geometry.structure_version(),
            last_data_version: geometry.data_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_geometries.insert(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
        self.gpu_geometries.get(handle)
    }
}
