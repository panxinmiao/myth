//! Geometry 相关操作

use crate::assets::{AssetServer, GeometryHandle};
use crate::resources::geometry::Geometry;

use super::{ResourceManager, GpuGeometry};

/// Geometry 准备结果
/// 
/// Vertex Buffer IDs 用于 Pipeline 缓存验证，不影响 Object BindGroup
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GeometryPrepareResult {
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer_id: Option<u64>,
    pub any_recreated: bool,
}

impl ResourceManager {
    /// 准备 Geometry 的 GPU 资源
    /// 
    /// 返回 GeometryPrepareResult，包含所有物理资源 ID
    pub(crate) fn prepare_geometry(&mut self, assets: &AssetServer, handle: GeometryHandle) -> Option<GeometryPrepareResult> {
        let geometry = assets.geometries.get(handle)?;

        // 快速路径：检查是否需要任何更新
        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            if geometry.structure_version() == gpu_geo.version && geometry.data_version() == gpu_geo.last_data_version {
                gpu_geo.last_used_frame = self.frame_index;
                return Some(GeometryPrepareResult {
                    vertex_buffer_ids: gpu_geo.vertex_buffer_ids.clone(),
                    index_buffer_id: gpu_geo.index_buffer.as_ref().map(|(_, _, _, id)| *id),
                    any_recreated: false,
                });
            }
        }

        // 确保所有属性缓冲区
        let mut any_buffer_recreated = false;
        let mut new_vertex_ids = Vec::new();
        
        for attr in geometry.attributes().values() {
            let result = self.prepare_attribute_buffer(attr);
            new_vertex_ids.push(result.resource_id);
            if result.was_recreated {
                any_buffer_recreated = true;
            }
        }
        
        let mut new_index_id = None;
        if let Some(indices) = geometry.index_attribute() {
            let result = self.prepare_attribute_buffer(indices);
            new_index_id = Some(result.resource_id);
            if result.was_recreated {
                any_buffer_recreated = true;
            }
        }

        // 检查是否需要重建 GpuGeometry
        let needs_rebuild = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.structure_version() > gpu_geo.version || 
            any_buffer_recreated ||
            gpu_geo.vertex_buffer_ids != new_vertex_ids
        } else { 
            true 
        };

        if needs_rebuild {
            self.create_gpu_geometry(&geometry, handle);
        } else {
            // 仅更新数据版本
            if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
                gpu_geo.last_data_version = geometry.data_version();
            }
        }

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            gpu_geo.last_used_frame = self.frame_index;
            Some(GeometryPrepareResult {
                vertex_buffer_ids: gpu_geo.vertex_buffer_ids.clone(),
                index_buffer_id: new_index_id,
                any_recreated: any_buffer_recreated || needs_rebuild,
            })
        } else {
            None
        }
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry, handle: GeometryHandle) {
        let layout_info = crate::renderer::pipeline::vertex::generate_vertex_layout(geometry);

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
