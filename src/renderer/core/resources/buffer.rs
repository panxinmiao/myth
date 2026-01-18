//! Buffer 相关操作

use crate::resources::buffer::BufferRef;
use super::{ResourceManager, GpuBuffer};

impl ResourceManager {
    pub fn write_buffer(&mut self, buffer_ref: &BufferRef, data: &[u8]) -> u64 {
        let id = buffer_ref.id();
        let gpu_buf = if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            gpu_buf
        } else {
            let gpu_buf = GpuBuffer::new(&self.device, data, buffer_ref.usage(), buffer_ref.label());
            self.gpu_buffers.insert(id, gpu_buf);
            self.gpu_buffers.get_mut(&id).unwrap()
        };

        if buffer_ref.version > gpu_buf.last_uploaded_version {
            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
            gpu_buf.last_uploaded_version = buffer_ref.version;
        }
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    pub fn prepare_attribute_buffer(&mut self, attr: &crate::resources::geometry::Attribute) -> u64 {
        let id = attr.buffer.id();

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            if attr.version > gpu_buf.last_uploaded_version
                && let Some(data) = &attr.data {
                    let bytes: &[u8] = data.as_ref();
                    self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                    gpu_buf.last_uploaded_version = attr.version;
                }
            gpu_buf.last_used_frame = self.frame_index;
            return gpu_buf.id;
        }

        if let Some(data) = &attr.data {
            let bytes: &[u8] = data.as_ref();
            let mut gpu_buf = GpuBuffer::new(&self.device, bytes, attr.buffer.usage(), attr.buffer.label());
            gpu_buf.last_uploaded_version = attr.version;
            gpu_buf.last_used_frame = self.frame_index;
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        } else {
            log::error!("Geometry attribute buffer {:?} missing CPU data!", attr.buffer.label());
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
                return gpu_buf.id;
            }
            let dummy_data = [0u8; 1];
            let gpu_buf = GpuBuffer::new(&self.device, &dummy_data, attr.buffer.usage(), Some("Dummy Fallback Buffer"));
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        }
    }

    pub fn prepare_uniform_slot_data(&mut self, slot_id: u64, data: &[u8], label: &str) -> u64 {
        let gpu_buf = self.gpu_buffers.entry(slot_id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(&self.device, data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some(label));
            buf.enable_shadow_copy();
            buf
        });
        gpu_buf.update_with_data(&self.device, &self.queue, data);
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }
}
