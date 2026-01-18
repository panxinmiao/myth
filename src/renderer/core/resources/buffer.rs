//! Buffer 相关操作

use rustc_hash::FxHashMap;

use crate::resources::buffer::BufferRef;
use super::{ResourceManager, GpuBuffer};

impl ResourceManager {

    /// 静态辅助方法：只借用必要的字段，解决 borrow checker 冲突
    /// 可以在持有 ResourceManager 其他字段引用的同时调用此方法
    pub fn write_buffer_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_buffers: &mut FxHashMap<u64, GpuBuffer>,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> u64 {
        let id = buffer_ref.id();
        
        // 1. 获取或创建 GpuBuffer
        let gpu_buf = gpu_buffers.entry(id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
            buf.last_uploaded_version = 0; // 强制新 buffer 必须上传
            buf
        });

        // 2. 检查版本并上传
        // 注意：这里不需要再 clone data 了，直接用 slice
        if buffer_ref.version > gpu_buf.last_uploaded_version {
            // 如果 GPU buffer 太小，需要 resize (会销毁重建)
            if (data.len() as u64) > gpu_buf.size {
                log::debug!("Resizing buffer {:?} from {} to {}", buffer_ref.label(), gpu_buf.size, data.len());
                *gpu_buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
            } else {
                queue.write_buffer(&gpu_buf.buffer, 0, data);
            }
            gpu_buf.last_uploaded_version = buffer_ref.version;
        }
        
        gpu_buf.last_used_frame = frame_index;
        gpu_buf.id
    }


    pub fn write_buffer(&mut self, buffer_ref: &BufferRef, data: &[u8]) -> u64 {
        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            buffer_ref,
            data,
        )
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
