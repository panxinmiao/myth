//! Buffer 相关操作

use rustc_hash::FxHashMap;

use crate::resources::buffer::BufferRef;
use super::{ResourceManager, GpuBuffer, EnsureResult};

impl ResourceManager {

    /// 静态辅助方法：只借用必要的字段，解决 borrow checker 冲突
    /// 可以在持有 ResourceManager 其他字段引用的同时调用此方法
    /// 
    /// 返回 EnsureResult，包含物理资源 ID 和是否重建的标志
    pub fn write_buffer_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_buffers: &mut FxHashMap<u64, GpuBuffer>,
        frame_index: u64,
        buffer_ref: &BufferRef,
        data: &[u8],
    ) -> EnsureResult {
        let cpu_id = buffer_ref.id();
        let mut was_recreated = false;

        match gpu_buffers.entry(cpu_id) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let gpu_buf = entry.get_mut();
                
                // 检查版本并上传
                if buffer_ref.version > gpu_buf.last_uploaded_version {
                    if (data.len() as u64) > gpu_buf.size {
                        log::debug!("Resizing buffer {:?}...", buffer_ref.label());
                        let old_id = gpu_buf.id;
                        // 原地替换
                        *gpu_buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
                        was_recreated = gpu_buf.id != old_id;
                    } else {
                        queue.write_buffer(&gpu_buf.buffer, 0, data);
                    }
                    gpu_buf.last_uploaded_version = buffer_ref.version;
                }
                gpu_buf.last_used_frame = frame_index;
                EnsureResult::new(gpu_buf.id, was_recreated)
            },
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut buf = GpuBuffer::new(device, data, buffer_ref.usage, buffer_ref.label());
                buf.last_uploaded_version = buffer_ref.version;
                buf.last_used_frame = frame_index;
                let id = buf.id;
                entry.insert(buf);
                EnsureResult::created(id)
            }
        }
    }


    /// 确保 CpuBuffer 对应的 GpuBuffer 已经创建并上传最新数据
    /// 
    /// 返回 EnsureResult，包含物理资源 ID 和是否重建的标志
    pub fn ensure_buffer<T: super::GpuData>(&mut self, cpu_buffer: &super::CpuBuffer<T>) -> EnsureResult {
        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            cpu_buffer.handle(),
            cpu_buffer.as_bytes(),
        )
    }

    /// 确保 CpuBuffer 对应的 GpuBuffer 已经创建并上传最新数据
    /// 
    /// 仅返回物理资源 ID（兼容旧代码）
    #[inline]
    pub fn ensure_buffer_id<T: super::GpuData>(&mut self, cpu_buffer: &super::CpuBuffer<T>) -> u64 {
        self.ensure_buffer(cpu_buffer).resource_id
    }

    pub fn prepare_attribute_buffer(&mut self, attr: &crate::resources::geometry::Attribute) -> EnsureResult {
        let cpu_id = attr.buffer.id();
        let mut was_recreated = false;

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&cpu_id) {
            if attr.version > gpu_buf.last_uploaded_version
                && let Some(data) = &attr.data {
                    let bytes: &[u8] = data.as_ref();
                    
                    // 检查是否需要扩容
                    if (bytes.len() as u64) > gpu_buf.size {
                        let old_id = gpu_buf.id;
                        *gpu_buf = GpuBuffer::new(&self.device, bytes, attr.buffer.usage(), attr.buffer.label());
                        was_recreated = gpu_buf.id != old_id;
                    } else {
                        self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                    }
                    gpu_buf.last_uploaded_version = attr.version;
                }
            gpu_buf.last_used_frame = self.frame_index;
            return EnsureResult::new(gpu_buf.id, was_recreated);
        }

        if let Some(data) = &attr.data {
            let bytes: &[u8] = data.as_ref();
            let mut gpu_buf = GpuBuffer::new(&self.device, bytes, attr.buffer.usage(), attr.buffer.label());
            gpu_buf.last_uploaded_version = attr.version;
            gpu_buf.last_used_frame = self.frame_index;
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(cpu_id, gpu_buf);
            EnsureResult::created(buf_id)
        } else {
            log::error!("Geometry attribute buffer {:?} missing CPU data!", attr.buffer.label());
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(&cpu_id) {
                return EnsureResult::existing(gpu_buf.id);
            }
            let dummy_data = [0u8; 1];
            let gpu_buf = GpuBuffer::new(&self.device, &dummy_data, attr.buffer.usage(), Some("Dummy Fallback Buffer"));
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(cpu_id, gpu_buf);
            EnsureResult::created(buf_id)
        }
    }

    pub fn prepare_uniform_slot_data(&mut self, slot_id: u64, data: &[u8], label: &str) -> EnsureResult {
        let mut was_recreated = false;
        let existed = self.gpu_buffers.contains_key(&slot_id);
        
        let gpu_buf = self.gpu_buffers.entry(slot_id).or_insert_with(|| {
            was_recreated = true;
            let mut buf = GpuBuffer::new(&self.device, data, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some(label));
            buf.enable_shadow_copy();
            buf
        });
        
        if existed {
            let size_changed = gpu_buf.update_with_data(&self.device, &self.queue, data);
            if size_changed {
                was_recreated = true;
            }
        }
        
        gpu_buf.last_used_frame = self.frame_index;
        EnsureResult::new(gpu_buf.id, was_recreated)
    }
}
