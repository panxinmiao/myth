use uuid::Uuid;
use std::sync::{Arc, RwLock};
use bytemuck::Pod;

/// 通用 CPU 端数据缓冲
/// 统一管理 Vertex, Index, Uniform, Storage 等数据的 CPU 副本
#[derive(Debug)]
pub struct CpuBuffer {
    pub id: Uuid,
    pub label: String,
    pub data: Vec<u8>,
    pub version: u64,
    pub usage: wgpu::BufferUsages,
}

impl CpuBuffer {
    pub fn new<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: label.unwrap_or("Buffer").to_string(),
            data: bytemuck::cast_slice(data).to_vec(),
            version: 0,
            usage,
        }
    }

    /// 更新数据并增加版本号
    pub fn update<T: Pod>(&mut self, data: &[T]) {
        self.data = bytemuck::cast_slice(data).to_vec();
        self.version = self.version.wrapping_add(1);
    }
}

/// 线程安全的 Buffer 引用
pub type BufferRef = Arc<RwLock<CpuBuffer>>;