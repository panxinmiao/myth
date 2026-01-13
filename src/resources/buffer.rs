use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use bytemuck::Pod;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct BufferData {
    pub id: u64,
    pub label: String,
    pub usage: wgpu::BufferUsages,
    pub size: usize,
    data: Mutex<Option<Vec<u8>>>,
}

#[derive(Debug, Clone)]
pub struct BufferRef(Arc<BufferData>);

impl PartialEq for BufferRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for BufferRef {}

impl std::hash::Hash for BufferRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

impl BufferRef {
    pub fn new<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        let size = raw_data.len();
        Self(Arc::new(BufferData {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            usage,
            size,
            data: Mutex::new(Some(raw_data)),
        }))
    }

    pub fn from_bytes(data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self(Arc::new(BufferData {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            usage,
            size: data.len(),
            data: Mutex::new(Some(data.to_vec())),
        }))
    }

    pub fn empty(usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self(Arc::new(BufferData {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            usage,
            size: 0,
            data: Mutex::new(Some(Vec::new())),
        }))
    }

    pub fn new_with_capacity(capacity: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let data = vec![0u8; capacity];
        Self(Arc::new(BufferData {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            label: label.unwrap_or("Buffer").to_string(),
            usage,
            size: capacity,
            data: Mutex::new(Some(data)),
        }))
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.0.usage
    }

    pub fn label(&self) -> &str {
        &self.0.label
    }

    pub fn size(&self) -> usize {
        self.0.size
    }

    pub fn take_data(&self) -> Option<Vec<u8>> {
        self.0.data.lock().unwrap().take()
    }

    pub fn peek_data<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let guard = self.0.data.lock().unwrap();
        guard.as_ref().map(|data| f(data))
    }

    pub fn update<T: Pod>(&self, data: &[T]) {
        let raw_data = bytemuck::cast_slice(data).to_vec();
        *self.0.data.lock().unwrap() = Some(raw_data);
    }

    pub fn update_bytes(&self, data: &[u8]) {
        *self.0.data.lock().unwrap() = Some(data.to_vec());
    }
}