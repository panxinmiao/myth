use thunderdome::Index;
use crate::assets::{GeometryHandle, MaterialHandle};
use crate::renderer::managers::CachedBindGroupId;

pub type MeshHandle = Index;

/// 渲染代理缓存
/// 
/// 存储 Mesh 实例的渲染缓存数据，避免每帧重复查找
#[derive(Debug, Clone, Default)]
pub struct RenderCache {
    /// 缓存的 BindGroup ID（用于快速路径）
    pub bind_group_id: Option<CachedBindGroupId>,
    /// 缓存的 Pipeline ID
    pub pipeline_id: Option<u16>,
    /// 缓存版本号（用于失效检测）
    pub cache_version: u64,
}

impl RenderCache {
    /// 使缓存失效
    #[inline]
    pub fn invalidate(&mut self) {
        self.bind_group_id = None;
        self.pipeline_id = None;
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: String,
    
    // === 场景图节点 ===
    pub node_id: Option<Index>,
    
    // === 资源引用 ===
    pub geometry: GeometryHandle,
    pub material: MaterialHandle,
    
    // === 实例特定的渲染设置 ===
    pub visible: bool, 
    
    // 绘制顺序 (Render Order)
    pub render_order: i32,
    
    // === 渲染缓存 ===
    /// 渲染代理缓存，避免每帧重复查找
    pub render_cache: RenderCache,
}

impl Mesh {
    pub fn new(
        geometry: GeometryHandle, 
        material: MaterialHandle
    ) -> Self {
        Self {
            name: "Mesh".to_string(),
            node_id : None,
            geometry,
            material,
            visible: true,
            render_order: 0,
            render_cache: RenderCache::default(),
        }
    }
    
    /// 使渲染缓存失效（当 geometry 或 material 改变时调用）
    pub fn invalidate_cache(&mut self) {
        self.render_cache.invalidate();
    }
}