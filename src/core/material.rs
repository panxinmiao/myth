use uuid::Uuid;
use glam::{Vec3, Vec4};
use bytemuck::{Pod, Zeroable};
use bitflags::bitflags;

// ============================================================================
// 1. GPU 数据层 (POD - Plain Old Data)
// 只有需要直接 memcpy 到 UniformBuffer 的数据才放在这里
// 必须严格遵守 std140 内存布局 (16字节对齐)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct MeshBasicUniforms {
    pub color: Vec4, // 16 bytes
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshStandardUniforms {
    pub color: Vec4,       // 16 bytes
    pub emissive: Vec3,    // 12 bytes
    pub roughness: f32,    // 4 bytes (紧跟 Vec3 填补空缺) -> 共 16 bytes
    pub metalness: f32,    // 4 bytes
    pub _padding: [f32; 3],// 12 bytes (补齐 alignment) -> 共 16 bytes
}

impl Default for MeshStandardUniforms {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            emissive: Vec3::ZERO,
            roughness: 0.5,
            metalness: 0.0,
            _padding: [0.0; 3],
        }
    }
}

// ============================================================================
// 2. 材质特性标志位 (Feature Flags)
// 用于生成 Shader 的 #define 宏，替代之前的 HashMap key 检查
// ============================================================================

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct MaterialFeatures: u32 {
        const USE_MAP           = 1 << 0; // 基础颜色贴图
        const USE_NORMAL_MAP    = 1 << 1; // 法线贴图
        const USE_ROUGHNESS_MAP = 1 << 2; // 粗糙度贴图
        const USE_EMISSIVE_MAP  = 1 << 3; // 自发光贴图
        // const USE_SKINNING   = 1 << 4; // 骨骼动画 (预留)
    }
}

// ============================================================================
// 3. 具体材质逻辑结构体 (User Layer)
// 这里消除了“概念泄漏”：Basic 材质里根本没有 roughness 字段
// ============================================================================

#[derive(Debug, Clone)]
pub struct MeshBasicMaterial {
    pub uniforms: MeshBasicUniforms,
    // Basic 材质只支持基础贴图
    pub map: Option<Uuid>, 
}

#[derive(Debug, Clone)]
pub struct MeshStandardMaterial {
    pub uniforms: MeshStandardUniforms,
    // Standard 材质支持完整的 PBR 贴图
    pub map: Option<Uuid>,
    pub normal_map: Option<Uuid>,
    pub roughness_map: Option<Uuid>,
    pub emissive_map: Option<Uuid>,
}

// ============================================================================
// 4. 材质类型枚举 (Polymorphic Container)
// 引擎核心通过这个 Enum 统一管理不同类型的材质
// ============================================================================

#[derive(Debug, Clone)]
pub enum MaterialType {
    Basic(MeshBasicMaterial),
    Standard(MeshStandardMaterial),
    // 未来可扩展: Toon(MeshToonMaterial), Shader(ShaderMaterial) ...
}

// ============================================================================
// 5. 主 Material 结构体 (The Public API)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Material {
    pub id: Uuid,
    pub version: u64,
    pub name: Option<String>,
    
    // 核心：持有具体的材质数据
    pub data: MaterialType,

    // 通用渲染状态 (Render States) - 所有材质共享
    pub transparent: bool,
    pub opacity: f32,
    pub depth_write: bool,
    pub depth_test: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub side: u32, // 0: Front, 1: Back, 2: Double (配合 cull_mode 使用)
}

impl Material {
    /// 构造基础材质
    pub fn new_basic(color: Vec4) -> Self {
        Self {
            id: Uuid::new_v4(),
            version: 0,
            name: Some("MeshBasic".to_string()),
            // 只能初始化 Basic 允许的字段
            data: MaterialType::Basic(MeshBasicMaterial {
                uniforms: MeshBasicUniforms { color },
                map: None,
            }),
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    /// 构造标准 PBR 材质
    pub fn new_standard() -> Self {
        Self {
            id: Uuid::new_v4(),
            version: 0,
            name: Some("MeshStandard".to_string()),
            // Standard 拥有完整的 PBR 字段
            data: MaterialType::Standard(MeshStandardMaterial {
                uniforms: MeshStandardUniforms::default(),
                map: None,
                normal_map: None,
                roughness_map: None,
                emissive_map: None,
            }),
            transparent: false,
            opacity: 1.0,
            depth_write: true,
            depth_test: true,
            cull_mode: Some(wgpu::Face::Back),
            side: 0,
        }
    }

    /// 计算当前材质启用的特性 (Feature Flags)
    /// ShaderGenerator 会使用这个来注入 #define
    pub fn features(&self) -> MaterialFeatures {
        let mut features = MaterialFeatures::empty();
        match &self.data {
            MaterialType::Basic(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
            }
            MaterialType::Standard(m) => {
                if m.map.is_some() { features |= MaterialFeatures::USE_MAP; }
                if m.normal_map.is_some() { features |= MaterialFeatures::USE_NORMAL_MAP; }
                if m.roughness_map.is_some() { features |= MaterialFeatures::USE_ROUGHNESS_MAP; }
                if m.emissive_map.is_some() { features |= MaterialFeatures::USE_EMISSIVE_MAP; }
            }
        }
        features
    }

    /// 获取 Shader 模板名称
    pub fn shader_name(&self) -> &'static str {
        match &self.data {
            MaterialType::Basic(_) => "MeshBasic",
            MaterialType::Standard(_) => "MeshStandard",
        }
    }

    /// 获取 Uniform Buffer 的字节数据 (零拷贝)
    /// 直接用于 wgpu::Queue::write_buffer
    pub fn as_bytes(&self) -> &[u8] {
        match &self.data {
            MaterialType::Basic(m) => bytemuck::bytes_of(&m.uniforms),
            MaterialType::Standard(m) => bytemuck::bytes_of(&m.uniforms),
        }
    }
    
    /// 获取 WGSL 对应的结构体定义 (单一事实来源)
    /// 必须与上面的 #[repr(C)] 结构体严格一致
    pub fn wgsl_struct_def(&self) -> &'static str {
        match &self.data {
            MaterialType::Basic(_) => r#"
                struct MaterialUniforms {
                    color: vec4<f32>,
                };
            "#,
            MaterialType::Standard(_) => r#"
                struct MaterialUniforms {
                    color: vec4<f32>,
                    emissive: vec3<f32>,
                    roughness: f32,
                    metalness: f32,
                };
            "#,
        }
    }

    /// 获取所有纹理资源 ID 及其对应的 BindGroup 槽位顺序
    /// 返回值: Vec<Option<Uuid>>
    /// 顺序必须与 ShaderGenerator 中的 @binding(n) 顺序一致
    pub fn textures(&self) -> Vec<Option<Uuid>> {
        match &self.data {
            MaterialType::Basic(m) => vec![m.map],
            // 顺序约定: [Map, Normal, Roughness, Emissive]
            MaterialType::Standard(m) => vec![
                m.map, 
                m.normal_map, 
                m.roughness_map, 
                m.emissive_map
            ],
        }
    }

    /// 标记材质数据已变更 (用户修改属性后必须调用，或者封装在 setter 中)
    pub fn needs_update(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}