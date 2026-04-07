[English](README.md)

---
<div align="center">

# Myth

**基于 wgpu 的高性能 Rust 渲染引擎。**

[![CI](https://github.com/panxinmiao/myth/actions/workflows/ci.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](LICENSE)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-Ready-green.svg)](https://gpuweb.github.io/gpuweb/)

[![Myth Engine Hero](https://raw.githubusercontent.com/panxinmiao/myth/main/docs/images/hero.jpg)](https://panxinmiao.github.io/myth/)

[**演示案例**](https://panxinmiao.github.io/myth/) | [**glTF 示例**](https://panxinmiao.github.io/myth/gltf/) | [**glTF 查看器与检查器**](https://panxinmiao.github.io/myth/gltf_viewer/)  | [**示例代码**](examples/)

</div>

---

> 📢 当前状态：Beta 阶段
>
> Myth 已进入 Beta 阶段。核心架构已趋于稳定，可用于实际项目。API 仍在演进过程中，可能会有不兼容的变更。

## 简介

**Myth** 是一款对开发者友好的、使用 **Rust** 编写的高性能 3D 渲染引擎。 

它深受 **Three.js** 简洁且符合直觉的 API 设计启发，并基于现代图形 API **wgpu** 构建。Myth 旨在弥合底层图形 API 与高层游戏引擎之间的鸿沟。

## 为什么选择 Myth？

wgpu 的功能极其强大 —— 但即使是绘制一个简单的场景，也需要编写数百行样板代码。  

而当你仅仅需要一个精简的渲染库时，Bevy 和 Godot 又显得过于笨重。  

Myth 通过引入**严格的基于 SSA（静态单赋值）的渲染图 (RenderGraph)** 解决了这个问题，它将渲染过程视为一个编译器问题：
- 自动拓扑排序 + 死阶段消除 (Dead-pass elimination)
- 激进的临时内存别名复用 (完全无需手动管理屏障)
- 复杂度为 O(n) 的每帧重建，且零内存分配

**一套代码，全平台运行**：  
原生 (Windows, macOS, Linux, iOS, Android) + WebGPU/WASM + Python 绑定。


## 核心特性

* **核心架构与平台**
    * **真正的跨平台，一套代码**：原生 (Windows, macOS, Linux, iOS, Android) + WebGPU/WASM + Python 绑定。
    * **现代后端**：基于 **wgpu** 构建，全面支持 Vulkan、Metal、DX12 和 WebGPU。
    * **基于 SSA 的渲染图**：声明式、由编译器驱动的渲染架构。你只需声明拓扑需求，引擎会处理剩下的所有事情：
        * **自动同步**：零手动内存屏障或布局转换。
        * **激进的内存别名**：在不同的逻辑阶段之间，完美复用高分辨率的临时物理纹理。
        * **死阶段消除**：自动剔除无用的渲染工作负载。
        * **零分配的每帧重建**：每帧都会对整个 DAG（有向无环图）进行评估与编译。
    * **无头与离线渲染**
        * **服务端就绪**：无需窗口表面 (Window Surface) 即可全功能运行。非常适合 CI/CD 自动化测试、云端渲染和离线视频生成。
        * **高吞吐量回读流**：内置基于环形缓冲区 (Ring-buffer) 架构的非阻塞、异步 GPU 到 CPU 像素回读管线，并带有自动背压 (Back-pressure) 与防 OOM 熔断机制。

* **高级渲染与光照**
    * **基于物理的材质 (PBR)**：强大的 PBR 管线，支持清漆 (Clearcoat)、彩虹色 (Iridescence)、透射 (Transmission)、光泽 (Sheen) 和各向异性 (Anisotropy)。
    * **基于图像的光照 (IBL)** + **动态阴影 (CSM)**。
    * **屏幕空间环境光遮蔽 (SSAO) / 屏幕空间次表面散射 (SSSS) / 天空盒 (Skybox)**。

* **后期处理与特效**
    * **HDR 管线** + **泛光 (Bloom)** + **色彩分级 (Color Grading)** + **TAA / FXAA / MSAA**。

* **资产与工具链**
    * **完整支持 glTF 2.0**（PBR、动画、变形目标）。
    * **异步资产系统** + **内嵌 egui 检查器**。

## 引擎底层：渲染图编译器 (The Graph Compiler)

Myth 使用了严格的基于 SSA 的 RenderGraph，因此引擎能够：

* 自动调度渲染阶段（拓扑排序）
* 消除未使用的渲染工作（死渲染过程消除）
* 激进地复用内存（瞬时别名复用）

所有这些都无需手动设置内存屏障。

深入了解：[docs/RenderGraph_CN.md](https://github.com/panxinmiao/myth/blob/main/docs/RenderGraph_CN.md)

以下是 Myth 引擎在渲染复杂帧时，自动生成的 RenderGraph 的真实拓扑转储：

<details>
<summary><b>点击展开 RenderGraph 拓扑图</b></summary>

```mermaid
flowchart TD
    classDef alive fill:#2b3c5a,stroke:#4a6f9f,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef dead fill:#222,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5,color:#777,rx:5,ry:5;
    classDef external_out fill:#5a2b3c,stroke:#9f4a6f,stroke-width:2px,color:#fff;
    classDef external_in fill:#3c5a2b,stroke:#6f9f4a,stroke-width:2px,color:#fff;
    P29(["UI_Pass"]):::alive
    subgraph Shadow ["Shadow"]
        direction TB
        P0(["Shadow_Pass"]):::alive
    end
    style Shadow fill:#ec489914,stroke:#ec4899,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
    subgraph Scene ["Scene"]
        direction TB
        P1(["Pre_Pass"]):::alive
        P4(["Opaque_Pass"]):::alive
        P7(["Skybox_Pass"]):::alive
        P14(["Transparent_Pass"]):::alive
        subgraph SSAO_System ["SSAO_System"]
            direction TB
            P2(["SSAO_Raw"]):::alive
            P3(["SSAO_Blur"]):::alive
        end
        style SSAO_System fill:#3b82f614,stroke:#3b82f6,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
        subgraph SSSS_System ["SSSS_System"]
            direction TB
            P5(["SSSS_Blur_H"]):::alive
            P6(["SSSS_Blur_V"]):::alive
        end
        style SSSS_System fill:#10b98114,stroke:#10b981,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
        subgraph TAA_System ["TAA_System"]
            direction TB
            P8(["TAA_Resolve"]):::alive
            P9(["TAA_Save_History_Color"]):::alive
            P10(["TAA_Save_History_Depth"]):::alive
            P11(["CAS_Pass"]):::alive
        end
        style TAA_System fill:#8b5cf614,stroke:#8b5cf6,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
        subgraph Transmission_Map ["Transmission_Map"]
            direction TB
            P12(["Copy_Texture_Pass"]):::alive
            P13(["Generate_Mipmap_Pass"]):::alive
        end
        style Transmission_Map fill:#ec489914,stroke:#ec4899,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
    end
    style Scene fill:#ef444414,stroke:#ef4444,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
    subgraph PostProcess ["PostProcess"]
        direction TB
        P27(["ToneMap_Pass"]):::alive
        P28(["FXAA_Pass"]):::alive
        subgraph Bloom_System ["Bloom_System"]
            direction TB
            P15(["Bloom_Extract"]):::alive
            P16(["Bloom_Downsample_1"]):::alive
            P17(["Bloom_Downsample_2"]):::alive
            P18(["Bloom_Downsample_3"]):::alive
            P19(["Bloom_Downsample_4"]):::alive
            P20(["Bloom_Downsample_5"]):::alive
            P21(["Bloom_Upsample_4"]):::alive
            P22(["Bloom_Upsample_3"]):::alive
            P23(["Bloom_Upsample_2"]):::alive
            P24(["Bloom_Upsample_1"]):::alive
            P25(["Bloom_Upsample_0"]):::alive
            P26(["Bloom_Composite"]):::alive
        end
        style Bloom_System fill:#06b6d414,stroke:#06b6d4,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10
    end
    style PostProcess fill:#8b5cf614,stroke:#8b5cf6,stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10

    %% --- Data Flow (Edges) ---
    IN_13[\"TAA_History_Color_Read"\]:::external_in
    IN_14[\"TAA_History_Depth_Read"\]:::external_in
    OUT_16[/"TAA_History_Color_Write"/]:::external_out
    OUT_17[/"TAA_History_Depth_Write"/]:::external_out
    OUT_35[/"Surface_With_UI"/]:::external_out
    P0 -->|"Shadow_Array_Map"| P4;
    P0 -->|"Shadow_Array_Map"| P14;
    P1 -->|"Scene_Depth"| P2;
    P1 -->|"Scene_Depth"| P3;
    P1 -->|"Scene_Depth"| P4;
    P1 -->|"Scene_Depth"| P5;
    P1 -->|"Scene_Depth"| P6;
    P1 -->|"Scene_Depth"| P7;
    P1 -->|"Scene_Depth"| P8;
    P1 -->|"Scene_Depth"| P10;
    P1 -->|"Scene_Depth"| P14;
    P1 -->|"Scene_Normals"| P2;
    P1 -->|"Scene_Normals"| P3;
    P1 -->|"Scene_Normals"| P5;
    P1 -->|"Scene_Normals"| P6;
    P1 -->|"Feature_ID"| P5;
    P1 -->|"Feature_ID"| P6;
    P1 -->|"Velocity_Buffer"| P8;
    P2 -->|"SSAO_Raw_Tex"| P3;
    P3 -->|"SSAO_Output"| P4;
    P3 -->|"SSAO_Output"| P14;
    P4 -->|"Scene_Color_HDR"| P5;
    P4 -->|"Scene_Color_HDR"| P6;
    P4 -->|"Specular_MRT"| P6;
    P5 -->|"SSSS_Temp"| P6;
    P6 ==>|"Scene_Color_SSSS"| P7;
    P7 ==>|"Scene_Color_Skybox"| P8;
    IN_13 -.-> P8;
    IN_14 -.-> P8;
    P8 -->|"TAA_Resolved"| P9;
    P8 -->|"TAA_Resolved"| P11;
    P9 --> OUT_16;
    P10 --> OUT_17;
    P11 -->|"CAS_Output"| P12;
    P11 -->|"CAS_Output"| P14;
    P12 -->|"Transmission_Tex"| P13;
    P13 ==>|"Transmission_Tex_Mipmapped"| P14;
    P14 ==>|"Scene_Color_Transparent"| P15;
    P14 ==>|"Scene_Color_Transparent"| P26;
    P15 -->|"Bloom_Mip_0"| P16;
    P15 -->|"Bloom_Mip_0"| P25;
    P16 -->|"Bloom_Mip_1"| P17;
    P16 -->|"Bloom_Mip_1"| P24;
    P17 -->|"Bloom_Mip_2"| P18;
    P17 -->|"Bloom_Mip_2"| P23;
    P18 -->|"Bloom_Mip_3"| P19;
    P18 -->|"Bloom_Mip_3"| P22;
    P19 -->|"Bloom_Mip_4"| P20;
    P19 -->|"Bloom_Mip_4"| P21;
    P20 -->|"Bloom_Mip_5"| P21;
    P21 ==>|"Bloom_Up_4"| P22;
    P22 ==>|"Bloom_Up_3"| P23;
    P23 ==>|"Bloom_Up_2"| P24;
    P24 ==>|"Bloom_Up_1"| P25;
    P25 ==>|"Bloom_Up_0"| P26;
    P26 -->|"Scene_Color_Bloom"| P27;
    P27 -->|"LDR_Intermediate"| P28;
    P28 -->|"Surface_View"| P29;
    P29 --> OUT_35;
```

*(* **图例：** *单箭头 `-->` 表示逻辑数据依赖；双箭头 `==>` 表示物理内存别名/原地复用)*

</details>

## 在线演示

直接在你的浏览器中体验引擎（WebGPU 需要 Chrome/Edge 113+）：

  - **[Showcase (首页)](https://panxinmiao.github.io/myth/)**：高性能渲染展示。
  - **[glTF 查看器与检查器](https://panxinmiao.github.io/myth/gltf_viewer/)**：拖拽你自己的 .glb 文件进行查看。
  - **[glTF 示例模型](https://panxinmiao.github.io/myth/gltf/)**：浏览多个使用 Myth 渲染的 Khronos 官方 glTF 资产。

## 快速开始

将 `myth-engine` 添加到你的 `Cargo.toml` 中：

```toml
[dependencies]
myth-engine = "0.2.0"

# 或者从 GitHub 获取最新版本
# myth-engine = { git = "[https://github.com/panxinmiao/myth](https://github.com/panxinmiao/myth)", branch = "main" }

```

### "Hello World"

用不到 50 行代码，实现一个带有棋盘格纹理的旋转立方体：

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _: &dyn Window) -> Self {
        // 0. 创建场景
        let scene = engine.scene_manager.create_active();

        // 1. 创建带有棋盘格纹理的立方体网格
        let tex_handle = engine.assets.checkerboard(512, 64);
        let mesh_handle = scene.spawn_box(
            1.0, 1.0, 1.0, 
            PhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0)).with_map(tex_handle),
            &engine.assets,
        );
        // 2. 设置相机
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene.node(&cam_node_id).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);
        // 3. 添加光源
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 4. 设置更新回调以旋转立方体
        scene.on_update(move |scene, _input, _dt| {
            if let Some(node) = scene.get_node_mut(mesh_handle) {
                let rot_y = Quat::from_rotation_y(0.02);
                let rot_x = Quat::from_rotation_x(0.01);
                node.transform.rotation = node.transform.rotation * rot_y * rot_x;
            }
        });
        Self {}
    }
}

fn main() -> myth::Result<()> {
    App::new().with_title("Myth-Engine Demo").run::<MyApp>()
}
```

### 运行示例

克隆仓库并直接运行示例代码：

```bash
# 运行 earth 示例
cargo run --example earth --release

# 运行 glTF 查看器 (桌面端)
cargo run --example gltf_viewer --release

# 运行影视级演示案例 (Web/WASM)
# 添加 `gltf-meshopt` 特性以启用使用 MeshOptimizer 优化的 glTF 网格。
./scripts/build_wasm.sh showcase --features="gltf-meshopt"
python3 -m http.server 8080 --directory examples/showcase/web

# 运行 glTF 查看器 (Web/WASM)
# 添加 `rdg_inspector` 特性以启用精美的 RenderGraph 可视化。
./scripts/build_wasm.sh gltf_viewer --features="rdg_inspector"
python3 -m http.server 8080 --directory examples/gltf_viewer/web

```

### Python 绑定

Myth 引擎还提供了 Python 绑定，用于快速原型设计和科学可视化。
有关安装方法和示例，请参阅 [Python Bindings](https://github.com/panxinmiao/myth/tree/main/bindings/python)。

## 开源协议

本项目采用 **MIT License** 或 **Apache-2.0 License** 双重授权。