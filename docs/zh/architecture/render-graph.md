# Render Graph：基于 SSA 的声明式渲染图

现代图形 API 赋予了开发者前所未有的 GPU 资源与同步控制能力。但这种控制是有代价的。一旦渲染器需要处理复杂的特效链，开发者很容易深陷于管理资源生命周期、内存屏障和瞬时内存分配的泥潭之中。

Myth Engine 的核心竞争力之一，就是内置了一个**基于 SSA（静态单赋值）的、严格的、声明式的渲染图编译器 (Render Dependency Graph, 简称 RDG)**。

## 1. 核心理念：严格的 SSA 架构

一个 RenderGraph 不应该只是一个用来共享纹理的 HashMap；它应该是一个**编译器**。

Myth 的 RDG 核心思想来源于编译器设计中的 **SSA (Static Single Assignment, 静态单赋值)**：*每个变量只被赋值一次*。在传统的渲染中，一个 Pass 可能会“绑定一个纹理并原位绘制修改它”。而在 Myth 的 SSA 架构中，逻辑资源（`TextureNodeId`）是严格不可变的。

**如何处理原位修改 (Read-Modify-Write)？**
我们引入了 **别名 (Aliasing)** 的概念：
当一个 Pass 需要修改纹理时，它会消费前一个逻辑版本并产生一个 **新的** 逻辑版本。图编译器理解这个拓扑链，并保证在物理层面，**它们指向同一块物理 GPU 内存**。

```rust
let pass_out = graph.add_pass("Some_Pass", |builder| {
    // 声明对输入资源的只读依赖
    builder.read_texture(input_id);
    
    // 声明一个 alias 输入资源的新逻辑资源 (Read-Modify-Write)
    let output_texture = builder.mutate_texture(input_id, "Some_Out_Res", TextureDesc::new(...));
    
    // ...
});

```

## 2. 编译期优化：零开销、全自动

图编译器的生命周期被严格划分为：**Setup (声明) -> Compilation (编译) -> Preparation (准备) -> Execution (执行)**。

通过这种架构，引擎自动为你完成了极其复杂的底层优化：

### 自动内存别名 (Memory Aliasing)

在复杂的后处理管线中，逻辑上完全不同且不可变的资源，编译器会智能地将它们的分配重叠到完全相同的临时 GPU 纹理上。零显存浪费。

### 死节点剔除 (Dead Pass Elimination, DPE)

如果我们禁用了某些高级特效（例如 SSAO），其前置的依赖节点（如仅仅为了 SSAO 存在的 Normal Pre-pass）会变成零引用状态。编译器在编译期间会自动检测并将其标记为 Dead，自动绕过其物理内存分配和 GPU 命令录制。**全程零配置。**

### 零分配的即时编译

Myth 选择了**每帧重建 (per-frame rebuild)** 图的策略。
由于采用了 `FrameArena` 纯 bump 分配器和高度优化的数据结构，单帧编译总耗时在复杂场景下仅约 **1.6 微秒**（占 60fps 帧预算不到 0.01%），实现了真正的 $O(n)$ 线性扩展，免去了维护庞大拓扑缓存带来的潜在 Bug。

## 3. 动态拓扑可视化

Myth 引擎支持在运行时实时 Dump 渲染图拓扑。以下是开启多项后处理的高保真渲染管线 (High-Fidelity) 自动生成的局部图谱：

*(注：单箭头 `-->` 代表逻辑数据依赖；双箭头 `==>` 代表物理内存别名 / 就地复用)*

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

将复杂性交给编译器，把创造力还给渲染工程师。这就是 Myth RDG 的力量。

