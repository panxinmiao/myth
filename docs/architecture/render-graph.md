# Render Graph: A Declarative, SSA-based Compiler

Modern graphics APIs give developers unprecedented control over GPU resources and synchronization. But that control comes at a price. Once a renderer needs to handle a chain of complex effects, it's easy to drown in managing resource lifetimes, memory barriers, and transient allocations.

One of Myth Engine's core strengths is a built-in **strict, declarative Render Dependency Graph (RDG) compiler based on SSA (Static Single Assignment)**.

## 1. Core Idea: A Strict SSA Architecture

A RenderGraph should not be just a HashMap for sharing textures; it should be a **compiler**.

Myth's RDG is rooted in **SSA** from compiler design: *each variable is assigned exactly once*. In traditional rendering, a pass might "bind a texture and draw into it in place." In Myth's SSA architecture, a logical resource (`TextureNodeId`) is strictly immutable.

**How do we handle Read-Modify-Write?**
We introduce the concept of **aliasing**:
when a pass needs to modify a texture, it consumes the previous logical version and produces a **new** logical version. The graph compiler understands this topological chain and guarantees that, at the physical level, **they point to the same physical GPU memory**.

```rust
let pass_out = graph.add_pass("Some_Pass", |builder| {
    // Declare a read-only dependency on the input resource
    builder.read_texture(input_id);

    // Declare a new logical resource aliasing the input (Read-Modify-Write)
    let output_texture = builder.mutate_texture(input_id, "Some_Out_Res", TextureDesc::new(...));

    // ...
});
```

## 2. Compile-Time Optimization: Zero-Overhead, Fully Automatic

The compiler's lifecycle is strictly divided into: **Setup (declaration) → Compilation → Preparation → Execution**.

Through this architecture, the engine performs extremely complex low-level optimizations for you automatically:

### Automatic Memory Aliasing

In a complex post-processing pipeline, the compiler intelligently overlaps logically distinct, immutable resources onto the exact same transient GPU texture. Zero VRAM waste.

### Dead Pass Elimination (DPE)

If we disable certain advanced effects (e.g. SSAO), their upstream dependency nodes (such as a Normal Pre-pass that existed only for SSAO) become zero-referenced. During compilation the compiler detects this and marks them Dead, automatically bypassing their physical memory allocation and GPU command recording. **Zero configuration throughout.**

### Zero-Allocation Just-in-Time Compilation

Myth opts for a **per-frame rebuild** strategy.
Thanks to a `FrameArena` pure bump allocator and highly optimized data structures, total per-frame compilation takes only about **1.6 microseconds** in complex scenes (less than 0.01% of a 60fps frame budget), achieving true $O(n)$ linear scaling and avoiding the latent bugs of maintaining a giant topology cache.

## 3. Dynamic Topology Visualization

Myth can dump the render graph topology live at runtime. Below is a partial, auto-generated graph of the High-Fidelity pipeline with multiple post-processing effects enabled:

*(Note: a single arrow `-->` represents a logical data dependency; a double arrow `==>` represents physical memory aliasing / in-place reuse.)*

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

Leave the complexity to the compiler, and give the creativity back to the rendering engineer. That is the power of Myth's RDG.
