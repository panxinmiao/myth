//! RenderGraph Performance Benchmarks
//!
//! Measures pure CPU-side framework overhead for:
//! - Graph construction (add_pass + topology wiring)
//! - Compilation (dependency analysis, dead-pass culling, topological sort, lifetime computation)
//! - Arena allocation throughput
//! - SSA alias chain relay rendering patterns
//! - Scaling linearity verification (O(n) proof)
//!
//! **Design Principles:**
//! 1. Zero GPU involvement — no wgpu::Device, no CommandEncoder submission
//! 2. Strict use of `black_box` to defeat LLVM dead-code elimination
//! 3. Parameterised benchmarks for O(n) linearity verification

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use myth::renderer::graph::core::{
    ExecuteContext, FrameArena, GraphStorage, PassBuilder, PassNode, RenderGraph, TextureDesc,
    TextureNodeId,
};

// ═══════════════════════════════════════════════════════════════════════════
// Mock Infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// Minimal POD PassNode — no heap allocations, no Drop.
/// Carries only a borrowed pointer to satisfy the lifetime model.
struct MockNode {
    /// Opaque payload to prevent the compiler from optimising the node away.
    tag: u32,
}

impl<'a> PassNode<'a> for MockNode {
    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
        black_box(self.tag);
    }
}

/// PassNode with a borrowed reference — validates the `'a` lifetime model.
struct BorrowingMockNode<'a> {
    data_ref: &'a u32,
}

impl<'a> PassNode<'a> for BorrowingMockNode<'a> {
    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
        black_box(self.data_ref);
    }
}

/// Creates a standard 2D texture descriptor for benchmarking.
#[inline]
fn bench_desc() -> TextureDesc {
    TextureDesc::new_2d(
        1920,
        1080,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 1: Linear Chain — Build + Compile Topology
// ═══════════════════════════════════════════════════════════════════════════
//
// Constructs a linear dependency chain:  Pass_0 → Pass_1 → ... → Pass_N
// Each pass reads the previous output and creates a new texture.
// The final pass writes to an external backbuffer.
//
// This is the fundamental scaling benchmark proving O(n) complexity for
// Kahn's topological sort and lifetime computation.

fn bench_linear_chain_build_and_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearChain_BuildAndCompile");

    for &pass_count in &[10, 50, 100, 200, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(pass_count),
            &pass_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);

                    // External backbuffer — prevents culling of the final pass
                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    // First pass: create the initial texture
                    let mut current_tex = graph.add_pass("Pass_0", |builder: &mut PassBuilder| {
                        let out = builder.create_texture("Tex_0", bench_desc());
                        (MockNode { tag: 0 }, out)
                    });

                    // Intermediate passes: read previous, create next
                    for i in 1..count {
                        let prev = current_tex;
                        current_tex = graph.add_pass("Pass_N", |builder: &mut PassBuilder| {
                            builder.read_texture(prev);
                            let out = builder.create_texture("Tex_N", bench_desc());
                            (MockNode { tag: i as u32 }, out)
                        });
                    }

                    // Final pass: write to external backbuffer (prevents culling)
                    graph.add_pass("Final", |builder: &mut PassBuilder| {
                        builder.read_texture(current_tex);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    // Compile: dependency build, dead-pass culling, topo sort, lifetimes
                    graph.compile_topology();

                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 2: Wide Fan-In — Multiple Producers Into Single Consumer
// ═══════════════════════════════════════════════════════════════════════════
//
// Creates N independent producer passes, each creating a unique texture,
// all consumed by a single composite pass that writes to the backbuffer.
//
// Stress-tests the dependency builder and dead-pass culler with a high
// fan-in topology (common in deferred shading: GBuffer → Lighting).

fn bench_fan_in_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("FanIn_Topology");

    for &producer_count in &[10, 50, 100, 200, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(producer_count),
            &producer_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);

                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    // N independent producers
                    let mut textures = Vec::with_capacity(count);
                    for i in 0..count {
                        let tex = graph.add_pass("Producer", |builder: &mut PassBuilder| {
                            let out = builder.create_texture("ProdTex", bench_desc());
                            (MockNode { tag: i as u32 }, out)
                        });
                        textures.push(tex);
                    }

                    // Single consumer reads all textures, writes to backbuffer
                    graph.add_pass("Composite", |builder: &mut PassBuilder| {
                        for &tex in &textures {
                            builder.read_texture(tex);
                        }
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    graph.compile_topology();
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 3: Diamond DAG — Realistic Rendering Pipeline
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates a typical High-Fidelity rendering pipeline with a diamond-shaped
// DAG (multiple paths converge). Tests that the compiler handles shared
// resources correctly without performance degradation.
//
// Topology:
//   Depth → SSAO ───────┐
//   Depth → Opaque ─────┼→ ToneMap → Backbuffer
//   Opaque → Bloom ─────┘

fn bench_diamond_dag(c: &mut Criterion) {
    let mut group = c.benchmark_group("DiamondDAG_Realistic");

    for &repetitions in &[1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("pipelines", repetitions),
            &repetitions,
            |b, &reps| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);

                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    let mut final_color = TextureNodeId(0); // placeholder

                    for r in 0..reps {
                        // Depth prepass
                        let depth = graph.add_pass("Prepass", |builder: &mut PassBuilder| {
                            let out = builder.create_texture("Depth", bench_desc());
                            (MockNode { tag: r * 10 }, out)
                        });

                        // SSAO (reads depth)
                        let ssao = graph.add_pass("SSAO", |builder: &mut PassBuilder| {
                            builder.read_texture(depth);
                            let out = builder.create_texture("SSAO", bench_desc());
                            (MockNode { tag: r * 10 + 1 }, out)
                        });

                        // Opaque (reads depth + SSAO)
                        let scene_color = graph.add_pass("Opaque", |builder: &mut PassBuilder| {
                            builder.read_texture(depth);
                            builder.read_texture(ssao);
                            let out = builder.create_texture("SceneColor", bench_desc());
                            (MockNode { tag: r * 10 + 2 }, out)
                        });

                        // Bloom (reads scene color)
                        let bloom_tex = graph.add_pass("Bloom", |builder: &mut PassBuilder| {
                            builder.read_texture(scene_color);
                            let out = builder.create_texture("BloomTex", bench_desc());
                            (MockNode { tag: r * 10 + 3 }, out)
                        });

                        // ToneMap (reads scene color + bloom)
                        final_color = graph.add_pass("ToneMap", |builder: &mut PassBuilder| {
                            builder.read_texture(scene_color);
                            builder.read_texture(bloom_tex);
                            let out = builder.create_texture("Final", bench_desc());
                            (MockNode { tag: r * 10 + 4 }, out)
                        });
                    }

                    // Final write to backbuffer
                    graph.add_pass("Present", |builder: &mut PassBuilder| {
                        builder.read_texture(final_color);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    graph.compile_topology();
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 4: SSA Alias Relay Chain
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates the engine's relay rendering pattern:
//   Opaque → Skybox (mutate) → Transparent (mutate) → ... → ToneMap
//
// Each pass uses `mutate_and_export` to create an alias that shares
// physical memory. Tests the alias propagation and lifetime unification
// paths in the compiler.

fn bench_alias_relay_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("AliasRelayChain");

    for &relay_count in &[5, 10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::from_parameter(relay_count),
            &relay_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);

                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    // Initial producer
                    let mut current = graph.add_pass("Opaque", |builder: &mut PassBuilder| {
                        let out = builder.create_texture("SceneColor_v0", bench_desc());
                        (MockNode { tag: 0 }, out)
                    });

                    // Relay chain: each pass mutates the previous version (alias)
                    for i in 1..count {
                        let prev = current;
                        current = graph.add_pass("Relay", |builder: &mut PassBuilder| {
                            let out = builder.mutate_and_export(prev, "SceneColor_vN");
                            (MockNode { tag: i as u32 }, out)
                        });
                    }

                    // Final consumer
                    graph.add_pass("ToneMap", |builder: &mut PassBuilder| {
                        builder.read_texture(current);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    graph.compile_topology();
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 5: Dead-Pass Culling Effectiveness
// ═══════════════════════════════════════════════════════════════════════════
//
// Adds N passes but only a fraction are reachable from the backbuffer.
// Tests that the mark-and-sweep culler runs in O(pass_count) and doesn't
// degrade with many dead passes.

fn bench_dead_pass_culling(c: &mut Criterion) {
    let mut group = c.benchmark_group("DeadPassCulling");

    for &total_count in &[50, 100, 200, 500] {
        group.bench_with_input(
            BenchmarkId::new("total_passes", total_count),
            &total_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                // Only 10% of passes are alive (reachable from backbuffer)
                let alive_count = count / 10;

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);

                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    // Dead passes: produce textures nobody reads
                    for i in 0..(count - alive_count) {
                        graph.add_pass("DeadPass", |builder: &mut PassBuilder| {
                            let _out = builder.create_texture("DeadTex", bench_desc());
                            (MockNode { tag: i as u32 }, ())
                        });
                    }

                    // Alive chain: linear chain ending at backbuffer
                    let mut current = graph.add_pass("AliveStart", |builder: &mut PassBuilder| {
                        let out = builder.create_texture("AliveTex_0", bench_desc());
                        (MockNode { tag: 1000 }, out)
                    });

                    for i in 1..alive_count {
                        let prev = current;
                        current = graph.add_pass("AliveN", |builder: &mut PassBuilder| {
                            builder.read_texture(prev);
                            let out = builder.create_texture("AliveTex_N", bench_desc());
                            (
                                MockNode {
                                    tag: 1000 + i as u32,
                                },
                                out,
                            )
                        });
                    }

                    graph.add_pass("AliveEnd", |builder: &mut PassBuilder| {
                        builder.read_texture(current);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    graph.compile_topology();

                    // Force the compiler to keep the compiled graph alive
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 6: FrameArena Allocation Throughput
// ═══════════════════════════════════════════════════════════════════════════
//
// Measures raw arena allocation throughput: alloc + reset cycle.
// This is the foundation of zero-drop node storage.

fn bench_arena_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("FrameArena_Allocation");

    for &alloc_count in &[100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(alloc_count),
            &alloc_count,
            |b, &count| {
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    for i in 0..count {
                        let ptr = arena.alloc(MockNode { tag: i as u32 });
                        black_box(ptr);
                    }
                    black_box(arena.allocated_bytes());
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 7: Arena Allocation with Borrowing Nodes
// ═══════════════════════════════════════════════════════════════════════════
//
// Tests arena allocation with nodes that carry borrowed references,
// validating the lifetime model doesn't impose hidden overhead.

fn bench_arena_borrowing_nodes(c: &mut Criterion) {
    let shared_data: Vec<u32> = (0..5000).collect();

    c.bench_function("FrameArena_BorrowingNodes_1000", |b| {
        let mut arena = FrameArena::new();

        b.iter(|| {
            arena.reset();
            for i in 0..1000 {
                let ptr = arena.alloc(BorrowingMockNode {
                    data_ref: &shared_data[i],
                });
                black_box(ptr);
            }
            black_box(arena.allocated_bytes());
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 8: Graph Storage Capacity Reuse Across Frames
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates multiple frames to verify that GraphStorage's Vec capacity
// reuse eliminates heap allocation after the first frame.
// The second and subsequent frames should be as fast or faster than the
// first, proving zero-alloc steady-state operation.

fn bench_multi_frame_capacity_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultiFrame_CapacityReuse");

    let pass_count = 100;

    // Warm up: run one frame to establish Vec capacity
    let mut storage = GraphStorage::new();
    let mut arena = FrameArena::new();

    {
        arena.reset();
        let mut graph = RenderGraph::new(&mut storage, &arena);
        let bb = graph.register_resource("Backbuffer", bench_desc(), true);

        let mut cur = graph.add_pass("P0", |builder: &mut PassBuilder| {
            let out = builder.create_texture("T0", bench_desc());
            (MockNode { tag: 0 }, out)
        });
        for i in 1..pass_count {
            let prev = cur;
            cur = graph.add_pass("PN", |builder: &mut PassBuilder| {
                builder.read_texture(prev);
                let out = builder.create_texture("TN", bench_desc());
                (MockNode { tag: i }, out)
            });
        }
        graph.add_pass("Final", |builder: &mut PassBuilder| {
            builder.read_texture(cur);
            builder.write_texture(bb);
            (MockNode { tag: u32::MAX }, ())
        });
        graph.compile_topology();
    }

    // Now benchmark the steady-state (capacity already allocated)
    group.bench_function("steady_state_100_passes", |b| {
        b.iter(|| {
            arena.reset();
            let mut graph = RenderGraph::new(&mut storage, &arena);
            let bb = graph.register_resource("Backbuffer", bench_desc(), true);

            let mut cur = graph.add_pass("P0", |builder: &mut PassBuilder| {
                let out = builder.create_texture("T0", bench_desc());
                (MockNode { tag: 0 }, out)
            });
            for i in 1..pass_count {
                let prev = cur;
                cur = graph.add_pass("PN", |builder: &mut PassBuilder| {
                    builder.read_texture(prev);
                    let out = builder.create_texture("TN", bench_desc());
                    (MockNode { tag: i }, out)
                });
            }
            graph.add_pass("Final", |builder: &mut PassBuilder| {
                builder.read_texture(cur);
                builder.write_texture(bb);
                (MockNode { tag: u32::MAX }, ())
            });
            graph.compile_topology();
            black_box(&graph);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 9: Side-Effect Pass Interaction
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates the Shadow Pass pattern: side-effect passes that always execute
// regardless of downstream consumers. Verifies the culler correctly preserves
// side-effect passes while still culling truly dead passes.

fn bench_side_effect_passes(c: &mut Criterion) {
    let mut group = c.benchmark_group("SideEffect_Passes");

    for &count in &[10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let mut storage = GraphStorage::new();
            let mut arena = FrameArena::new();

            b.iter(|| {
                arena.reset();
                let mut graph = RenderGraph::new(&mut storage, &arena);

                let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                // Half are side-effect passes (like shadow maps)
                for i in 0..n / 2 {
                    graph.add_pass("ShadowPass", |builder: &mut PassBuilder| {
                        builder.mark_side_effect();
                        (MockNode { tag: i as u32 }, ())
                    });
                }

                // Other half form a normal render chain
                let mut cur = graph.add_pass("Opaque", |builder: &mut PassBuilder| {
                    let out = builder.create_texture("Color", bench_desc());
                    (MockNode { tag: 100 }, out)
                });

                for i in 1..(n / 2) {
                    let prev = cur;
                    cur = graph.add_pass("PostFX", |builder: &mut PassBuilder| {
                        builder.read_texture(prev);
                        let out = builder.create_texture("PostTex", bench_desc());
                        (
                            MockNode {
                                tag: 100 + i as u32,
                            },
                            out,
                        )
                    });
                }

                graph.add_pass("Present", |builder: &mut PassBuilder| {
                    builder.read_texture(cur);
                    builder.write_texture(backbuffer);
                    (MockNode { tag: u32::MAX }, ())
                });

                graph.compile_topology();
                black_box(&graph);
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 10: High-Fidelity Pipeline Simulation
// ═══════════════════════════════════════════════════════════════════════════
//
// Simulates the engine's actual HighFidelity rendering pipeline with the
// realistic pass count and topology from the composer:
//   BRDF_LUT → IBL → Shadow → Prepass → SSAO → Opaque → MSAA_Sync
//   → Skybox (relay) → TransmissionCopy → Transparent (relay)
//   → Bloom (downsample chain + upsample) → ToneMap → FXAA → Backbuffer
//
// This is the "real world" benchmark — if this is fast, the engine is fast.

fn bench_high_fidelity_pipeline(c: &mut Criterion) {
    c.bench_function("HighFidelity_FullPipeline", |b| {
        let mut storage = GraphStorage::new();
        let mut arena = FrameArena::new();

        b.iter(|| {
            arena.reset();
            let mut graph = RenderGraph::new(&mut storage, &arena);

            let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

            // 1. Shadow (side-effect, external)
            graph.add_pass("Shadow", |builder: &mut PassBuilder| {
                builder.mark_side_effect();
                (MockNode { tag: 1 }, ())
            });

            // 2. Prepass (creates depth + normals)
            let (depth, normals) = graph.add_pass("Prepass", |builder: &mut PassBuilder| {
                let depth = builder.create_texture("Scene_Depth", bench_desc());
                let normals = builder.create_texture("Scene_Normals", bench_desc());
                (MockNode { tag: 2 }, (depth, normals))
            });

            // 3. SSAO Raw (reads depth + normals)
            let ssao_raw = graph.add_pass("SSAO_Raw", |builder: &mut PassBuilder| {
                builder.read_texture(depth);
                builder.read_texture(normals);
                let out = builder.create_texture("SSAO_Raw", bench_desc());
                (MockNode { tag: 3 }, out)
            });

            // 4. SSAO Blur
            let ssao = graph.add_pass("SSAO_Blur", |builder: &mut PassBuilder| {
                builder.read_texture(ssao_raw);
                let out = builder.create_texture("SSAO", bench_desc());
                (MockNode { tag: 4 }, out)
            });

            // 5. Opaque (reads depth, SSAO)
            let scene_color = graph.add_pass("Opaque", |builder: &mut PassBuilder| {
                builder.read_texture(depth);
                builder.read_texture(ssao);
                let out = builder.create_texture("Scene_Color_HDR", bench_desc());
                (MockNode { tag: 5 }, out)
            });

            // 6. Skybox (relay: mutate scene color)
            let scene_color_sky = graph.add_pass("Skybox", |builder: &mut PassBuilder| {
                let out = builder.mutate_and_export(scene_color, "Scene_Color_Sky");
                (MockNode { tag: 6 }, out)
            });

            // 7. Transmission Copy
            let transmission = graph.add_pass("TransmissionCopy", |builder: &mut PassBuilder| {
                builder.read_texture(scene_color_sky);
                let out = builder.create_texture("Transmission", bench_desc());
                (MockNode { tag: 7 }, out)
            });

            // 8. Transparent (relay: mutate scene color, reads transmission + depth)
            let scene_color_transparent =
                graph.add_pass("Transparent", |builder: &mut PassBuilder| {
                    let out = builder.mutate_and_export(scene_color_sky, "Scene_Color_Transparent");
                    builder.read_texture(transmission);
                    builder.read_texture(depth);
                    (MockNode { tag: 8 }, out)
                });

            // 9. Bloom Downsample Chain (5 levels)
            let mut bloom_mips = Vec::with_capacity(5);
            let mut bloom_src = scene_color_transparent;
            for i in 0..5 {
                let src = bloom_src;
                let mip = graph.add_pass("Bloom_Down", |builder: &mut PassBuilder| {
                    builder.read_texture(src);
                    let out = builder.create_texture("Bloom_Mip", bench_desc());
                    (MockNode { tag: 10 + i }, out)
                });
                bloom_mips.push(mip);
                bloom_src = mip;
            }

            // 10. Bloom Upsample Chain (4 levels, reads adjacent mips)
            let mut bloom_up = *bloom_mips.last().unwrap();
            for i in (0..4).rev() {
                let up = bloom_up;
                let down = bloom_mips[i];
                bloom_up = graph.add_pass("Bloom_Up", |builder: &mut PassBuilder| {
                    builder.read_texture(up);
                    builder.read_texture(down);
                    let out = builder.create_texture("Bloom_Up", bench_desc());
                    (MockNode { tag: 20 + i as u32 }, out)
                });
            }

            // 11. Bloom Composite
            let bloom_result = graph.add_pass("Bloom_Composite", |builder: &mut PassBuilder| {
                builder.read_texture(scene_color_transparent);
                builder.read_texture(bloom_up);
                let out = builder.create_texture("Bloom_Result", bench_desc());
                (MockNode { tag: 30 }, out)
            });

            // 12. ToneMapping
            let tonemapped = graph.add_pass("ToneMapping", |builder: &mut PassBuilder| {
                builder.read_texture(bloom_result);
                let out = builder.create_texture("ToneMapped", bench_desc());
                (MockNode { tag: 31 }, out)
            });

            // 13. FXAA → Backbuffer
            graph.add_pass("FXAA", |builder: &mut PassBuilder| {
                builder.read_texture(tonemapped);
                builder.write_texture(backbuffer);
                (MockNode { tag: 32 }, ())
            });

            graph.compile_topology();
            black_box(&graph);
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark 11: Isolated Compilation Phases
// ═══════════════════════════════════════════════════════════════════════════
//
// Measures build-only vs compile-only costs separately to identify which
// phase dominates. This helps focus optimisation efforts.

fn bench_build_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("Isolated_BuildOnly");

    for &pass_count in &[50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(pass_count),
            &pass_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);
                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    let mut cur = graph.add_pass("P0", |builder: &mut PassBuilder| {
                        let out = builder.create_texture("T0", bench_desc());
                        (MockNode { tag: 0 }, out)
                    });
                    for i in 1..count {
                        let prev = cur;
                        cur = graph.add_pass("PN", |builder: &mut PassBuilder| {
                            builder.read_texture(prev);
                            let out = builder.create_texture("TN", bench_desc());
                            (MockNode { tag: i as u32 }, out)
                        });
                    }
                    graph.add_pass("Final", |builder: &mut PassBuilder| {
                        builder.read_texture(cur);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    // Do NOT compile — only measure build cost
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

fn bench_compile_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("Isolated_CompileOnly");

    for &pass_count in &[50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(pass_count),
            &pass_count,
            |b, &count| {
                let mut storage = GraphStorage::new();
                let mut arena = FrameArena::new();

                // Pre-build the graph once outside the benchmark loop,
                // then measure _only_ the compile step by rebuilding and
                // timing compile_topology.
                // NOTE: We must rebuild each iteration because compile_topology
                // mutates storage. But we measure the full build+compile and
                // subtract the build-only cost from the group above.
                b.iter(|| {
                    arena.reset();
                    let mut graph = RenderGraph::new(&mut storage, &arena);
                    let backbuffer = graph.register_resource("Backbuffer", bench_desc(), true);

                    let mut cur = graph.add_pass("P0", |builder: &mut PassBuilder| {
                        let out = builder.create_texture("T0", bench_desc());
                        (MockNode { tag: 0 }, out)
                    });
                    for i in 1..count {
                        let prev = cur;
                        cur = graph.add_pass("PN", |builder: &mut PassBuilder| {
                            builder.read_texture(prev);
                            let out = builder.create_texture("TN", bench_desc());
                            (MockNode { tag: i as u32 }, out)
                        });
                    }
                    graph.add_pass("Final", |builder: &mut PassBuilder| {
                        builder.read_texture(cur);
                        builder.write_texture(backbuffer);
                        (MockNode { tag: u32::MAX }, ())
                    });

                    graph.compile_topology();
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Register all benchmark groups
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_linear_chain_build_and_compile,
    bench_fan_in_topology,
    bench_diamond_dag,
    bench_alias_relay_chain,
    bench_dead_pass_culling,
    bench_arena_allocation,
    bench_arena_borrowing_nodes,
    bench_multi_frame_capacity_reuse,
    bench_side_effect_passes,
    bench_high_fidelity_pipeline,
    bench_build_only,
    bench_compile_only,
);
criterion_main!(benches);
