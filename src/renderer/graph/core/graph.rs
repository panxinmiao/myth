use crate::renderer::graph::core::allocator::TransientPool;
use crate::renderer::graph::core::types::TextureDesc;

use super::builder::PassBuilder;
use super::context::{ExecuteContext, PrepareContext};
use super::node::{PassNode, PassRecord};
use super::types::{ResourceRecord, TextureNodeId};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use wgpu::Device;

/// Per-frame rendering configuration stored in the [`RenderGraph`].
///
/// Set once per frame via [`RenderGraph::begin_frame`] and read by passes
/// during graph construction through [`PassBuilder`] accessors.  This
/// eliminates the need for passes to receive screen-size or format
/// information through push parameters — they can derive `TextureDesc`s
/// directly.
#[derive(Debug, Clone, Copy)]
pub struct FrameConfig {
    /// Framebuffer width in pixels.
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Device depth-stencil format (e.g. `Depth24PlusStencil8`).
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count (1 = disabled).
    pub msaa_samples: u32,
    /// Swap-chain surface format (e.g. `Bgra8UnormSrgb`).
    pub surface_format: wgpu::TextureFormat,
    /// HDR render-target format (e.g. `Rgba16Float`).
    pub hdr_format: wgpu::TextureFormat,
}

impl FrameConfig {
    /// Returns a `RdgTextureDesc` for a 2D render target matching the frame
    /// configuration's resolution and HDR format, with the given usage flags.
    #[must_use]
    pub fn create_render_target_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.hdr_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    #[must_use]
    pub fn create_surface_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.surface_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    #[must_use]
    pub fn create_depth_desc(&self, usage: wgpu::TextureUsages) -> TextureDesc {
        TextureDesc::new_2d(
            self.width,
            self.height,
            self.depth_format,
            usage | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }
}

/// Declarative Render Graph.
///
/// Stores pass and resource records, performs dependency analysis,
/// dead-pass culling, topological sorting, and physical resource allocation.
pub struct RenderGraph {
    /// All pass records for the current frame.
    pub passes: Vec<PassRecord>,
    /// All resource records for the current frame.
    pub resources: Vec<ResourceRecord>,
    /// Compiled execution queue (topologically sorted pass indices).
    pub execution_queue: Vec<usize>,

    /// Name-based resource registry for self-wiring passes.
    resource_registry: FxHashMap<&'static str, TextureNodeId>,

    /// Per-frame rendering configuration (resolution, formats, MSAA).
    frame_config: FrameConfig,

    // --- Compile-time scratch buffers (zero-alloc across frames) ---
    compile_stack: Vec<usize>,
    compile_in_degrees: Vec<usize>,
    compile_queue: Vec<usize>,
    compile_dependency_graph: Vec<SmallVec<[usize; 8]>>,

    #[cfg(debug_assertions)]
    prev_execution_names: Vec<&'static str>,
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
            execution_queue: Vec::new(),
            resource_registry: FxHashMap::default(),
            frame_config: FrameConfig {
                width: 1,
                height: 1,
                depth_format: wgpu::TextureFormat::Depth24PlusStencil8,
                msaa_samples: 1,
                surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
                hdr_format: wgpu::TextureFormat::Rgba16Float,
            },
            compile_stack: Vec::new(),
            compile_in_degrees: Vec::new(),
            compile_queue: Vec::new(),
            compile_dependency_graph: Vec::new(),
            #[cfg(debug_assertions)]
            prev_execution_names: Vec::new(),
        }
    }

    /// Resets per-frame state while retaining allocated capacity.
    ///
    /// Must be called once at the start of each frame, before any
    /// `register_resource` or `add_pass` calls.  The [`FrameConfig`] is
    /// stored and made available to passes via [`PassBuilder`] accessors
    /// so they can derive `RdgTextureDesc`s without external push params.
    pub fn begin_frame(&mut self, config: FrameConfig) {
        self.passes.clear();
        self.resources.clear();
        self.execution_queue.clear();
        self.resource_registry.clear();
        self.frame_config = config;
    }

    /// Returns the current frame's rendering configuration.
    #[inline]
    #[must_use]
    pub fn frame_config(&self) -> &FrameConfig {
        &self.frame_config
    }

    /// Registers a named texture resource.
    ///
    /// The name is recorded in the resource registry so that passes can
    /// look it up via [`PassBuilder::find_resource`] during setup.
    pub fn register_resource(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        is_external: bool,
    ) -> TextureNodeId {
        let id = TextureNodeId(self.resources.len() as u32);
        self.resources.push(ResourceRecord {
            name,
            desc,
            is_external,
            producer: None,
            consumers: smallvec::SmallVec::new(),
            first_use: usize::MAX,
            last_use: 0,
            physical_index: None,
            alias_of: None,
        });
        self.resource_registry.insert(name, id);
        id
    }

    /// Looks up a resource by name. Returns `None` if the name has not been
    /// registered in the current frame.
    #[inline]
    #[must_use]
    pub fn find_resource(&self, name: &str) -> Option<TextureNodeId> {
        self.resource_registry.get(name).copied()
    }

    /// Creates a versioned alias of `input_id` that shares the same physical
    /// GPU memory.
    ///
    /// This is the foundation of the SSA (Static Single Assignment) resource
    /// model: any in-place modification of a texture (e.g. Skybox drawing
    /// over Opaque output) produces a *new* logical ID while reusing the
    /// same physical allocation.  The resulting DAG has no ambiguous
    /// read-write edges, enabling cycle-free topological sorting without
    /// reliance on `add_pass` registration order.
    ///
    /// Used by `Feature::add_to_graph()` when the output ID must be known
    /// before the pass is inserted.  For self-contained passes, prefer
    /// [`PassBuilder::mutate_and_export`] which calls this internally.
    pub fn create_alias(&mut self, input_id: TextureNodeId, name: &'static str) -> TextureNodeId {
        let root_idx = self.resolve_alias_root(input_id.0 as usize);
        let root_id = TextureNodeId(root_idx as u32);

        let root_res = &self.resources[root_idx];
        let desc = root_res.desc;
        let is_external = root_res.is_external;

        let new_id = self.register_resource(name, desc, is_external);
        self.resources[new_id.0 as usize].alias_of = Some(root_id);
        new_id
    }

    /// Chases the `alias_of` chain to find the root (non-alias) resource.
    #[inline]
    fn resolve_alias_root(&self, idx: usize) -> usize {
        if let Some(root_id) = self.resources[idx].alias_of {
            root_id.0 as usize
        } else {
            idx
        }
    }

    /// Adds a pass to the graph using eager setup.
    ///
    /// The closure receives a [`PassBuilder`] and must return a tuple of
    /// `(PassNode, Out)`.  All resource creation and dependency wiring
    /// happens immediately inside the closure; the returned `Out` (usually
    /// a [`TextureNodeId`] or typed output struct) is forwarded to the
    /// caller for downstream wiring.
    ///
    /// # Type Parameters
    ///
    /// - `N` — concrete [`PassNode`] type (boxed automatically).
    /// - `Out` — caller-chosen return payload (e.g. `TextureNodeId`).
    /// - `F` — closure type.
    pub fn add_pass<N, Out, F>(&mut self, name: &'static str, setup_fn: F) -> Out
    where
        N: PassNode + 'static,
        F: FnOnce(&mut PassBuilder) -> (N, Out),
    {
        let pass_index = self.passes.len();

        // Phase 1: placeholder record (node = None) so the PassBuilder can
        // reference the correct index.
        self.passes.push(PassRecord::new_empty(name));

        // Phase 2: execute the closure — all topology wiring happens here.
        let (node, output) = {
            let mut builder = PassBuilder {
                graph: self,
                pass_index,
            };
            setup_fn(&mut builder)
        };

        // Phase 3: move owned node into the record.
        self.passes[pass_index].node = Some(Box::new(node));
        output
    }

    /// Adds a borrowed pass to the graph with eager setup.
    ///
    /// The closure receives a [`PassBuilder`] for topology wiring, but the
    /// pass node is provided externally and stored via a thin raw-pointer
    /// wrapper.  This method exists for long-lived passes (such as UI
    /// overlays) where the `PassNode`'s state spans multiple frames.
    ///
    /// # Safety Contract
    ///
    /// The caller **must** ensure that `node` remains valid (not moved,
    /// dropped, or mutably aliased) from this call until after the graph's
    /// `prepare` and `execute` phases complete.  In practice this means
    /// the reference must outlive [`FrameComposer::render()`].
    pub fn add_pass_borrowed<Out, F>(
        &mut self,
        name: &'static str,
        node: &mut dyn PassNode,
        setup_fn: F,
    ) -> Out
    where
        F: FnOnce(&mut PassBuilder) -> Out,
    {
        let pass_index = self.passes.len();
        self.passes.push(PassRecord::new_empty(name));

        let output = {
            let mut builder = PassBuilder {
                graph: self,
                pass_index,
            };
            setup_fn(&mut builder)
        };

        // SAFETY: BorrowedPass stores a raw pointer to the externally-owned
        // PassNode.  The caller guarantees the reference outlives the graph's
        // prepare→execute window (bounded by FrameComposer::render()).
        self.passes[pass_index].node = Some(Box::new(BorrowedPass(node as *mut dyn PassNode)));
        output
    }

    pub fn compile_topology(&mut self) {
        self.build_physical_dependencies();
        self.cull_dead_passes();
        self.topological_sort();

        self.compute_resource_lifetimes();
    }

    pub fn compile(&mut self, pool: &mut TransientPool, device: &Device) {
        self.compile_topology();

        self.allocate_physical_resources(pool, device);
    }

    /// Builds physical pass-to-pass dependencies from resource read/write edges.
    ///
    /// For each pass, every resource it reads creates a dependency on all
    /// passes that produce (write) that resource.  Self-loops (a pass that
    /// both reads and writes the same resource) are excluded.
    ///
    /// With the SSA alias model, every relay-write produces a *new* logical
    /// resource ID, so the dependency direction is fully determined by the
    /// graph edges — no reliance on `add_pass` registration order.
    fn build_physical_dependencies(&mut self) {
        for pass_idx in 0..self.passes.len() {
            let num_reads = self.passes[pass_idx].reads.len();
            for read_i in 0..num_reads {
                let res_id = self.passes[pass_idx].reads[read_i];

                // ✅ 直接获取唯一的生产者（如果有的话）
                if let Some(producer_idx) = self.resources[res_id.0 as usize].producer
                    && producer_idx < pass_idx
                    && !self.passes[pass_idx]
                        .physical_dependencies
                        .contains(&producer_idx)
                {
                    self.passes[pass_idx]
                        .physical_dependencies
                        .push(producer_idx);
                }
            }
        }
    }

    /// Marks passes as "alive" by back-propagating reference counts from
    /// root passes (those with side effects or external outputs).
    ///
    /// Dead passes that contribute to no external output are left with
    /// `reference_count == 0` and will be excluded from the execution queue.
    ///
    /// Uses index-based iteration to avoid cloning dependency lists.
    fn cull_dead_passes(&mut self) {
        self.compile_stack.clear();

        for (i, pass) in self.passes.iter_mut().enumerate() {
            pass.reference_count = 0;
            if pass.has_side_effect {
                self.compile_stack.push(i);
                pass.reference_count += 1;
                continue;
            }
            for write_id in &pass.writes {
                if self.resources[write_id.0 as usize].is_external {
                    self.compile_stack.push(i);
                    pass.reference_count += 1;
                    break;
                }
            }
        }

        while let Some(pass_idx) = self.compile_stack.pop() {
            let num_deps = self.passes[pass_idx].physical_dependencies.len();
            for dep_i in 0..num_deps {
                let dep_idx = self.passes[pass_idx].physical_dependencies[dep_i];
                if self.passes[dep_idx].reference_count == 0 {
                    self.compile_stack.push(dep_idx);
                }
                self.passes[dep_idx].reference_count += 1;
            }
        }
    }

    fn topological_sort(&mut self) {
        let pass_count = self.passes.len();

        // 重置缓冲区，复用内存
        self.compile_in_degrees.clear();
        self.compile_in_degrees.resize(pass_count, 0);

        self.compile_dependency_graph.clear();
        self.compile_dependency_graph
            .resize(pass_count, SmallVec::new());

        self.compile_queue.clear();

        // 统计存活节点的入度和依赖反转图
        for (i, pass) in self.passes.iter().enumerate() {
            if pass.reference_count > 0 {
                self.compile_in_degrees[i] = pass.physical_dependencies.len();

                if self.compile_in_degrees[i] == 0 {
                    self.compile_queue.push(i);
                }

                for &dep in &pass.physical_dependencies {
                    self.compile_dependency_graph[dep].push(i);
                }
            }
        }

        // Kahn's algorithm — uses a cursor instead of remove(0) to avoid O(N²).
        let mut cursor = 0;
        while cursor < self.compile_queue.len() {
            let node = self.compile_queue[cursor];
            cursor += 1;
            self.execution_queue.push(node);

            for &downstream in &self.compile_dependency_graph[node] {
                self.compile_in_degrees[downstream] -= 1;
                if self.compile_in_degrees[downstream] == 0 {
                    self.compile_queue.push(downstream);
                }
            }
        }

        let alive_count = self.passes.iter().filter(|p| p.reference_count > 0).count();
        assert_eq!(
            self.execution_queue.len(),
            alive_count,
            "Render Graph Detected Circular Dependency!"
        );

        #[cfg(debug_assertions)]
        self.debug_print_topology_changes();
    }

    fn compute_resource_lifetimes(&mut self) {
        for res in &mut self.resources {
            res.first_use = usize::MAX;
            res.last_use = 0;
        }

        for (timeline_index, &pass_idx) in self.execution_queue.iter().enumerate() {
            let pass = &self.passes[pass_idx];

            let mut touch_resource = |id: TextureNodeId| {
                let res = &mut self.resources[id.0 as usize];
                res.first_use = res.first_use.min(timeline_index);
                res.last_use = res.last_use.max(timeline_index);
            };

            for &id in &pass.reads {
                touch_resource(id);
            }
            for &id in &pass.writes {
                touch_resource(id);
            }
            for &id in &pass.creates {
                touch_resource(id);
            }
        }
    }

    /// Allocates physical GPU textures for transient resources, with
    /// alias-aware sharing.
    ///
    /// **Phase 1** — Calculate the unified lifetime for each alias group by propagating the
    /// lifetimes from aliases to their root resources.This ensures that the root resource's lifetime encompasses all of its aliases, preventing premature deallocation of shared physical resources.
    ///
    /// **Phase 2** — Propagate the root's unified lifetime back to every alias.\
    /// This ensures that the execute-phase `StoreOp` deduction sees the correct final lifetime and never discards data prematurely for intermediate versions.
    ///
    /// **Phase 3** — Request physical allocations from the pool for root resources.
    /// Aliases will be skipped since they share the same physical memory.
    ///
    /// **Phase 4** — Propagate the root's unified `last_use` back to every alias.\
    /// This ensures that the execute-phase `LoadOp` deduction sees the correct lifetime and keeps the physical memory alive for the entire duration of all aliases.    
    fn allocate_physical_resources(&mut self, pool: &mut TransientPool, device: &Device) {
        pool.begin_frame();

        // ✅ Phase 1: Before allocating physical memory, merge the lifetimes of all aliases into their root
        for i in 0..self.resources.len() {
            if self.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                let alias_first = self.resources[i].first_use;
                let alias_last = self.resources[i].last_use;
                self.resources[root_idx].first_use =
                    self.resources[root_idx].first_use.min(alias_first);
                self.resources[root_idx].last_use =
                    self.resources[root_idx].last_use.max(alias_last);
            }
        }

        // ✅ Phase 2: Propagate the root's unified lifetime back to every alias
        // (This step is crucial for accurate LoadOp / StoreOp deduction during execution)
        for i in 0..self.resources.len() {
            if self.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.resources[i].first_use = self.resources[root_idx].first_use;
                self.resources[i].last_use = self.resources[root_idx].last_use;
            }
        }

        // ✅ Phase 3: Request physical allocations from the pool for root resources. Aliases will be skipped since they share the same physical memory.
        for i in 0..self.resources.len() {
            let res = &mut self.resources[i];
            if res.is_external || res.first_use == usize::MAX || res.alias_of.is_some() {
                continue; // skip external resources, unused resources, and aliases (they share the root's physical memory)
            }
            res.physical_index = Some(pool.acquire(device, &res.desc, res.first_use, res.last_use));
        }

        // ✅ Phase 4: Aliases share the same physical memory as their root, so propagate the root's unified `last_use` back to every alias.
        // This ensures that the execute-phase `LoadOp` deduction sees the correct lifetime and keeps the physical memory alive for the entire duration of all aliases.
        for i in 0..self.resources.len() {
            if self.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.resources[i].physical_index = self.resources[root_idx].physical_index;
            }
        }
    }

    #[cfg(debug_assertions)]
    fn debug_print_topology_changes(&mut self) {
        let current_names: Vec<&'static str> = self
            .execution_queue
            .iter()
            .map(|&idx| self.passes[idx].name)
            .collect();

        if current_names != self.prev_execution_names {
            log::info!(
                "🌈 RDG Topology Changed! New Execution Order ({} passes): \n{:#?}",
                current_names.len(),
                current_names
            );

            println!("🌈 RDG Topology Changed! New Execution Order: {current_names:?}");

            let dump = self.dump_mermaid();
            println!("\n🌈 RDG Topology Mermaid Dump:\n{dump}");
            self.prev_execution_names = current_names;
        }
    }

    /// Dumps the current Render Graph topology as a Markdown Mermaid chart.
    /// This is an incredibly powerful tool for debugging SSA data flows,
    /// missing connections, and dead-pass elimination.
    #[must_use]
    pub fn dump_mermaid(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // writeln!(&mut out, "```mermaid").unwrap();

        writeln!(&mut out, "flowchart").unwrap();

        writeln!(
            &mut out,
            "    classDef alive fill:#2b3c5a,stroke:#4a6f9f,stroke-width:2px,color:#fff,rx:5,ry:5;"
        )
        .unwrap();
        writeln!(
            &mut out,
            "    classDef dead fill:#222,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5,color:#777,rx:5,ry:5;"
        )
        .unwrap();
        writeln!(
            &mut out,
            "    classDef external fill:#5a2b3c,stroke:#9f4a6f,stroke-width:2px,color:#fff;"
        )
        .unwrap();

        writeln!(&mut out, "\n    %% --- Passes (Nodes) ---").unwrap();
        for (i, pass) in self.passes.iter().enumerate() {
            // 使用圆角矩形表示 Pass
            let shape_open = "([";
            let shape_close = "])";
            let class = if pass.reference_count > 0 {
                "alive"
            } else {
                "dead"
            };

            writeln!(
                &mut out,
                "    P{}{}\"{}\"{}:::{}",
                i, shape_open, pass.name, shape_close, class
            )
            .unwrap();
        }

        writeln!(&mut out, "\n    %% --- Data Flow (Edges) ---").unwrap();
        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for &write_id in &pass.writes {
                let res = &self.resources[write_id.0 as usize];

                // 1. Draw edges to subsequent Consumers
                for &consumer_idx in &res.consumers {
                    let edge_style = if res.alias_of.is_some() {
                        "==>" // Bold line indicates an alias relationship (same physical resource) 
                    } else {
                        "-->" // Thin line indicates a normal data dependency
                    };

                    writeln!(
                        &mut out,
                        "    P{} {}|\"{}\"| P{};",
                        pass_idx, edge_style, res.name, consumer_idx
                    )
                    .unwrap();
                }

                // 2. If this is a resource with no subsequent consumers but it is an external output (e.g., Surface)
                // Draw a special endpoint to indicate the data flows out of the RDG pipeline
                if res.consumers.is_empty() && res.is_external {
                    writeln!(
                        &mut out,
                        "    OUT_{id}[/\"{name}\"/]:::external",
                        id = write_id.0,
                        name = res.name
                    )
                    .unwrap();
                    writeln!(
                        &mut out,
                        "    P{} -->|\"{}\"| OUT_{};",
                        pass_idx, res.name, write_id.0
                    )
                    .unwrap();
                }
            }
        }

        // writeln!(&mut out, "```").unwrap();

        // println!("\n🌈 RDG Topology Mermaid Dump:\n{out}");
        log::info!("\n🌈 RDG Topology Mermaid Dump:\n{out}");
        out
    }
}

// ─── BorrowedPass ──────────────────────────────────────────────────────

/// Thin raw-pointer wrapper for externally-owned `PassNode`s.
///
/// Used by [`RenderGraph::add_pass_borrowed`] to integrate long-lived
/// pass nodes (e.g. UI overlays) that cannot transfer ownership into the
/// graph.  Lifetime correctness is guaranteed by the caller's contract.
struct BorrowedPass(*mut dyn PassNode);

// SAFETY: The graph's single-threaded frame execution model ensures no
// concurrent access.  The caller guarantees the pointer is valid for the
// duration of prepare + execute.
unsafe impl Send for BorrowedPass {}
unsafe impl Sync for BorrowedPass {}

impl PassNode for BorrowedPass {
    fn name(&self) -> &'static str {
        // SAFETY: Valid during graph execution per caller contract.
        unsafe { &*self.0 }.name()
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // SAFETY: Valid during graph execution per caller contract.
        unsafe { &mut *self.0 }.prepare(ctx);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // SAFETY: Valid during graph execution per caller contract.
        unsafe { &*self.0 }.execute(ctx, encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> FrameConfig {
        FrameConfig {
            width: 1920,
            height: 1080,
            depth_format: wgpu::TextureFormat::Depth24PlusStencil8,
            msaa_samples: 1,
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            hdr_format: wgpu::TextureFormat::Rgba16Float,
        }
    }

    fn dummy_desc() -> TextureDesc {
        TextureDesc::new_2d(
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    // ─── Shared Mock Pass Type ───────────────────────────────────────

    struct MockExec;
    impl PassNode for MockExec {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[test]
    fn test_zero_alloc_graph() {
        let mut graph = RenderGraph::new();

        // Run two frames to verify begin_frame capacity reuse.
        for frame in 0..2 {
            graph.begin_frame(dummy_config());

            let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

            let scene_color = graph.add_pass("Opaque", |builder| {
                let out = builder.create_and_export("SceneColor", dummy_desc());
                (MockExec, out)
            });

            let bloom_tex = graph.add_pass("Bloom", |builder| {
                builder.read_texture(scene_color);
                let out = builder.create_and_export("BloomTex", dummy_desc());
                (MockExec, out)
            });

            graph.add_pass("ToneMapping", |builder| {
                builder.read_texture(scene_color);
                builder.read_texture(bloom_tex);
                builder.write_texture(backbuffer);
                (MockExec, ())
            });

            graph.compile_topology();

            assert_eq!(graph.execution_queue.len(), 3);
            assert_eq!(graph.passes[graph.execution_queue[0]].name, "Opaque");
            assert_eq!(graph.passes[graph.execution_queue[1]].name, "Bloom");
            assert_eq!(graph.passes[graph.execution_queue[2]].name, "ToneMapping");

            println!(
                "Frame {} executed: {:?}",
                frame,
                graph
                    .execution_queue
                    .iter()
                    .map(|&i| graph.passes[i].name)
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Verifies that transient resources with no consumers are treated as
    /// dead resources and skip physical allocation, while the producing
    /// pass itself remains alive (because it has other consumed outputs).
    #[test]
    fn test_dead_resource_culling() {
        let mut graph = RenderGraph::new();
        graph.begin_frame(dummy_config());

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        // GBuffer writes both Color and MotionVector.
        let (color, motion) = graph.add_pass("GBuffer", |builder| {
            let color = builder.create_and_export("Color", dummy_desc());
            let motion = builder.create_and_export("MotionVec", dummy_desc());
            (MockExec, (color, motion))
        });

        // ToneMap reads only Color, ignoring MotionVector.
        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        // GBuffer stays alive because Color is consumed.
        assert_eq!(graph.execution_queue.len(), 2);

        // Color has a consumer (ToneMap) → should be allocated.
        assert!(!graph.resources[color.0 as usize].consumers.is_empty());

        // MotionVector has no consumer → dead resource, skip allocation.
        assert!(graph.resources[motion.0 as usize].consumers.is_empty());
        // first_use is still valid because the producing pass is alive.
        assert_ne!(graph.resources[motion.0 as usize].first_use, usize::MAX);
    }

    /// Verifies that a self-read (write + read by the same pass) protects
    /// an internal resource from being culled.
    #[test]
    fn test_self_read_prevents_culling() {
        let mut graph = RenderGraph::new();
        graph.begin_frame(dummy_config());

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let internal = graph.add_pass("MacroNode", |builder| {
            let internal = builder.create_and_export("Internal", dummy_desc());
            builder.read_texture(internal); // self-read
            builder.write_texture(backbuffer);
            (MockExec, internal)
        });

        graph.compile_topology();

        // Internal resource has a consumer (self-read) → survives culling.
        assert!(!graph.resources[internal.0 as usize].consumers.is_empty());
    }

    /// Verifies auto-deduced resource lifetimes: first_use and last_use
    /// correctly reflect the compiled execution timeline.
    #[test]
    fn test_resource_lifetime_deduction() {
        let mut graph = RenderGraph::new();
        graph.begin_frame(dummy_config());

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Opaque", |builder| {
            let out = builder.create_and_export("Color", dummy_desc());
            (MockExec, out)
        });

        let bloom = graph.add_pass("Bloom", |builder| {
            builder.read_texture(color);
            let out = builder.create_and_export("Bloom", dummy_desc());
            (MockExec, out)
        });

        graph.add_pass("ToneMapping", |builder| {
            builder.read_texture(color);
            builder.read_texture(bloom);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        // Timeline: [0]=Opaque, [1]=Bloom, [2]=ToneMapping
        let color_res = &graph.resources[color.0 as usize];
        assert_eq!(
            color_res.first_use, 0,
            "Color first written by Opaque at timeline 0"
        );
        assert_eq!(
            color_res.last_use, 2,
            "Color last read by ToneMapping at timeline 2"
        );

        let bloom_res = &graph.resources[bloom.0 as usize];
        assert_eq!(
            bloom_res.first_use, 1,
            "Bloom first written by BloomPass at timeline 1"
        );
        assert_eq!(
            bloom_res.last_use, 2,
            "Bloom last read by ToneMapping at timeline 2"
        );
    }

    /// Verifies the SSA alias (mutate_and_export / create_alias) mechanism:
    ///
    /// - Relay passes (Opaque → Skybox → Transparent) produce unique
    ///   logical IDs sharing the same physical memory.
    /// - Topological ordering is determined purely by graph edges, not
    ///   by `add_pass` registration order.
    /// - Alias resources inherit `LoadOp::Load` semantics.
    #[test]
    fn test_ssa_alias_relay_passes() {
        let mut graph = RenderGraph::new();
        graph.begin_frame(dummy_config());

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        // 1. Opaque creates the root colour resource.
        let color_v0 = graph.add_pass("Opaque", |builder| {
            let out = builder.create_and_export("SceneColor_v0", dummy_desc());
            (MockExec, out)
        });

        // — Alias metadata checks —
        assert!(
            graph.resources[color_v0.0 as usize].alias_of.is_none(),
            "v0 is a root resource"
        );

        // 2. Skybox mutates v0 → v1 (SSA relay).
        let color_v1 = graph.add_pass("Skybox", |builder| {
            let out = builder.mutate_and_export(color_v0, "SceneColor_v1");
            (MockExec, out)
        });

        assert_eq!(
            graph.resources[color_v1.0 as usize].alias_of,
            Some(color_v0),
            "v1 aliases v0"
        );

        // 3. Transparent mutates v1 → v2.
        let color_v2 = graph.add_pass("Transparent", |builder| {
            let out = builder.mutate_and_export(color_v1, "SceneColor_v2");
            (MockExec, out)
        });

        assert_eq!(
            graph.resources[color_v2.0 as usize].alias_of,
            Some(color_v0),
            "v2 aliases v0 (root)"
        );

        // 4. ToneMap reads the final version.
        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color_v2);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        // 5. Verify topological order: Opaque → Skybox → Transparent → ToneMap.
        assert_eq!(graph.execution_queue.len(), 4);
        let names: Vec<&str> = graph
            .execution_queue
            .iter()
            .map(|&i| graph.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Opaque", "Skybox", "Transparent", "ToneMap"]);

        // 6. Verify alias chain resolves to root: v2 → v1 → v0.
        assert_eq!(
            graph.resolve_alias_root(color_v2.0 as usize),
            color_v0.0 as usize
        );
        assert_eq!(
            graph.resolve_alias_root(color_v1.0 as usize),
            color_v0.0 as usize
        );
        assert_eq!(
            graph.resolve_alias_root(color_v0.0 as usize),
            color_v0.0 as usize
        );
    }

    /// Verifies that `mutate_and_export` on the PassBuilder correctly
    /// creates an alias, reads the input, and writes the new version.
    #[test]
    fn test_mutate_and_export_api() {
        let mut graph = RenderGraph::new();
        graph.begin_frame(dummy_config());

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Writer", |builder| {
            let out = builder.create_and_export("Color", dummy_desc());
            (MockExec, out)
        });

        let mutated_id = graph.add_pass("Mutator", |builder| {
            let out = builder.mutate_and_export(color, "Color_Mutated");
            (MockExec, out)
        });

        // After Mutator, the graph has a new resource "Color_Mutated".
        let found = graph
            .find_resource("Color_Mutated")
            .expect("mutate_and_export should register the new resource");
        assert_eq!(found, mutated_id);
        assert!(graph.resources[mutated_id.0 as usize].alias_of.is_some());

        // Reader consumes the mutated version.
        graph.add_pass("Reader", |builder| {
            builder.read_texture(mutated_id);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.execution_queue.len(), 3);
        let names: Vec<&str> = graph
            .execution_queue
            .iter()
            .map(|&i| graph.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Writer", "Mutator", "Reader"]);
    }
}
