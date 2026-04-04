use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::core::gpu::Tracked;
use crate::graph::core::allocator::TransientPool;
use crate::graph::core::arena::FrameArena;
use crate::graph::core::types::TextureDesc;

use super::builder::PassBuilder;
use super::node::{NodeSlot, PassNode, PassRecord};
use super::types::{ResourceRecord, TextureNodeId};
use smallvec::SmallVec;
use wgpu::Device;

/// Per-frame rendering configuration stored in the [`RenderGraph`].
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
    /// Device depth format (e.g. `Depth32Float`).
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

// ReadyNode is a helper struct used during topological sorting of the render graph. It represents a pass that is ready to be executed, along with its priority score for scheduling.
#[derive(Eq, PartialEq)]
struct ReadyNode {
    /// The priority score of the node, which can be based on various heuristics such as the number of downstream dependencies, resource usage, or custom user-defined metrics. Higher scores indicate higher priority for execution.
    priority: i32,
    pass_idx: usize,
}

// Rust's BinaryHeap is a max-heap, so we define the ordering such that higher priority scores come first.
impl Ord for ReadyNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for ReadyNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Declarative Render Graph — persistent capacity storage.
///
/// Holds the backing `Vec`s whose heap capacity is reused across frames.
/// This struct never appears in the public API — it is hidden inside
/// [`RendererState`] and lent to [`RenderGraph`] each frame via a
/// mutable reference.
///
/// # Why separate?
///
/// A [`RenderGraph`] has a per-frame lifetime `'a` tied to the
/// [`FrameArena`].  Storing `Vec`s directly on that view would
/// require reconstructing them every frame, wasting the capacity
/// that `Vec::clear()` preserves for free.
pub struct GraphStorage {
    /// All pass records for the current frame.
    pub passes: Vec<PassRecord>,
    /// All resource records for the current frame.
    pub resources: Vec<ResourceRecord>,
    /// Compiled execution queue (topologically sorted pass indices).
    pub execution_queue: Vec<usize>,

    // /// Name-based resource registry for self-wiring passes.
    // resource_registry: FxHashMap<&'static str, TextureNodeId>,

    // --- Compile-time scratch buffers (zero-alloc across frames) ---
    compile_stack: Vec<usize>,
    compile_in_degrees: Vec<usize>,
    compile_ready_heap: BinaryHeap<ReadyNode>,
    compile_dependency_graph: Vec<SmallVec<[usize; 8]>>,

    /// Group-name stack for logical subgraph tagging.
    #[cfg(feature = "rdg_inspector")]
    current_group_stack: Vec<&'static str>,

    #[cfg(debug_assertions)]
    prev_execution_names: Vec<&'static str>,
}

impl Default for GraphStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStorage {
    #[must_use]
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
            execution_queue: Vec::new(),
            // resource_registry: FxHashMap::default(),
            compile_stack: Vec::new(),
            compile_in_degrees: Vec::new(),
            compile_ready_heap: BinaryHeap::new(),
            compile_dependency_graph: Vec::new(),
            #[cfg(feature = "rdg_inspector")]
            current_group_stack: Vec::new(),
            #[cfg(debug_assertions)]
            prev_execution_names: Vec::new(),
        }
    }

    /// Clears all per-frame data while retaining heap capacity.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.resources.clear();
        self.execution_queue.clear();
        // self.resource_registry.clear();
        #[cfg(feature = "rdg_inspector")]
        self.current_group_stack.clear();
    }

    /// Dumps the current Render Graph topology as a Mermaid flowchart.
    #[must_use]
    pub fn dump_mermaid(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        writeln!(&mut out, "flowchart TD").unwrap();

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
            "    classDef external_out fill:#5a2b3c,stroke:#9f4a6f,stroke-width:2px,color:#fff;"
        )
        .unwrap();
        writeln!(
            &mut out,
            "    classDef external_in fill:#3c5a2b,stroke:#6f9f4a,stroke-width:2px,color:#fff;"
        )
        .unwrap();

        #[cfg(feature = "rdg_inspector")]
        {
            #[derive(Default)]
            struct GroupNode {
                name: &'static str,
                passes: Vec<usize>,
                children: Vec<GroupNode>,
            }

            impl GroupNode {
                fn insert(&mut self, path: &[&'static str], pass_idx: usize) {
                    if path.is_empty() {
                        self.passes.push(pass_idx);
                    } else {
                        let next_name = path[0];
                        if let Some(child) = self.children.iter_mut().find(|c| c.name == next_name)
                        {
                            child.insert(&path[1..], pass_idx);
                        } else {
                            let mut new_child = GroupNode {
                                name: next_name,
                                ..Default::default()
                            };
                            new_child.insert(&path[1..], pass_idx);
                            self.children.push(new_child);
                        }
                    }
                }

                fn write_mermaid(&self, out: &mut String, storage: &GraphStorage, depth: usize) {
                    use std::fmt::Write;
                    let indent = "    ".repeat(depth);

                    if !self.name.is_empty() {
                        writeln!(out, "{indent}subgraph {} [\"{}\"]", self.name, self.name)
                            .unwrap();
                        writeln!(out, "{indent}    direction TB").unwrap();
                    }

                    let inner_indent = if self.name.is_empty() {
                        indent.clone()
                    } else {
                        format!("{indent}    ")
                    };

                    for &pass_idx in &self.passes {
                        GraphStorage::write_pass_node(
                            out,
                            pass_idx,
                            &storage.passes[pass_idx],
                            &inner_indent,
                        );
                    }

                    for child in &self.children {
                        child.write_mermaid(
                            out,
                            storage,
                            if self.name.is_empty() {
                                depth
                            } else {
                                depth + 1
                            },
                        );
                    }

                    if !self.name.is_empty() {
                        writeln!(out, "{indent}end").unwrap();

                        let hash = self
                            .name
                            .bytes()
                            .fold(0usize, |acc, b| acc.wrapping_add(b as usize));

                        const STYLES: &[(&str, &str)] = &[
                            ("#3b82f614", "#3b82f6"),
                            ("#10b98114", "#10b981"),
                            ("#8b5cf614", "#8b5cf6"),
                            ("#f59e0b14", "#f59e0b"),
                            ("#ef444414", "#ef4444"),
                            ("#ec489914", "#ec4899"),
                            ("#06b6d414", "#06b6d4"),
                        ];

                        let (bg, stroke) = STYLES[hash % STYLES.len()];

                        writeln!(
                            out,
                            "{indent}style {} fill:{},stroke:{},stroke-width:2px,stroke-dasharray: 5 5,color:#fff,rx:10,ry:10",
                            self.name, bg, stroke
                        ).unwrap();
                    }
                }
            }

            let mut root = GroupNode::default();
            for (i, pass) in self.passes.iter().enumerate() {
                root.insert(&pass.groups, i);
            }

            root.write_mermaid(&mut out, self, 1);
        }

        #[cfg(not(feature = "rdg_inspector"))]
        {
            writeln!(&mut out, "\n    %% --- Passes ---").unwrap();
            for (i, pass) in self.passes.iter().enumerate() {
                Self::write_pass_node(&mut out, i, pass, "    ");
            }
        }

        writeln!(&mut out, "\n    %% --- Data Flow (Edges) ---").unwrap();

        for (i, res) in self.resources.iter().enumerate() {
            if res.is_external {
                if res.producer.is_none() {
                    writeln!(
                        &mut out,
                        "    IN_{i}[\\\"{name}\"\\]:::external_in",
                        name = res.name
                    )
                    .unwrap();
                }
                if res.consumers.is_empty() {
                    writeln!(
                        &mut out,
                        "    OUT_{i}[/\"{name}\"/]:::external_out",
                        name = res.name
                    )
                    .unwrap();
                }
            }
        }

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            // External resources (read)
            for &read_id in &pass.reads {
                let res = &self.resources[read_id.0 as usize];
                if res.is_external && res.producer.is_none() {
                    writeln!(&mut out, "    IN_{id} -.-> P{pass_idx};", id = read_id.0,).unwrap();
                }
            }

            for &write_id in &pass.writes {
                let res = &self.resources[write_id.0 as usize];

                for &consumer_idx in &res.consumers {
                    let edge_style = if res.alias_of.is_some() { "==>" } else { "-->" };
                    writeln!(
                        &mut out,
                        "    P{pass_idx} {edge_style}|\"{}\"| P{consumer_idx};",
                        res.name
                    )
                    .unwrap();
                }

                if res.consumers.is_empty() && res.is_external {
                    writeln!(&mut out, "    P{pass_idx} --> OUT_{};", write_id.0).unwrap();
                }
            }
        }

        log::info!("\n🌈 RDG Topology Mermaid Dump:\n{out}");
        out
    }

    fn write_pass_node(out: &mut String, index: usize, pass: &PassRecord, indent: &str) {
        use std::fmt::Write;
        let class = if pass.reference_count > 0 {
            "alive"
        } else {
            "dead"
        };
        writeln!(
            out,
            "{indent}P{index}([\"{name}\"]):::{class}",
            name = pass.name
        )
        .unwrap();
    }
}

/// Zero-drop marker trait for PassNodes.
trait AssertNoDrop {
    const VALID: ();
}

// Any PassNode that requires Drop will fail to compile due to the static assertion in AssertNoDrop::VALID.
impl<T> AssertNoDrop for T {
    // This constant will fail to compile if T needs Drop, which enforces that PassNodes cannot have destructors or own heap data.
    const VALID: () = assert!(
        !std::mem::needs_drop::<Self>(),
        "FATAL ERROR: PassNode MUST NOT implement Drop or contain heap allocations (like Arc, Vec, String). It must be a POD type or only hold references."
    );
}

/// Declarative Render Graph — per-frame view with lifetime `'a`.
///
/// Provides the full graph construction, compilation, and execution API.
/// Created each frame by [`FrameComposer`] from a [`GraphStorage`]
/// reference and a [`FrameArena`] reference, ensuring that:
///
/// - All `PassNode<'a>` values can borrow data with frame scope.
/// - The underlying `Vec` capacity is silently reused across frames.
/// - The `RenderGraph` is *not* `'static`, accurately reflecting its
///   single-frame semantic lifetime.
pub struct RenderGraph<'a> {
    pub(crate) storage: &'a mut GraphStorage,
    arena: &'a FrameArena,
}

impl<'a> RenderGraph<'a> {
    /// Creates a new per-frame render graph view.
    ///
    /// Clears the storage's per-frame data and records the frame config
    /// for pass-level access via [`PassBuilder::frame_config`].
    pub fn new(storage: &'a mut GraphStorage, arena: &'a FrameArena) -> Self {
        storage.clear();
        Self { storage, arena }
    }

    // ─── Logical Grouping (Inspector) ────────────────────────────────

    #[cfg(feature = "rdg_inspector")]
    #[inline]
    pub(crate) fn push_group(&mut self, name: &'static str) {
        self.storage.current_group_stack.push(name);
    }

    #[cfg(feature = "rdg_inspector")]
    #[inline]
    pub(crate) fn pop_group(&mut self) {
        self.storage.current_group_stack.pop();
    }

    /// Registers a named texture resource.
    pub fn register_resource(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        is_external: bool,
    ) -> TextureNodeId {
        let id = TextureNodeId(self.storage.resources.len() as u32);
        self.storage.resources.push(ResourceRecord {
            name,
            desc,
            is_external,
            producer: None,
            consumers: smallvec::SmallVec::new(),
            first_use: usize::MAX,
            last_use: 0,
            physical_index: None,
            external_view_ptr: None,
            alias_of: None,
        });
        id
    }

    pub fn import_external_resource(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        let id = self.register_resource(name, desc, true);
        self.storage.resources[id.0 as usize].external_view_ptr = Some(std::ptr::from_ref(view));
        id
    }

    /// Creates a versioned alias of `input_id` that shares the same physical
    /// GPU memory.
    pub fn create_alias(&mut self, input_id: TextureNodeId, name: &'static str) -> TextureNodeId {
        let root_idx = self.resolve_alias_root(input_id.0 as usize);
        let root_id = TextureNodeId(root_idx as u32);

        let root_res = &self.storage.resources[root_idx];
        let desc = root_res.desc;
        let is_external = root_res.is_external;

        let new_id = self.register_resource(name, desc, is_external);
        self.storage.resources[new_id.0 as usize].alias_of = Some(root_id);
        new_id
    }

    /// Chases the `alias_of` chain to find the root (non-alias) resource.
    #[inline]
    fn resolve_alias_root(&self, idx: usize) -> usize {
        if let Some(root_id) = self.storage.resources[idx].alias_of {
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
    /// # Arena Allocation
    ///
    /// The returned `PassNode` is allocated on the current frame's
    /// [`FrameArena`] — a simple $O(1)$ pointer bump with no system
    /// allocator overhead.  All nodes reside in contiguous memory,
    /// maximising CPU cache utilisation during the execute phase.
    ///
    /// # Zero Drop
    ///
    /// Arena-allocated nodes are **never** dropped.  [`FrameArena::reset()`]
    /// reclaims all memory in O(1) without running destructors.  Nodes
    /// must therefore hold only borrowed references (`&'a T`) or `Copy`
    /// types — never owned heap data.
    pub fn add_pass<N, Out, F>(&mut self, name: &'static str, setup_fn: F) -> Out
    where
        N: PassNode<'a> + 'a,
        F: FnOnce(&mut PassBuilder<'_, 'a>) -> (N, Out),
    {
        // Static assertion to enforce that N does not implement Drop and does not contain heap data.
        // This ensures that PassNodes are safe to allocate on the FrameArena without needing to run destructors.
        let () = <N as AssertNoDrop>::VALID;

        let pass_index = self.storage.passes.len();

        // Phase 1: placeholder record (node = None) so the PassBuilder can
        // reference the correct index.
        self.storage.passes.push(PassRecord::new_empty(name));

        #[cfg(feature = "rdg_inspector")]
        {
            self.storage.passes[pass_index].groups =
                self.storage.current_group_stack.iter().copied().collect();
        }

        // Phase 2: execute the closure — all topology wiring happens here.
        let (node, output) = {
            let mut builder = PassBuilder {
                graph: self,
                pass_index,
            };
            setup_fn(&mut builder)
        };

        // Phase 3: allocate the node on the frame arena (O(1) pointer bump).
        // let node_ref: &mut N = self.arena.alloc(node);
        let node_ref: &'a mut N = self.arena.alloc(node);
        let ptr = node_ref as *mut N;
        self.storage.passes[pass_index].node = Some(NodeSlot::new(ptr));
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

    fn build_physical_dependencies(&mut self) {
        for pass_idx in 0..self.storage.passes.len() {
            let num_reads = self.storage.passes[pass_idx].reads.len();
            for read_i in 0..num_reads {
                let res_id = self.storage.passes[pass_idx].reads[read_i];

                if let Some(producer_idx) = self.storage.resources[res_id.0 as usize].producer
                    && producer_idx < pass_idx
                    && !self.storage.passes[pass_idx]
                        .physical_dependencies
                        .contains(&producer_idx)
                {
                    self.storage.passes[pass_idx]
                        .physical_dependencies
                        .push(producer_idx);
                }
            }
        }
    }

    fn cull_dead_passes(&mut self) {
        self.storage.compile_stack.clear();

        for (i, pass) in self.storage.passes.iter_mut().enumerate() {
            pass.reference_count = 0;
            if pass.has_side_effect {
                self.storage.compile_stack.push(i);
                pass.reference_count += 1;
                continue;
            }
            for write_id in &pass.writes {
                if self.storage.resources[write_id.0 as usize].is_external {
                    self.storage.compile_stack.push(i);
                    pass.reference_count += 1;
                    break;
                }
            }
        }

        while let Some(pass_idx) = self.storage.compile_stack.pop() {
            let num_deps = self.storage.passes[pass_idx].physical_dependencies.len();
            for dep_i in 0..num_deps {
                let dep_idx = self.storage.passes[pass_idx].physical_dependencies[dep_i];
                if self.storage.passes[dep_idx].reference_count == 0 {
                    self.storage.compile_stack.push(dep_idx);
                }
                self.storage.passes[dep_idx].reference_count += 1;
            }
        }
    }

    fn topological_sort(&mut self) {
        let pass_count = self.storage.passes.len();

        self.storage.compile_in_degrees.clear();
        self.storage.compile_in_degrees.resize(pass_count, 0);

        self.storage.compile_dependency_graph.clear();
        self.storage
            .compile_dependency_graph
            .resize(pass_count, SmallVec::new());

        self.storage.compile_ready_heap.clear();

        for (i, pass) in self.storage.passes.iter().enumerate() {
            if pass.reference_count > 0 {
                self.storage.compile_in_degrees[i] = pass.physical_dependencies.len();

                if self.storage.compile_in_degrees[i] == 0 {
                    self.storage.compile_ready_heap.push(ReadyNode {
                        priority: 0,
                        pass_idx: i,
                    });
                }

                for &dep in &pass.physical_dependencies {
                    self.storage.compile_dependency_graph[dep].push(i);
                }
            }
        }

        let mut sequence_counter = 0;

        while let Some(ReadyNode { pass_idx: node, .. }) = self.storage.compile_ready_heap.pop() {
            self.storage.execution_queue.push(node);
            sequence_counter += 1;

            for &downstream in &self.storage.compile_dependency_graph[node] {
                self.storage.compile_in_degrees[downstream] -= 1;
                if self.storage.compile_in_degrees[downstream] == 0 {
                    let heuristic_score = sequence_counter;

                    self.storage.compile_ready_heap.push(ReadyNode {
                        priority: heuristic_score,
                        pass_idx: downstream,
                    });
                }
            }
        }

        let alive_count = self
            .storage
            .passes
            .iter()
            .filter(|p| p.reference_count > 0)
            .count();
        assert_eq!(
            self.storage.execution_queue.len(),
            alive_count,
            "Render Graph Detected Circular Dependency!"
        );

        #[cfg(debug_assertions)]
        self.debug_print_topology_changes();
    }

    fn compute_resource_lifetimes(&mut self) {
        for res in &mut self.storage.resources {
            res.first_use = usize::MAX;
            res.last_use = 0;
        }

        for (timeline_index, &pass_idx) in self.storage.execution_queue.iter().enumerate() {
            let pass = &self.storage.passes[pass_idx];

            let mut touch_resource = |id: TextureNodeId| {
                let res = &mut self.storage.resources[id.0 as usize];
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

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                let alias_first = self.storage.resources[i].first_use;
                let alias_last = self.storage.resources[i].last_use;
                self.storage.resources[root_idx].first_use =
                    self.storage.resources[root_idx].first_use.min(alias_first);
                self.storage.resources[root_idx].last_use =
                    self.storage.resources[root_idx].last_use.max(alias_last);
            }
        }

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.storage.resources[i].first_use = self.storage.resources[root_idx].first_use;
                self.storage.resources[i].last_use = self.storage.resources[root_idx].last_use;
            }
        }

        for i in 0..self.storage.resources.len() {
            let res = &mut self.storage.resources[i];
            if res.is_external || res.first_use == usize::MAX || res.alias_of.is_some() {
                continue;
            }
            res.physical_index = Some(pool.acquire(device, &res.desc, res.first_use, res.last_use));
        }

        for i in 0..self.storage.resources.len() {
            if self.storage.resources[i].alias_of.is_some() {
                let root_idx = self.resolve_alias_root(i);
                self.storage.resources[i].physical_index =
                    self.storage.resources[root_idx].physical_index;
            }
        }
    }

    #[cfg(debug_assertions)]
    fn debug_print_topology_changes(&mut self) {
        let current_names: Vec<&'static str> = self
            .storage
            .execution_queue
            .iter()
            .map(|&idx| self.storage.passes[idx].name)
            .collect();

        if current_names != self.storage.prev_execution_names {
            log::info!(
                "🌈 RDG Topology Changed! New Execution Order ({} passes): \n{:#?}",
                current_names.len(),
                current_names
            );

            log::info!("🌈 RDG Topology Changed! New Execution Order: {current_names:?}");

            let dump = self.dump_mermaid();
            log::info!("\n🌈 RDG Topology Mermaid Dump:\n{dump}");
            self.storage.prev_execution_names = current_names;
        }
    }

    /// Dumps the current Render Graph topology as a Mermaid flowchart.
    ///
    /// When the `rdg_inspector` feature is enabled, passes that share the
    /// same [`with_group`](Self::with_group) scope are wrapped in Mermaid
    /// `subgraph` blocks for high-level readability.  Without the feature
    /// the output is a flat graph (still fully functional for debugging).
    #[must_use]
    pub fn dump_mermaid(&self) -> String {
        self.storage.dump_mermaid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{composer::GraphBuilderContext, core::context::ExecuteContext};

    fn dummy_config() -> FrameConfig {
        FrameConfig {
            width: 1920,
            height: 1080,
            depth_format: wgpu::TextureFormat::Depth32Float,
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

    fn dummy_pipeline_cache() -> crate::pipeline::PipelineCache {
        crate::pipeline::PipelineCache::new()
    }

    // ─── Shared Mock Pass Type ───────────────────────────────────────

    struct MockExec;
    impl<'a> PassNode<'a> for MockExec {
        fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    /// Helper: create a fresh RenderGraph for a test frame.
    fn begin_test_frame<'a>(
        storage: &'a mut GraphStorage,
        arena: &'a FrameArena,
    ) -> RenderGraph<'a> {
        storage.clear();
        RenderGraph::new(storage, arena)
    }

    #[test]
    fn test_zero_alloc_graph() {
        let mut storage = GraphStorage::new();
        let mut arena = FrameArena::new();

        // Run two frames to verify capacity reuse.
        for frame in 0..2 {
            arena.reset();
            let mut graph = begin_test_frame(&mut storage, &arena);

            let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

            let scene_color = graph.add_pass("Opaque", |builder| {
                let out = builder.create_texture("SceneColor", dummy_desc());
                (MockExec, out)
            });

            let bloom_tex = graph.add_pass("Bloom", |builder| {
                builder.read_texture(scene_color);
                let out = builder.create_texture("BloomTex", dummy_desc());
                (MockExec, out)
            });

            graph.add_pass("ToneMapping", |builder| {
                builder.read_texture(scene_color);
                builder.read_texture(bloom_tex);
                builder.write_texture(backbuffer);
                (MockExec, ())
            });

            graph.compile_topology();

            assert_eq!(graph.storage.execution_queue.len(), 3);
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[0]].name,
                "Opaque"
            );
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[1]].name,
                "Bloom"
            );
            assert_eq!(
                graph.storage.passes[graph.storage.execution_queue[2]].name,
                "ToneMapping"
            );

            println!(
                "Frame {} executed: {:?}",
                frame,
                graph
                    .storage
                    .execution_queue
                    .iter()
                    .map(|&i| graph.storage.passes[i].name)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_dead_resource_culling() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let (color, motion) = graph.add_pass("GBuffer", |builder| {
            let color = builder.create_texture("Color", dummy_desc());
            let motion = builder.create_texture("MotionVec", dummy_desc());
            (MockExec, (color, motion))
        });

        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 2);
        assert!(
            !graph.storage.resources[color.0 as usize]
                .consumers
                .is_empty()
        );
        assert!(
            graph.storage.resources[motion.0 as usize]
                .consumers
                .is_empty()
        );
        assert_ne!(
            graph.storage.resources[motion.0 as usize].first_use,
            usize::MAX
        );
    }

    #[test]
    fn test_self_read_prevents_culling() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let internal = graph.add_pass("MacroNode", |builder| {
            let internal = builder.create_texture("Internal", dummy_desc());
            builder.read_texture(internal);
            builder.write_texture(backbuffer);
            (MockExec, internal)
        });

        graph.compile_topology();
        assert!(
            !graph.storage.resources[internal.0 as usize]
                .consumers
                .is_empty()
        );
    }

    #[test]
    fn test_resource_lifetime_deduction() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Opaque", |builder| {
            let out = builder.create_texture("Color", dummy_desc());
            (MockExec, out)
        });

        let bloom = graph.add_pass("Bloom", |builder| {
            builder.read_texture(color);
            let out = builder.create_texture("Bloom", dummy_desc());
            (MockExec, out)
        });

        graph.add_pass("ToneMapping", |builder| {
            builder.read_texture(color);
            builder.read_texture(bloom);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        let color_res = &graph.storage.resources[color.0 as usize];
        assert_eq!(
            color_res.first_use, 0,
            "Color first written by Opaque at timeline 0"
        );
        assert_eq!(
            color_res.last_use, 2,
            "Color last read by ToneMapping at timeline 2"
        );

        let bloom_res = &graph.storage.resources[bloom.0 as usize];
        assert_eq!(
            bloom_res.first_use, 1,
            "Bloom first written by BloomPass at timeline 1"
        );
        assert_eq!(
            bloom_res.last_use, 2,
            "Bloom last read by ToneMapping at timeline 2"
        );
    }

    #[test]
    fn test_ssa_alias_relay_passes() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let color_v0 = graph.add_pass("Opaque", |builder| {
            let out = builder.create_texture("SceneColor_v0", dummy_desc());
            (MockExec, out)
        });

        assert!(
            graph.storage.resources[color_v0.0 as usize]
                .alias_of
                .is_none(),
            "v0 is a root resource"
        );

        let color_v1 = graph.add_pass("Skybox", |builder| {
            let out = builder.mutate_texture(color_v0, "SceneColor_v1");
            (MockExec, out)
        });

        assert_eq!(
            graph.storage.resources[color_v1.0 as usize].alias_of,
            Some(color_v0),
            "v1 aliases v0"
        );

        let color_v2 = graph.add_pass("Transparent", |builder| {
            let out = builder.mutate_texture(color_v1, "SceneColor_v2");
            (MockExec, out)
        });

        assert_eq!(
            graph.storage.resources[color_v2.0 as usize].alias_of,
            Some(color_v0),
            "v2 aliases v0 (root)"
        );

        graph.add_pass("ToneMap", |builder| {
            builder.read_texture(color_v2);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 4);
        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Opaque", "Skybox", "Transparent", "ToneMap"]);

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

    #[test]
    fn test_mutate_texture_api() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let color = graph.add_pass("Writer", |builder| {
            let out = builder.create_texture("Color", dummy_desc());
            (MockExec, out)
        });

        let mutated_id = graph.add_pass("Mutator", |builder| {
            let out = builder.mutate_texture(color, "Color_Mutated");
            (MockExec, out)
        });

        assert!(
            graph.storage.resources[mutated_id.0 as usize]
                .alias_of
                .is_some()
        );

        graph.add_pass("Reader", |builder| {
            builder.read_texture(mutated_id);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        assert_eq!(graph.storage.execution_queue.len(), 3);
        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Writer", "Mutator", "Reader"]);
    }

    #[test]
    fn test_with_group_preserves_topology() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let mut ctx = GraphBuilderContext {
            graph: &mut graph,
            pipeline_cache: &dummy_pipeline_cache(),
            frame_config: &dummy_config(),
        };

        let scene_color = ctx.with_group("Scene", |ctx| {
            let opaque_out = ctx.graph.add_pass("Opaque", |builder| {
                let out = builder.create_texture("SceneColor", dummy_desc());
                (MockExec, out)
            });

            let skybox_out = ctx.graph.add_pass("Skybox", |builder| {
                let out = builder.mutate_texture(opaque_out, "SceneColor_Sky");
                (MockExec, out)
            });

            skybox_out
        });

        ctx.with_group("PostProcess", |ctx| {
            ctx.graph.add_pass("ToneMap", |builder| {
                builder.read_texture(scene_color);
                builder.write_texture(backbuffer);
                (MockExec, ())
            });
        });

        graph.compile_topology();

        let names: Vec<&str> = graph
            .storage
            .execution_queue
            .iter()
            .map(|&i| graph.storage.passes[i].name)
            .collect();
        assert_eq!(names, vec!["Opaque", "Skybox", "ToneMap"]);
    }

    #[cfg(feature = "rdg_inspector")]
    #[test]
    fn test_dump_mermaid_subgraphs() {
        let mut storage = GraphStorage::new();
        let arena = FrameArena::new();
        let mut graph = begin_test_frame(&mut storage, &arena);

        let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

        let mut ctx = GraphBuilderContext {
            graph: &mut graph,
            pipeline_cache: &dummy_pipeline_cache(),
            frame_config: &dummy_config(),
        };

        let bloom_out = ctx.with_group("Bloom_System", |ctx| {
            let extract_out = ctx.graph.add_pass("Bloom_Extract", |builder| {
                let out = builder.create_texture("Bloom_Mip0", dummy_desc());
                (MockExec, out)
            });

            ctx.graph.add_pass("Bloom_Composite", |builder| {
                builder.read_texture(extract_out);
                let out = builder.create_texture("Bloom_Final", dummy_desc());
                (MockExec, out)
            })
        });

        ctx.graph.add_pass("ToneMap", |builder| {
            builder.read_texture(bloom_out);
            builder.write_texture(backbuffer);
            (MockExec, ())
        });

        graph.compile_topology();

        let mermaid = graph.dump_mermaid();

        assert!(
            mermaid.contains("subgraph Bloom_System"),
            "Mermaid output should contain the Bloom_System subgraph"
        );
        assert!(
            mermaid.contains("Bloom_Extract"),
            "Subgraph should contain Bloom_Extract pass"
        );
        assert!(
            mermaid.contains("Bloom_Composite"),
            "Subgraph should contain Bloom_Composite pass"
        );
        assert!(
            mermaid.contains("ToneMap"),
            "Mermaid should have an ungrouped section for ToneMap"
        );
        assert_eq!(graph.storage.passes[0].groups.as_slice(), &["Bloom_System"]);
        assert_eq!(graph.storage.passes[1].groups.as_slice(), &["Bloom_System"]);
        assert!(graph.storage.passes[2].groups.is_empty());
    }
}
