use crate::renderer::graph::rdg::allocator::RdgTransientPool;
use crate::renderer::graph::rdg::types::RdgTextureDesc;

use super::builder::PassBuilder;
use super::node::{PassNode, PassRecord};
use super::types::{ResourceRecord, TextureNodeId};
use smallvec::SmallVec;
use wgpu::Device;

#[derive(Default)]
pub struct RenderGraph {
    // 核心数据
    pub passes: Vec<PassRecord>,
    pub resources: Vec<ResourceRecord>,

    // 最终生成的执行队列
    pub execution_queue: Vec<usize>,

    // --- 编译期专用的驻留缓冲区 (彻底消灭每帧 malloc) ---
    compile_stack: Vec<usize>,
    compile_in_degrees: Vec<usize>,
    compile_queue: Vec<usize>,
    compile_dependency_graph: Vec<SmallVec<[usize; 8]>>,

    #[cfg(debug_assertions)]
    prev_execution_names: Vec<&'static str>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// 每帧开始前调用，保留所有 Vec 的 Capacity！
    pub fn begin_frame(&mut self) {
        self.passes.clear();
        self.resources.clear();
        self.execution_queue.clear();
    }

    pub fn register_resource(
        &mut self,
        name: &'static str,
        desc: RdgTextureDesc,
        is_external: bool,
    ) -> super::types::TextureNodeId {
        let id = super::types::TextureNodeId(self.resources.len() as u32);
        self.resources.push(super::types::ResourceRecord {
            name,
            desc,
            is_external,
            producers: smallvec::SmallVec::new(),
            consumers: smallvec::SmallVec::new(),
            first_use: usize::MAX,
            last_use: 0,
            physical_index: None,
        });
        id
    }

    pub fn add_pass(&mut self, node: &mut dyn PassNode) {
        let pass_index = self.passes.len();

        let name = node.name();
        let local_ptr = node as *mut dyn PassNode;

        let static_ptr: *mut (dyn PassNode + 'static) = unsafe { std::mem::transmute(local_ptr) };

        self.passes.push(PassRecord::new(name, static_ptr));

        let mut builder = PassBuilder {
            graph: self,
            pass_index,
        };

        unsafe { (*static_ptr).setup(&mut builder) };
    }

    pub fn compile_topology(&mut self) {
        self.build_physical_dependencies();
        self.cull_dead_passes();
        self.topological_sort();

        self.compute_resource_lifetimes();
    }

    pub fn compile(&mut self, pool: &mut RdgTransientPool, device: &Device) {
        self.compile_topology();

        self.allocate_physical_resources(pool, device);
    }

    fn build_physical_dependencies(&mut self) {
        for pass_idx in 0..self.passes.len() {
            let reads = self.passes[pass_idx].reads.clone();
            for res_id in reads {
                let producers = &self.resources[res_id.0 as usize].producers;
                for &producer_idx in producers {
                    // 只依赖在之前注册的 producer，打破 In-Place 读写形成的未来依赖
                    if producer_idx < pass_idx {
                        if !self.passes[pass_idx].physical_dependencies.contains(&producer_idx) {
                            self.passes[pass_idx]
                                .physical_dependencies
                                .push(producer_idx);
                        }
                    }
                }
            }
        }
    }

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
            let deps = self.passes[pass_idx].physical_dependencies.clone();
            for dep_idx in deps {
                let dep_pass = &mut self.passes[dep_idx];
                if dep_pass.reference_count == 0 {
                    self.compile_stack.push(dep_idx);
                }
                dep_pass.reference_count += 1;
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

        // Kahn's Algorithm
        while !self.compile_queue.is_empty() {
            let node = self.compile_queue.remove(0);
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

    fn allocate_physical_resources(&mut self, pool: &mut RdgTransientPool, device: &Device) {
        pool.begin_frame();

        for res in &mut self.resources {
            if res.is_external || res.first_use == usize::MAX {
                continue;
            }

            res.physical_index = Some(pool.acquire(device, &res.desc, res.first_use, res.last_use));
        }
    }


    #[cfg(debug_assertions)]
    fn debug_print_topology_changes(&mut self) {
        // 将当前帧的 execution_queue 映射为可读的 Pass 名字列表
        let current_names: Vec<&'static str> = self
            .execution_queue
            .iter()
            .map(|&idx| self.passes[idx].name)
            .collect();

        // 比较当前顺序和上一帧的顺序是否一致
        if current_names != self.prev_execution_names {
            // 如果发生了变化，打印出日志。
            log::info!(
                "🌈 RDG Topology Changed! New Execution Order ({} passes): \n{:#?}",
                current_names.len(),
                current_names
            );

            println!("🌈 RDG Topology Changed! New Execution Order: {:?}", current_names);

            // 更新缓存，以便下一帧对比
            self.prev_execution_names = current_names;
        }
    }

}

#[cfg(test)]
mod tests {
    use super::super::context::RdgExecuteContext;
    use super::super::types::RdgTextureDesc;
    use super::*;

    fn dummy_desc() -> RdgTextureDesc {
        RdgTextureDesc::new_2d(
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    }

    struct MockOpaquePass {
        out_color: TextureNodeId,
    }
    impl PassNode for MockOpaquePass {
        fn name(&self) -> &'static str {
            "Opaque"
        }
        fn setup(&mut self, builder: &mut PassBuilder) {
            builder.write_texture(self.out_color);
        }
        fn execute(&self, _ctx: &RdgExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    struct MockBloomPass {
        in_color: TextureNodeId,
        out_bloom: TextureNodeId,
    }
    impl PassNode for MockBloomPass {
        fn name(&self) -> &'static str {
            "Bloom"
        }
        fn setup(&mut self, builder: &mut PassBuilder) {
            builder.read_texture(self.in_color);
            builder.write_texture(self.out_bloom);
        }
        fn execute(&self, _ctx: &RdgExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    struct MockToneMappingPass {
        in_color: TextureNodeId,
        in_bloom: TextureNodeId,
        out_target: TextureNodeId,
    }
    impl PassNode for MockToneMappingPass {
        fn name(&self) -> &'static str {
            "ToneMapping"
        }
        fn setup(&mut self, builder: &mut PassBuilder) {
            builder.read_texture(self.in_color);
            builder.read_texture(self.in_bloom);
            builder.write_texture(self.out_target);
        }
        fn execute(&self, _ctx: &RdgExecuteContext, _encoder: &mut wgpu::CommandEncoder) {}
    }

    #[test]
    fn test_zero_alloc_graph() {
        let mut graph = RenderGraph::new();

        // 模拟跑两帧，验证 begin_frame 的复用逻辑
        for frame in 0..2 {
            graph.begin_frame();

            let scene_color = graph.register_resource("SceneColor", dummy_desc(), false);
            let bloom_tex = graph.register_resource("BloomTex", dummy_desc(), false);
            let backbuffer = graph.register_resource("Backbuffer", dummy_desc(), true);

            let mut tm_pass = MockToneMappingPass {
                in_color: scene_color,
                in_bloom: bloom_tex,
                out_target: backbuffer,
            };
            let mut bloom_pass = MockBloomPass {
                in_color: scene_color,
                out_bloom: bloom_tex,
            };
            let mut opaque_pass = MockOpaquePass {
                out_color: scene_color,
            };

            // 逆序添加，看是否能正确排序
            graph.add_pass(&mut tm_pass);
            graph.add_pass(&mut bloom_pass);
            graph.add_pass(&mut opaque_pass);

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
}
