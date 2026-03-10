use super::graph::{FrameConfig, RenderGraph};
use super::types::TextureNodeId;

/// Builder for declaring a pass's resource dependencies.
///
/// Provides ergonomic APIs for creating, reading, and writing texture
/// resources within the graph.
///
/// # Explicit Wiring
///
/// All cross-pass resource dependencies are expressed via explicit
/// `TextureNodeId` parameters passed from `add_to_graph()` into the
/// `PassNode` struct.  There is no longer a name-based blackboard
/// lookup for mutable resources.
///
/// - [`create_texture`](Self::create_texture) — create an internal
///   transient texture (e.g. scratch buffers) owned by this pass.
/// - [`read_texture`](Self::read_texture) — declare a read dependency.
/// - [`write_texture`](Self::write_texture) — declare a write dependency.
///
/// # Frame Configuration
///
/// The builder exposes the current frame's resolution and device format
/// information via [`frame_config`](Self::frame_config), so passes can
/// derive their own `RdgTextureDesc`s in `setup()` without external push.
pub struct PassBuilder<'a> {
    pub(crate) graph: &'a mut RenderGraph,
    pub(crate) pass_index: usize,
}

impl<'a> PassBuilder<'a> {
    // ─── Frame Configuration ─────────────────────────────────────────

    /// Returns the current frame's rendering configuration (resolution,
    /// depth format, MSAA samples, surface/HDR formats).
    #[inline]
    pub fn frame_config(&self) -> &FrameConfig {
        self.graph.frame_config()
    }

    /// Shorthand for `(config.width, config.height)`.
    #[inline]
    pub fn global_resolution(&self) -> (u32, u32) {
        let c = self.graph.frame_config();
        (c.width, c.height)
    }

    // ─── Low-Level Resource API ──────────────────────────────────────

    /// Creates a new transient texture resource owned by this pass.
    pub fn create_texture(
        &mut self,
        name: &'static str,
        desc: super::types::RdgTextureDesc,
    ) -> TextureNodeId {
        let id = self.graph.register_resource(name, desc, false);
        self.graph.passes[self.pass_index].creates.push(id);
        self.declare_output(id)
    }

    /// Declares that this pass reads from the given texture resource.
    pub fn read_texture(&mut self, id: TextureNodeId) {
        self.graph.passes[self.pass_index].reads.push(id);
        self.graph.resources[id.0 as usize]
            .consumers
            .push(self.pass_index);
    }

    /// Declares that this pass writes to the given texture resource.
    ///
    /// # Panics (Strict SSA Enforcement)
    /// Panics if the resource already has a producer. In a strict SSA Render Graph,
    /// a logical resource can only be written to ONCE. 
    /// To append/modify an existing resource, use [`mutate_texture`](Self::mutate_texture) instead.
    pub fn declare_output(&mut self, id: TextureNodeId) -> TextureNodeId {
        let res = &mut self.graph.resources[id.0 as usize];

        // 🚨 检查是否已经存在生产者
        if let Some(existing_producer) = res.producer {
            panic!(
                "SSA Violation in Pass '{}': Texture '{}' already has a producer (Pass '{}'). \
                 You cannot write to an existing resource. Use `builder.mutate_texture()` \
                 to create a new version (alias) of this resource.",
                self.graph.passes[self.pass_index].name,
                res.name,
                self.graph.passes[existing_producer].name
            );
        }

        self.graph.passes[self.pass_index].writes.push(id);
        res.producer = Some(self.pass_index);
        id
    }

    /// Declares that this pass performs an in-place read-modify-write on
    /// the given texture, producing a new logical version that shares the
    /// same physical GPU memory (SSA alias).
    ///
    /// Internally this performs:
    /// 1. `read_texture(input_id)` — establishes a topological dependency
    ///    on the previous writer.
    /// 2. Registers a new resource (alias) with the same descriptor.
    /// 3. `write_texture(new_id)` — marks this pass as the producer of
    ///    the new version.
    ///
    /// Returns the [`TextureNodeId`] of the new version.  All downstream
    /// consumers must reference this new ID rather than `input_id`.
    ///
    /// # When to use
    ///
    /// Use this for relay rendering patterns where multiple passes draw
    /// into the same physical render target in sequence (e.g. Opaque →
    /// Skybox → Transparent).  Each pass "mutates" the previous version,
    /// producing a clean DAG with no read-write ambiguity.
    #[must_use = "You must store and use the new mutated TextureNodeId for downstream passes"]
    pub fn mutate_texture(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.read_texture(input_id);
        let new_id = self.graph.create_alias(input_id, new_name);
        self.declare_output(new_id)
    }

    /// Marks this pass as having an externally-visible side effect.
    ///
    /// Side-effect passes are never culled, even if they don't write to
    /// any external resource.
    pub fn mark_side_effect(&mut self) {
        self.graph.passes[self.pass_index].has_side_effect = true;
    }
}
