use super::graph::{FrameConfig, RenderGraph};
use super::types::{TextureDesc, TextureNodeId};

/// Builder for declaring a pass's resource dependencies during eager graph
/// construction.
///
/// Obtained exclusively inside the closure passed to
/// [`RenderGraph::add_pass`].  All topology wiring — resource creation, read
/// / write declarations, and alias production — happens **immediately** and
/// is captured before the closure returns.
///
/// # Core API
///
/// | Method | Purpose |
/// |--------|---------|
/// | [`create_texture`](Self::create_texture) | Internal scratch resource (e.g. Bloom mip-chain) |
/// | [`create_and_export`](Self::create_and_export) | Create a new output resource and return it for downstream wiring |
/// | [`read_texture`](Self::read_texture) | Declare a read dependency on an existing resource |
/// | [`write_texture`](Self::write_texture) | Claim an externally-created resource as this pass's output |
/// | [`mutate_and_export`](Self::mutate_and_export) | SSA relay: read → alias → write, returning the new version |
/// | [`mark_side_effect`](Self::mark_side_effect) | Prevent dead-pass culling |
///
/// # Frame Configuration
///
/// [`frame_config`](Self::frame_config) exposes the current frame's
/// resolution and device format information so that descriptor derivation
/// stays self-contained within the closure.
pub struct PassBuilder<'a> {
    pub(crate) graph: &'a mut RenderGraph,
    pub(crate) pass_index: usize,
}

impl PassBuilder<'_> {
    // ─── Frame Configuration ─────────────────────────────────────────

    /// Returns the current frame's rendering configuration (resolution,
    /// depth format, MSAA samples, surface/HDR formats).
    #[inline]
    #[must_use]
    pub fn frame_config(&self) -> &FrameConfig {
        self.graph.frame_config()
    }

    /// Shorthand for `(config.width, config.height)`.
    #[inline]
    #[must_use]
    pub fn global_resolution(&self) -> (u32, u32) {
        let c = self.graph.frame_config();
        (c.width, c.height)
    }

    // ─── Resource Creation ───────────────────────────────────────────

    /// Creates an internal transient texture owned by this pass.
    ///
    /// Use this for scratch resources that are produced *and* consumed
    /// entirely within one pass (e.g. Bloom's mip-chain, SSAO's raw
    /// half-res intermediate).  The returned [`TextureNodeId`] is valid
    /// for the duration of the frame.
    pub fn create_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
    ) -> TextureNodeId {
        let id = self.graph.register_resource(name, desc, false);
        self.graph.passes[self.pass_index].creates.push(id);
        self.write_texture(id)
    }

    /// Creates a new transient texture and immediately exports it as an
    /// output of this pass.
    ///
    /// Semantically identical to [`create_texture`](Self::create_texture)
    /// but signals intent: the returned [`TextureNodeId`] is meant to be
    /// propagated to downstream passes via the closure's return value.
    #[inline]
    pub fn create_and_export(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
    ) -> TextureNodeId {
        self.create_texture(name, desc)
    }

    // ─── Dependency Declaration ──────────────────────────────────────

    /// Declares that this pass reads from the given texture resource.
    pub fn read_texture(&mut self, id: TextureNodeId) {
        self.graph.passes[self.pass_index].reads.push(id);
        self.graph.resources[id.0 as usize]
            .consumers
            .push(self.pass_index);
    }

    /// Declares that this pass writes to (produces) the given texture
    /// resource.
    ///
    /// Use this to claim ownership of a resource that was created outside
    /// the current closure (e.g. by the Composer or a preceding pass's
    /// return value).
    ///
    /// # Panics — Strict SSA Enforcement
    ///
    /// Panics if the resource already has a producer.  In a strict SSA
    /// render graph every logical resource can have at most **one**
    /// producer.  To modify an existing resource, use
    /// [`mutate_and_export`](Self::mutate_and_export) instead.
    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        let res = &mut self.graph.resources[id.0 as usize];

        if let Some(existing_producer) = res.producer {
            panic!(
                "SSA Violation in Pass '{}': Texture '{}' already has a producer (Pass '{}'). \
                 Use `builder.mutate_and_export()` to create a new version (alias).",
                self.graph.passes[self.pass_index].name,
                res.name,
                self.graph.passes[existing_producer].name
            );
        }

        self.graph.passes[self.pass_index].writes.push(id);
        res.producer = Some(self.pass_index);
        id
    }

    /// Performs an SSA relay: reads `input_id`, creates a new alias that
    /// shares the same physical GPU memory, and declares this pass as the
    /// alias's producer.
    ///
    /// Returns the **new** [`TextureNodeId`].  All downstream consumers
    /// must reference this new ID rather than `input_id`.
    ///
    /// # When to use
    ///
    /// Relay rendering patterns where multiple passes draw into the same
    /// physical render target in sequence (e.g. Opaque → Skybox →
    /// Transparent).  Each pass "mutates" the previous version, producing
    /// a clean DAG with no read-write ambiguity.
    #[must_use = "The returned TextureNodeId must be used for downstream wiring"]
    pub fn mutate_and_export(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.read_texture(input_id);
        let new_id = self.graph.create_alias(input_id, new_name);
        self.write_texture(new_id)
    }

    // ─── Flags ───────────────────────────────────────────────────────

    /// Marks this pass as having an externally-visible side effect.
    ///
    /// Side-effect passes are never culled by the graph compiler, even
    /// when they produce no resources consumed by downstream passes.
    pub fn mark_side_effect(&mut self) {
        self.graph.passes[self.pass_index].has_side_effect = true;
    }
}
