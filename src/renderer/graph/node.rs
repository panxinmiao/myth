//! Render Node Trait
//!
//! Defines the abstract interface for nodes in the render graph.
//! Each node represents a render pass or a compute task.

use super::context::{ExecuteContext, PrepareContext};

/// Render Node Trait
///
/// All render passes must implement this interface.
///
/// # Design Principles
/// - `prepare` receives a `PrepareContext` (mutable) for resource allocation and pipeline creation
/// - `run` receives an `ExecuteContext` (read-only) + `CommandEncoder` for recording GPU commands
/// - Nodes should complete all mutable operations in `prepare`; `run` should be read-only
///
/// # Performance Considerations
/// - Avoid memory allocations in `run`
/// - Use `encoder.push_debug_group` for GPU profiling
///
pub trait RenderNode {
    /// Returns the node name, used for debugging and profiling.
    fn name(&self) -> &str;

    /// Prepare phase: allocate resources, compile pipelines, and build BindGroups.
    ///
    /// Has mutable access to engine subsystems.
    fn prepare(&mut self, _ctx: &mut PrepareContext) {}

    /// Execute phase: record GPU rendering commands.
    ///
    /// # Arguments
    /// - `ctx`: Read-only execution context containing all shared resources
    /// - `encoder`: GPU command encoder
    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder);
}
