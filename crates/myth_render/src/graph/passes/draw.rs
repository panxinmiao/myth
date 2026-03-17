//! Unified Draw Command Submission
//!
//! Provides [`submit_draw_commands`], the single function responsible for
//! translating a slice of pre-baked [`DrawCommand`]s into GPU render-pass
//! calls.  All scene-drawing passes (opaque, transparent, shadow, prepass,
//! simple-forward) invoke this function, ensuring consistent state management
//! and eliminating duplicated draw-loop boilerplate.
//!
//! # Zero-Cost State Tracking
//!
//! Since [`DrawCommand`] stores physical `&wgpu::RenderPipeline` and
//! `&wgpu::BindGroup` references (which are `Arc`-backed, heap-stable),
//! the raw pointer address is a perfect identity key.  Comparing two
//! `*const T` is a single 64-bit integer comparison — true zero-cost.
//!
//! All tracking state lives in local variables (registers), eliminating
//! the need for a wrapper struct.

use crate::graph::frame::DrawCommand;

/// Submit a batch of pre-baked [`DrawCommand`]s through a raw render pass.
///
/// # State Tracking via Pointer Identity
///
/// `wgpu` objects (`RenderPipeline`, `BindGroup`, `Buffer`) are
/// `Arc`-backed with stable heap addresses.  Two references to the same
/// logical object always share the same pointer, making `std::ptr::eq`
/// the fastest possible identity check (< 1 ns, single CPU compare).
///
/// # Bind Group Convention
///
/// * **Group 0** (frame / camera / lights) — set by the caller before this
///   function, and left untouched.
/// * **Group 1** (material) — set per-command from [`DrawCommand::bind_group_1`].
/// * **Group 2** (object / transform) — set per-command from
///   [`DrawCommand::bind_group_2`] with its dynamic offset.
/// * **Group 3** (screen / transient) — set per-command from
///   [`DrawCommand::bind_group_3`] when present.
pub fn submit_draw_commands<'pass, 'cmd: 'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    commands: &'pass [DrawCommand<'cmd>],
) {
    if commands.is_empty() {
        return;
    }

    // ── Bare state registers — zero allocation, zero indirection ────
    let mut cur_pipeline: *const wgpu::RenderPipeline = std::ptr::null();
    let mut cur_bg1: *const wgpu::BindGroup = std::ptr::null();
    let mut cur_bg2: *const wgpu::BindGroup = std::ptr::null();
    let mut cur_bg3: *const wgpu::BindGroup = std::ptr::null();
    let mut cur_vertex_b: [*const wgpu::Buffer; 8] = [std::ptr::null(); 8];
    let mut cur_index_b: *const wgpu::Buffer = std::ptr::null();
    let mut cur_stencil: Option<u32> = None;

    for cmd in commands {
        // ── Pipeline ────────────────────────────────────────────────
        let pp = cmd.pipeline as *const wgpu::RenderPipeline;
        if pp != cur_pipeline {
            pass.set_pipeline(cmd.pipeline);
            cur_pipeline = pp;
        }

        // ── Bind Group 1: Material ──────────────────────────────────
        if let Some(bg1) = cmd.bind_group_1 {
            let p = bg1 as *const wgpu::BindGroup;
            if p != cur_bg1 {
                pass.set_bind_group(1, bg1, &[]);
                cur_bg1 = p;
            }
        }

        // ── Bind Group 2: Object / Transform ───────────────────────
        {
            let (bg2, offset) = &cmd.bind_group_2;
            let p = *bg2 as *const wgpu::BindGroup;
            // Always re-set when pointer OR dynamic offset may differ.
            // Object bind groups are often shared across instances with
            // different dynamic offsets, so we unconditionally set here.
            if p != cur_bg2 {
                cur_bg2 = p;
            }
            pass.set_bind_group(2, *bg2, &[*offset]);
        }

        // ── Bind Group 3: Screen / Transient ────────────────────────
        if let Some(bg3) = cmd.bind_group_3 {
            let p = bg3 as *const wgpu::BindGroup;
            if p != cur_bg3 {
                pass.set_bind_group(3, bg3, &[]);
                cur_bg3 = p;
            }
        }

        // ── Stencil Reference ───────────────────────────────────────
        if cmd.stencil_reference != cur_stencil {
            if let Some(val) = cmd.stencil_reference {
                pass.set_stencil_reference(val);
            }
            cur_stencil = cmd.stencil_reference;
        }

        // ── Vertex Buffers ──────────────────────────────────────────
        for (slot, buf) in cmd.vertex_buffers.iter().enumerate() {
            let p = *buf as *const wgpu::Buffer;
            if cur_vertex_b[slot] != p {
                pass.set_vertex_buffer(slot as u32, buf.slice(..));
                cur_vertex_b[slot] = p;
            }
        }

        // ── Index Buffer + Draw ─────────────────────────────────────
        if let Some((buf, fmt, count)) = cmd.index_buffer {
            let p = buf as *const wgpu::Buffer;
            if p != cur_index_b {
                pass.set_index_buffer(buf.slice(..), fmt);
                cur_index_b = p;
            }
            pass.draw_indexed(0..count, 0, cmd.instance_range.clone());
        } else {
            pass.draw(cmd.vertex_range.clone(), cmd.instance_range.clone());
        }
    }
}
