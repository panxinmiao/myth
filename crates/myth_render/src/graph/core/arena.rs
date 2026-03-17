//! Per-frame linear allocator for transient render graph objects.
//!
//! All [`PassNode`](super::node::PassNode)s are allocated on the
//! [`FrameArena`] during graph construction, providing $O(1)$ allocation
//! via simple pointer bump and excellent cache locality during the execute
//! phase.
//!
//! # Frame Lifecycle
//!
//! 1. **Frame start** — [`reset()`](FrameArena::reset) reclaims all memory
//!    from the previous frame in $O(1)$.
//! 2. **Build phase** — [`alloc()`](FrameArena::alloc) places objects
//!    contiguously in pre-allocated memory chunks.
//! 3. **Execute phase** — objects are accessed linearly, maximising
//!    CPU L1/L2 cache hit rates.
//! 4. Goto 1.
//!
//! # Safety Contract
//!
//! Objects allocated on this arena **should not own heap memory** (`String`,
//! `Vec`, `Arc`, etc.).  They should hold only plain-old-data (POD) fields
//! or borrowed references (`&'a T`).
//!
//! With the `PassNode<'a>` lifetime model, all nodes carry only borrowed
//! references and trivially-copy IDs.  No `Drop` glue is required —
//! [`FrameArena::reset()`] reclaims all memory in $O(1)$ without running
//! destructors.

use bumpalo::Bump;

/// Default initial capacity for the frame arena (64 KiB).
///
/// Typical per-frame `PassNode` allocation is 2–8 KiB.  64 KiB provides
/// generous headroom and avoids any OS allocation during normal operation
/// after the first frame.
const DEFAULT_CAPACITY: usize = 64 * 1024;

/// Per-frame bump allocator for transient render graph objects.
///
/// Wraps [`bumpalo::Bump`] with a focused API surface, preventing the
/// third-party library from leaking into the engine's public interface.
///
/// # Performance Characteristics
///
/// | Operation | Cost |
/// |-----------|------|
/// | [`alloc`](Self::alloc) | $O(1)$ — pointer bump |
/// | [`reset`](Self::reset) | $O(1)$ — pointer rewind |
///
/// All allocated objects reside in contiguous virtual memory pages,
/// maximising CPU cache utilisation during the execute phase.
pub struct FrameArena {
    bump: Bump,
}

impl FrameArena {
    /// Creates a new arena with the default initial capacity (64 KiB).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::with_capacity(DEFAULT_CAPACITY),
        }
    }

    /// Creates a new arena with the specified initial capacity in bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
        }
    }

    /// Resets the allocator, reclaiming all memory in $O(1)$.
    ///
    /// **All previously returned references are invalidated.**
    ///
    /// The caller must ensure that arena-allocated objects owning heap
    /// resources have been properly cleaned up (e.g. via
    /// [`drop_in_place`](std::ptr::drop_in_place)) before this call.
    #[inline]
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Allocates an object on the arena and returns a mutable reference.
    ///
    /// Allocation is $O(1)$ — a simple pointer bump within the arena's
    /// pre-allocated memory chunk.  The returned reference is valid until
    /// the next [`reset()`](Self::reset) call.
    #[inline]
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.bump.alloc(val)
    }

    /// Allocates a slice on the arena by copying from an existing slice.
    ///
    /// Useful for arena-allocated `PassNode`s that need a contiguous slice
    /// of data (e.g. shadow light instances) without heap-allocated `Vec`.
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(src)
    }

    /// Returns the total bytes allocated in the current frame
    /// (including alignment padding).
    #[inline]
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::new()
    }
}
