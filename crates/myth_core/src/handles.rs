//! Core handle types shared across the engine.

use slotmap::new_key_type;

new_key_type! {
    /// Strongly-typed handle for scene nodes.
    ///
    /// A SlotMap key providing generation tracking for safe handle reuse,
    /// lightweight (8 bytes) storage, and `Copy` semantics.
    pub struct NodeHandle;

    /// Strongly-typed handle for skeleton resources.
    ///
    /// Skeletons are shared resources that can be referenced
    /// by multiple skinned mesh instances.
    pub struct SkeletonKey;
}
