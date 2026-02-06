//! Utility Module
//!
//! This module provides various utility functions and types:
//!
//! - [`OrbitControls`]: Camera orbit controller for interactive viewing
//! - [`FpsCounter`]: Frame rate measurement utility
//! - [`interner`]: String interning for efficient symbol storage
//! - [`time`]: Time-related utilities
//!
//! # String Interning
//!
//! The interner module provides efficient string storage for frequently
//! used identifiers like shader macro names. Interned strings (Symbols)
//! can be compared in O(1) time.
//!
//! ```rust,ignore
//! use myth::utils::interner;
//!
//! let sym1 = interner::intern("HAS_NORMAL_MAP");
//! let sym2 = interner::intern("HAS_NORMAL_MAP");
//! assert_eq!(sym1, sym2); // O(1) comparison
//! ```

pub mod fps_counter;
pub mod interner;
pub mod orbit_control;
pub mod time;

pub use fps_counter::FpsCounter;
pub use interner::Symbol;
pub use orbit_control::OrbitControls;
