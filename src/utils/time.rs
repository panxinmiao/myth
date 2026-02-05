#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

/// Timer for tracking frame timing and elapsed time.
pub struct Timer {
    start_time: Instant,
    last_update: Instant,
    /// Time since last tick
    pub delta: Duration,
    /// Total elapsed time since creation
    pub elapsed: Duration,
    /// Total number of ticks
    pub frame_count: u64,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    /// Creates a new timer starting from now.
    #[must_use]
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_update: now,
            delta: Duration::ZERO,
            elapsed: Duration::ZERO,
            frame_count: 0,
        }
    }

    /// Updates the timer (called internally by the engine each frame).
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = now - self.last_update;
        self.elapsed = now - self.start_time;
        self.last_update = now;
        self.frame_count += 1;
    }

    #[must_use]
    pub fn dt_seconds(&self) -> f32 {
        self.delta.as_secs_f32()
    }
}
