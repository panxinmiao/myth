// src/animation/track.rs
use crate::animation::values::Interpolatable;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    Linear,
    Step,
    CubicSpline,
}

const MAX_SCAN_OFFSET: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct KeyframeCursor {
    pub last_index: usize,
}

#[derive(Debug, Clone)]
pub struct KeyframeTrack<T: Interpolatable> {
    pub times: Vec<f32>,
    pub values: Vec<T>, // For CubicSpline, length is times.len() * 3
    pub interpolation: InterpolationMode,
}

impl<T: Interpolatable> KeyframeTrack<T> {
    #[must_use]
    pub fn new(times: Vec<f32>, values: Vec<T>, interpolation: InterpolationMode) -> Self {
        Self {
            times,
            values,
            interpolation,
        }
    }

    #[must_use]
    pub fn sample(&self, time: f32) -> T {
        // Should ideally return Default or panic, depending on error handling strategy.
        // For simplicity, assume non-empty here.
        assert!(!self.times.is_empty(), "Track is empty");

        // 1. Find the keyframe
        // partition_point finds the first index where t > time, i.e. next_index
        let next_idx = self.times.partition_point(|&t| t <= time);

        self.sample_at_frame(next_idx, time)
    }

    /// Core optimization: sampling with cursor.
    /// cursor: mutable reference that will be updated.
    pub fn sample_with_cursor(&self, time: f32, cursor: &mut KeyframeCursor) -> T {
        if self.times.is_empty() {
            // Simple empty data handling; in production, may need to return Option or Default
            if let Some(val) = self.values.first() {
                return val.clone();
            }
            panic!("Track is empty"); // or return a default value
        }

        let len = self.times.len();
        // Fast path: static data (single keyframe)
        if len == 1 {
            return self.get_value_at(0).clone();
        }

        let i = cursor.last_index;

        // === O(1) optimization logic resurrected here ===

        // Get the time at the current cursor position
        // Safety check: if cursor is out of bounds (e.g. clip was switched), reset to 0
        let t_curr = *self.times.get(i).unwrap_or(&self.times[0]);

        // Decision: search forward or backward?
        let found_index = if time >= t_curr {
            // === Case A: Normal playback or fast-forward (time increasing) ===
            // Try a forward linear scan up to MAX_SCAN_OFFSET steps
            let mut res = None;
            // Starting from position i, check up to i + MAX_SCAN_OFFSET
            // i.e. check intervals: [i, i+1), [i+1, i+2)...
            for offset in 0..=MAX_SCAN_OFFSET {
                let idx = i + offset;
                // Boundary check: if this is the last frame and time >= last_time, clamp to last frame
                if idx >= len - 1 {
                    if time >= self.times[len - 1] {
                        res = Some(len - 1); // Clamp to end
                    }
                    break;
                }

                // Check interval [times[idx], times[idx+1])
                // We know time >= t_curr (i.e. times[i]), so only check the right boundary
                if time < self.times[idx + 1] {
                    res = Some(idx);
                    break;
                }
            }
            res
        } else {
            // === Case B: Reverse playback or loop reset (time decreasing) ===
            // Try a backward linear scan
            let mut res = None;
            for offset in 0..=MAX_SCAN_OFFSET {
                // Prevent underflow
                if i < offset {
                    break; // Reached the beginning without finding
                }
                let idx = i - offset;

                // Check interval [times[idx], times[idx+1])
                // Note: if idx is the last element, the logic differs slightly, but in this
                // "else" branch, time < t_curr means time is definitely less than times[i].
                // If idx == i, we know time < times[i], so it's definitely not in [i, i+1)
                // so the first iteration (offset=0) is effectively a no-op.
                // For logical consistency, we still check the standard interval definition.

                // Standard check: time >= times[idx]
                // (Right boundary time < times[idx+1] is usually satisfied in backward search
                // since we're scanning right to left)
                if time >= self.times[idx] {
                    // Found it! time is after frame idx and less than idx+1
                    // (previous loop iteration verified or implied the right boundary)
                    // Strictly, we only need to confirm the left boundary
                    res = Some(idx);
                    break;
                }
            }
            res
        };

        // Cursor update logic
        let final_index = if let Some(idx) = found_index {
            // Cache/local search hit! Update cursor
            cursor.last_index = idx;
            idx
        } else {
            // === Case C: Large jump (scrubbing / loop reset) ===
            // Local search failed, fall back to global binary search (O(log N))
            // partition_point returns the position of the first element > time, i.e. "next_index"
            let next_idx = self.times.partition_point(|&t| t <= time);
            let idx = if next_idx > 0 { next_idx - 1 } else { 0 };

            // Update cursor for next lookup
            cursor.last_index = idx;
            idx
        };

        self.sample_at_frame(final_index, time)
    }

    /// Helper method: unified value accessor.
    /// For Linear/Step, the index is used directly.
    /// For CubicSpline, the value is at index * 3 + 1.
    fn get_value_at(&self, index: usize) -> &T {
        match self.interpolation {
            InterpolationMode::CubicSpline => &self.values[index * 3 + 1],
            _ => &self.values[index],
        }
    }

    fn sample_at_frame(&self, index: usize, time: f32) -> T {
        let len = self.times.len();

        // 1. Boundary case: no next frame available
        if index >= len - 1 {
            return self.get_value_at(len - 1).clone();
        }

        let next_idx = index + 1;
        let t0 = self.times[index];
        let t1 = self.times[next_idx];
        let dt = t1 - t0;

        // Prevent division by zero
        let t = if dt > 1e-6 { (time - t0) / dt } else { 0.0 };
        // Clamp t to [0, 1] (should already be, but for floating-point safety)
        let t = t.clamp(0.0, 1.0);

        match self.interpolation {
            InterpolationMode::Step => self.get_value_at(index).clone(),
            InterpolationMode::Linear => {
                let v0 = self.get_value_at(index);
                let v1 = self.get_value_at(next_idx);
                T::interpolate_linear(v0, v1, t)
            }
            InterpolationMode::CubicSpline => {
                let i_prev = index * 3;
                let i_next = next_idx * 3;

                let v0 = &self.values[i_prev + 1];
                let out_tangent0 = &self.values[i_prev + 2];
                let in_tangent1 = &self.values[i_next];
                let v1 = &self.values[i_next + 1];

                T::interpolate_cubic(v0, out_tangent0, in_tangent1, v1, t, dt)
            }
        }
    }
}
