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

        // partition_point returns the first index where t > time (right boundary).
        // sample_at_frame expects the left boundary of the interval, so subtract 1.
        let next_idx = self.times.partition_point(|&t| t <= time);
        let idx = if next_idx > 0 { next_idx - 1 } else { 0 };

        self.sample_at_frame(idx, time)
    }

    fn find_frame_index(&self, time: f32, cursor: &mut KeyframeCursor) -> usize {
        // === O(1) optimization logic resurrected here ===

        let len = self.times.len();
        let i = cursor.last_index;
        // Get the time at the current cursor position
        // Safety check: if cursor is out of bounds (e.g. clip was switched), reset to 0
        let t_curr = *self.times.get(i).unwrap_or(&self.times[0]);

        // Decision: search forward or backward?
        if time >= t_curr {
            // === Case A: Normal playback or fast-forward (time increasing) ===
            // Try a forward linear scan up to MAX_SCAN_OFFSET steps

            // Starting from position i, check up to i + MAX_SCAN_OFFSET
            // i.e. check intervals: [i, i+1), [i+1, i+2)...
            for offset in 0..=MAX_SCAN_OFFSET {
                let idx = i + offset;
                if idx >= len - 1 {
                    return len - 1;
                }
                if time < self.times[idx + 1] {
                    return idx;
                }
            }
        } else {
            // === Case B: Reverse playback or loop reset (time decreasing) ===
            // Try a backward linear scan
            for offset in 0..=MAX_SCAN_OFFSET {
                if i < offset {
                    break;
                }
                let idx = i - offset;
                if time >= self.times[idx] {
                    return idx;
                }
            }
        }

        // === Case C: Large jump (scrubbing / loop reset) ===
        // Local search failed, fall back to global binary search (O(log N))
        // partition_point returns the position of the first element > time, i.e. "next_index"
        let next_idx = self.times.partition_point(|&t| t <= time);
        if next_idx > 0 { next_idx - 1 } else { 0 }
    }

    /// Core optimization: sampling with cursor.
    /// cursor: mutable reference that will be updated.
    pub fn sample_with_cursor(&self, time: f32, cursor: &mut KeyframeCursor) -> T {
        assert!(!self.times.is_empty(), "Track is empty");
        if self.times.len() == 1 {
            return self.get_value_at(0).clone();
        }
        let idx = self.find_frame_index(time, cursor);
        cursor.last_index = idx;
        self.sample_at_frame(idx, time)
    }

    pub fn sample_with_cursor_into(&self, time: f32, cursor: &mut KeyframeCursor, out: &mut T) {
        if self.times.is_empty() {
            return;
        }
        if self.times.len() == 1 {
            *out = self.get_value_at(0).clone();
            return;
        }
        let idx = self.find_frame_index(time, cursor);
        cursor.last_index = idx;
        self.sample_at_frame_into(idx, time, out);
    }

    fn sample_at_frame_into(&self, index: usize, time: f32, out: &mut T) {
        let len = self.times.len();
        if index >= len - 1 {
            *out = self.get_value_at(len - 1).clone();
            return;
        }

        let next_idx = index + 1;
        let t0 = self.times[index];
        let t1 = self.times[next_idx];
        let dt = t1 - t0;
        let t = if dt > 1e-6 { (time - t0) / dt } else { 0.0 }.clamp(0.0, 1.0);

        match self.interpolation {
            InterpolationMode::Step => *out = self.get_value_at(index).clone(),
            InterpolationMode::Linear => {
                T::interpolate_linear_into(
                    self.get_value_at(index),
                    self.get_value_at(next_idx),
                    t,
                    out,
                );
            }
            InterpolationMode::CubicSpline => {
                let i_prev = index * 3;
                let i_next = next_idx * 3;
                T::interpolate_cubic_into(
                    &self.values[i_prev + 1],
                    &self.values[i_prev + 2],
                    &self.values[i_next],
                    &self.values[i_next + 1],
                    t,
                    dt,
                    out,
                );
            }
        }
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
