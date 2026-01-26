use std::time::{Duration, Instant};

pub struct Timer {
    start_time: Instant,
    last_update: Instant,
    pub delta: Duration,
    pub elapsed: Duration,
    pub frame_count: u64,
}

impl Timer {
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

    /// 引擎内部调用：打点
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.delta = now - self.last_update;
        self.elapsed = now - self.start_time;
        self.last_update = now;
        self.frame_count += 1;
    }
    
    pub fn dt_seconds(&self) -> f32 {
        self.delta.as_secs_f32()
    }
}