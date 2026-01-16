use std::time::{Duration, Instant};

pub struct FpsCounter {
    last_update: Instant,
    frame_count: u32,
    accumulated_time: Duration,
    pub current_fps: f32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            frame_count: 0,
            accumulated_time: Duration::new(0, 0),
            current_fps: 0.0,
        }
    }

    pub fn update(&mut self) -> Option<f32> {
        self.frame_count += 1;
        let now = Instant::now();
        let delta = now - self.last_update;
        self.last_update = now;
        self.accumulated_time += delta;

        // 每 1 秒 (1000ms) 更新一次统计
        if self.accumulated_time.as_secs_f32() >= 1.0 {
            self.current_fps = self.frame_count as f32 / self.accumulated_time.as_secs_f32();
            
            // 重置计数器
            self.accumulated_time = Duration::new(0, 0);
            self.frame_count = 0;
            
            return Some(self.current_fps);
        }

        None
    }
}