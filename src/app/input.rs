use std::collections::HashSet;
use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

#[derive(Default, Debug, Clone)]
pub struct Input {
    /// 当前鼠标在窗口内的位置
    pub cursor_position: Vec2,
    /// 上一帧到这一帧的鼠标位移 (dx, dy)
    pub cursor_delta: Vec2,
    /// 这一帧的滚轮滚动量 (x, y)
    pub scroll_delta: Vec2,
    /// 窗口大小
    pub screen_size: Vec2,
    /// 当前按下的鼠标按键集合
    pub mouse_buttons: HashSet<MouseButton>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    /// 帧末清理（清除 delta 状态，防止一直旋转）
    pub fn end_frame(&mut self) {
        self.cursor_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        self.screen_size = Vec2::new(width as f32, height as f32);
    }

    pub fn handle_cursor_move(&mut self, x: f64, y: f64) {
        let new_pos = Vec2::new(x as f32, y as f32);
        // 如果是第一帧，delta 设为 0，否则计算差值
        if self.cursor_position != Vec2::ZERO {
            self.cursor_delta += new_pos - self.cursor_position;
        }
        self.cursor_position = new_pos;
    }

    pub fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        match state {
            ElementState::Pressed => {
                self.mouse_buttons.insert(button);
            }
            ElementState::Released => {
                self.mouse_buttons.remove(&button);
            }
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                self.scroll_delta += Vec2::new(x, y);
            }
            MouseScrollDelta::PixelDelta(pos) => {
                // 简单的缩放转换，通常 PixelDelta 值较大
                self.scroll_delta += Vec2::new(pos.x as f32, pos.y as f32) * 0.1;
            }
        }
    }

    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }
}