use std::collections::HashSet;
use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default, Debug, Clone)]
pub struct Input {
    // === 鼠标状态 ===
    /// 当前鼠标在窗口内的位置
    pub cursor_position: Vec2,
    /// 上一帧到这一帧的鼠标位移 (dx, dy)
    pub cursor_delta: Vec2,
    /// 这一帧的滚轮滚动量 (x, y)
    pub scroll_delta: Vec2,
    /// 当前按下的鼠标按键集合
    pub mouse_buttons: HashSet<MouseButton>,

    // === 键盘状态 ===
    /// 当前按下的键盘按键集合 (使用物理按键 Code，适应不同布局)
    pub keys_pressed: HashSet<KeyCode>,

    // === 窗口状态 ===
    /// 窗口大小
    pub screen_size: Vec2,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    /// 核心方法：统一处理所有窗口事件
    /// 这让 App 的主循环非常干净
    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            // 1. 鼠标移动
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                
                // 如果不是第一帧（防止初始化时的跳变），计算位移
                // 注意：这里简单的判断 != ZERO 还是有点隐患，但在渲染循环稳定后没问题
                if self.cursor_position != Vec2::ZERO {
                    self.cursor_delta += new_pos - self.cursor_position;
                }
                self.cursor_position = new_pos;
            }

            // 2. 鼠标按键
            WindowEvent::MouseInput { state, button, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_buttons.insert(*button);
                    }
                    ElementState::Released => {
                        self.mouse_buttons.remove(button);
                    }
                }
            }

            // 3. 鼠标滚轮
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        self.scroll_delta += Vec2::new(*x, *y);
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        // 简单的缩放转换，通常 PixelDelta 值较大，这里给一个经验系数
                        const PIXEL_SCALE: f32 = 0.01; 
                        self.scroll_delta += Vec2::new(pos.x as f32, pos.y as f32) * PIXEL_SCALE;
                    }
                }
            }

            // 4. 键盘输入 (新增)
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                self.handle_keyboard(key_event);
            }

            // 5. 窗口大小改变
            WindowEvent::Resized(size) => {
                self.screen_size = Vec2::new(size.width as f32, size.height as f32);
            }

            _ => {}
        }
    }

    /// 帧末清理
    /// 必须在每一帧逻辑更新结束后调用，清除“瞬时”状态（Delta）
    pub fn end_frame(&mut self) {
        self.cursor_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// 手动处理 Resize (如果 App 在事件之外需要调用)
    pub fn handle_resize(&mut self, width: u32, height: u32) {
        self.screen_size = Vec2::new(width as f32, height as f32);
    }

    // === 内部辅助 ===

    fn handle_keyboard(&mut self, event: &KeyEvent) {
        // 我们只关心物理按键 (PhysicalKey)，这样 WASD 在不同键盘布局下位置一致
        if let PhysicalKey::Code(code) = event.physical_key {
            match event.state {
                ElementState::Pressed => {
                    self.keys_pressed.insert(code);
                }
                ElementState::Released => {
                    self.keys_pressed.remove(&code);
                }
            }
        }
    }

    // === 用户查询接口 ===

    pub fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }

    pub fn is_key_pressed(&self, code: KeyCode) -> bool {
        self.keys_pressed.contains(&code)
    }
}