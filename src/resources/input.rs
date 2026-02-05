//! 平台无关的输入系统
//!
//! 定义了不依赖任何 GUI 库的输入类型和状态容器。
//! 具体的平台适配器（如 Winit Adapter）负责将平台事件翻译为这些类型。

use glam::Vec2;
use std::collections::HashSet;

/// 键盘按键枚举（平台无关）
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    // 字母键
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    // 数字键
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,

    // 功能键
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,

    // 控制键
    Space,
    Enter,
    Escape,
    Backspace,
    Tab,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,

    // 修饰键
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,

    // 方向键
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,

    // 标点符号
    Comma,
    Period,
    Slash,
    Backslash,
    Semicolon,
    Quote,
    BracketLeft,
    BracketRight,
    Minus,
    Equal,
    Grave,

    // 小键盘
    Numpad0,
    Numpad1,
    Numpad2,
    Numpad3,
    Numpad4,
    Numpad5,
    Numpad6,
    Numpad7,
    Numpad8,
    Numpad9,
    NumpadAdd,
    NumpadSubtract,
    NumpadMultiply,
    NumpadDivide,
    NumpadDecimal,
    NumpadEnter,
}

/// 鼠标按键枚举
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

/// 按键状态
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ButtonState {
    Pressed,
    Released,
}

/// 平台无关的输入状态容器
#[derive(Debug, Clone)]
pub struct Input {
    // 键盘状态
    pressed_keys: HashSet<Key>,
    just_pressed_keys: HashSet<Key>,
    just_released_keys: HashSet<Key>,

    // 鼠标按键状态
    pressed_mouse: HashSet<MouseButton>,
    just_pressed_mouse: HashSet<MouseButton>,
    just_released_mouse: HashSet<MouseButton>,

    // 鼠标位置和移动
    mouse_position: Vec2,
    mouse_delta: Vec2,
    scroll_delta: Vec2,

    // 窗口状态
    screen_size: Vec2,
}

impl Input {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            just_pressed_keys: HashSet::new(),
            just_released_keys: HashSet::new(),
            pressed_mouse: HashSet::new(),
            just_pressed_mouse: HashSet::new(),
            just_released_mouse: HashSet::new(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            screen_size: Vec2::ZERO,
        }
    }

    // ========== System API (供 Engine/Adapter 调用) ==========

    /// 帧开始时清理瞬时状态（JustPressed/JustReleased/Delta）
    pub fn start_frame(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_mouse.clear();
        self.just_released_mouse.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// 注入键盘事件
    pub fn inject_key(&mut self, key: Key, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                if self.pressed_keys.insert(key) {
                    self.just_pressed_keys.insert(key);
                }
            }
            ButtonState::Released => {
                if self.pressed_keys.remove(&key) {
                    self.just_released_keys.insert(key);
                }
            }
        }
    }

    /// 注入鼠标按键事件
    pub fn inject_mouse_button(&mut self, button: MouseButton, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                if self.pressed_mouse.insert(button) {
                    self.just_pressed_mouse.insert(button);
                }
            }
            ButtonState::Released => {
                if self.pressed_mouse.remove(&button) {
                    self.just_released_mouse.insert(button);
                }
            }
        }
    }

    /// 注入鼠标位置
    pub fn inject_mouse_position(&mut self, x: f32, y: f32) {
        let new_pos = Vec2::new(x, y);
        if self.mouse_position != Vec2::ZERO {
            self.mouse_delta += new_pos - self.mouse_position;
        }
        self.mouse_position = new_pos;
    }

    /// 注入滚轮滚动
    pub fn inject_scroll(&mut self, delta_x: f32, delta_y: f32) {
        self.scroll_delta += Vec2::new(delta_x, delta_y);
    }

    /// 注入窗口尺寸变化
    pub fn inject_resize(&mut self, width: u32, height: u32) {
        self.screen_size = Vec2::new(width as f32, height as f32);
    }

    // ========== User API (供游戏/场景逻辑查询) ==========

    /// 检查按键是否正在按下
    #[must_use]
    pub fn get_key(&self, key: Key) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// 检查按键是否在这一帧刚按下
    #[must_use]
    pub fn get_key_down(&self, key: Key) -> bool {
        self.just_pressed_keys.contains(&key)
    }

    /// 检查按键是否在这一帧刚释放
    #[must_use]
    pub fn get_key_up(&self, key: Key) -> bool {
        self.just_released_keys.contains(&key)
    }

    /// 检查鼠标按键是否正在按下
    #[must_use]
    pub fn get_mouse_button(&self, button: MouseButton) -> bool {
        self.pressed_mouse.contains(&button)
    }

    /// 检查鼠标按键是否在这一帧刚按下
    #[must_use]
    pub fn get_mouse_button_down(&self, button: MouseButton) -> bool {
        self.just_pressed_mouse.contains(&button)
    }

    /// 检查鼠标按键是否在这一帧刚释放
    #[must_use]
    pub fn get_mouse_button_up(&self, button: MouseButton) -> bool {
        self.just_released_mouse.contains(&button)
    }

    /// 获取当前鼠标位置
    #[must_use]
    pub fn mouse_position(&self) -> Vec2 {
        self.mouse_position
    }

    /// 获取这一帧的鼠标移动量
    #[must_use]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// 获取这一帧的滚轮滚动量
    #[must_use]
    pub fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }

    /// 获取窗口尺寸
    #[must_use]
    pub fn screen_size(&self) -> Vec2 {
        self.screen_size
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}
