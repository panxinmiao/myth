//! Platform-agnostic input system
//!
//! Defines input types and state containers that do not depend on any GUI library.
//! Concrete platform adapters (e.g., Winit Adapter) are responsible for translating
//! platform events into these types.

use glam::Vec2;
use std::collections::HashSet;

/// Keyboard key enumeration (platform-agnostic)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    // Letter keys
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

    // Number keys
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

    // Function keys
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

    // Control keys
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

    // Modifier keys
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,

    // Arrow keys
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,

    // Punctuation
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

    // Numpad
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

/// Mouse button enumeration
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

/// Button state
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ButtonState {
    Pressed,
    Released,
}

/// Platform-agnostic input state container
#[derive(Debug, Clone)]
pub struct Input {
    // Keyboard state
    pressed_keys: HashSet<Key>,
    just_pressed_keys: HashSet<Key>,
    just_released_keys: HashSet<Key>,

    // Mouse button state
    pressed_mouse: HashSet<MouseButton>,
    just_pressed_mouse: HashSet<MouseButton>,
    just_released_mouse: HashSet<MouseButton>,

    // Mouse position and movement
    mouse_position: Vec2,
    mouse_delta: Vec2,
    scroll_delta: Vec2,

    // Window state
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

    // ========== System API (called by Engine/Adapter) ==========

    /// Clears transient state at the start of each frame (JustPressed/JustReleased/Delta)
    pub fn start_frame(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_mouse.clear();
        self.just_released_mouse.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// Injects a keyboard event
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

    /// Injects a mouse button event
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

    /// Injects a mouse position update
    pub fn inject_mouse_position(&mut self, x: f32, y: f32) {
        let new_pos = Vec2::new(x, y);
        if self.mouse_position != Vec2::ZERO {
            self.mouse_delta += new_pos - self.mouse_position;
        }
        self.mouse_position = new_pos;
    }

    /// Injects a scroll wheel event
    pub fn inject_scroll(&mut self, delta_x: f32, delta_y: f32) {
        self.scroll_delta += Vec2::new(delta_x, delta_y);
    }

    /// Injects a window resize event
    pub fn inject_resize(&mut self, width: u32, height: u32) {
        self.screen_size = Vec2::new(width as f32, height as f32);
    }

    // ========== User API (for game/scene logic queries) ==========

    /// Checks whether a key is currently held down
    #[must_use]
    pub fn get_key(&self, key: Key) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Checks whether a key was just pressed this frame
    #[must_use]
    pub fn get_key_down(&self, key: Key) -> bool {
        self.just_pressed_keys.contains(&key)
    }

    /// Checks whether a key was just released this frame
    #[must_use]
    pub fn get_key_up(&self, key: Key) -> bool {
        self.just_released_keys.contains(&key)
    }

    /// Checks whether a mouse button is currently held down
    #[must_use]
    pub fn get_mouse_button(&self, button: MouseButton) -> bool {
        self.pressed_mouse.contains(&button)
    }

    /// Checks whether a mouse button was just pressed this frame
    #[must_use]
    pub fn get_mouse_button_down(&self, button: MouseButton) -> bool {
        self.just_pressed_mouse.contains(&button)
    }

    /// Checks whether a mouse button was just released this frame
    #[must_use]
    pub fn get_mouse_button_up(&self, button: MouseButton) -> bool {
        self.just_released_mouse.contains(&button)
    }

    /// Returns the current mouse position
    #[must_use]
    pub fn mouse_position(&self) -> Vec2 {
        self.mouse_position
    }

    /// Returns the mouse movement delta for this frame
    #[must_use]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// Returns the scroll wheel delta for this frame
    #[must_use]
    pub fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }

    /// Returns the window dimensions
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
