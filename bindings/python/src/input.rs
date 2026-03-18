//! Input proxy — read-only access to keyboard and mouse state.

use pyo3::prelude::*;

use myth_engine::resources::input::{Key, MouseButton};

use crate::with_engine;

/// Read-only input state proxy.
///
/// Access through `ctx.input` inside update callbacks.
#[pyclass(name = "Input")]
pub struct PyInput {
    _private: (),
}

impl PyInput {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

pub(crate) fn parse_key(name: &str) -> Option<Key> {
    match name {
        // Letters
        "KeyA" | "a" | "A" => Some(Key::A),
        "KeyB" | "b" | "B" => Some(Key::B),
        "KeyC" | "c" | "C" => Some(Key::C),
        "KeyD" | "d" | "D" => Some(Key::D),
        "KeyE" | "e" | "E" => Some(Key::E),
        "KeyF" | "f" | "F" => Some(Key::F),
        "KeyG" | "g" | "G" => Some(Key::G),
        "KeyH" | "h" | "H" => Some(Key::H),
        "KeyI" | "i" | "I" => Some(Key::I),
        "KeyJ" | "j" | "J" => Some(Key::J),
        "KeyK" | "k" | "K" => Some(Key::K),
        "KeyL" | "l" | "L" => Some(Key::L),
        "KeyM" | "m" | "M" => Some(Key::M),
        "KeyN" | "n" | "N" => Some(Key::N),
        "KeyO" | "o" | "O" => Some(Key::O),
        "KeyP" | "p" | "P" => Some(Key::P),
        "KeyQ" | "q" | "Q" => Some(Key::Q),
        "KeyR" | "r" | "R" => Some(Key::R),
        "KeyS" | "s" | "S" => Some(Key::S),
        "KeyT" | "t" | "T" => Some(Key::T),
        "KeyU" | "u" | "U" => Some(Key::U),
        "KeyV" | "v" | "V" => Some(Key::V),
        "KeyW" | "w" | "W" => Some(Key::W),
        "KeyX" | "x" | "X" => Some(Key::X),
        "KeyY" | "y" | "Y" => Some(Key::Y),
        "KeyZ" | "z" | "Z" => Some(Key::Z),

        // Digits
        "Digit0" | "0" => Some(Key::Key0),
        "Digit1" | "1" => Some(Key::Key1),
        "Digit2" | "2" => Some(Key::Key2),
        "Digit3" | "3" => Some(Key::Key3),
        "Digit4" | "4" => Some(Key::Key4),
        "Digit5" | "5" => Some(Key::Key5),
        "Digit6" | "6" => Some(Key::Key6),
        "Digit7" | "7" => Some(Key::Key7),
        "Digit8" | "8" => Some(Key::Key8),
        "Digit9" | "9" => Some(Key::Key9),

        // Special keys
        "Space" | "space" => Some(Key::Space),
        "Enter" | "Return" | "enter" => Some(Key::Enter),
        "Escape" | "Esc" | "escape" => Some(Key::Escape),
        "Tab" | "tab" => Some(Key::Tab),
        "Backspace" | "backspace" => Some(Key::Backspace),

        // Modifiers
        "ShiftLeft" | "Shift" | "shift" => Some(Key::ShiftLeft),
        "ShiftRight" => Some(Key::ShiftRight),
        "ControlLeft" | "Control" | "Ctrl" | "ctrl" => Some(Key::ControlLeft),
        "ControlRight" => Some(Key::ControlRight),
        "AltLeft" | "Alt" | "alt" => Some(Key::AltLeft),
        "AltRight" => Some(Key::AltRight),

        // Arrows
        "ArrowUp" | "Up" | "up" => Some(Key::ArrowUp),
        "ArrowDown" | "Down" | "down" => Some(Key::ArrowDown),
        "ArrowLeft" | "Left" | "left" => Some(Key::ArrowLeft),
        "ArrowRight" | "Right" | "right" => Some(Key::ArrowRight),

        // F-keys
        "F1" => Some(Key::F1),
        "F2" => Some(Key::F2),
        "F3" => Some(Key::F3),
        "F4" => Some(Key::F4),
        "F5" => Some(Key::F5),
        "F6" => Some(Key::F6),
        "F7" => Some(Key::F7),
        "F8" => Some(Key::F8),
        "F9" => Some(Key::F9),
        "F10" => Some(Key::F10),
        "F11" => Some(Key::F11),
        "F12" => Some(Key::F12),

        _ => None,
    }
}

fn parse_mouse_button(name: &str) -> Option<MouseButton> {
    match name {
        "Left" | "left" | "0" => Some(MouseButton::Left),
        "Right" | "right" | "2" => Some(MouseButton::Right),
        "Middle" | "middle" | "1" => Some(MouseButton::Middle),
        _ => None,
    }
}

#[pymethods]
impl PyInput {
    /// Returns true if the key is currently held down.
    fn key(&self, name: &str) -> PyResult<bool> {
        match parse_key(name) {
            Some(k) => with_engine(|e| e.input.get_key(k)),
            None => Ok(false),
        }
    }

    /// Returns true on the frame the key was first pressed.
    fn key_down(&self, name: &str) -> PyResult<bool> {
        match parse_key(name) {
            Some(k) => with_engine(|e| e.input.get_key_down(k)),
            None => Ok(false),
        }
    }

    /// Returns true on the frame the key was released.
    fn key_up(&self, name: &str) -> PyResult<bool> {
        match parse_key(name) {
            Some(k) => with_engine(|e| e.input.get_key_up(k)),
            None => Ok(false),
        }
    }

    /// Returns true if the mouse button is currently held.
    fn mouse_button(&self, name: &str) -> PyResult<bool> {
        match parse_mouse_button(name) {
            Some(b) => with_engine(|e| e.input.get_mouse_button(b)),
            None => Ok(false),
        }
    }

    /// Returns true on the frame the mouse button was first pressed.
    fn mouse_button_down(&self, name: &str) -> PyResult<bool> {
        match parse_mouse_button(name) {
            Some(b) => with_engine(|e| e.input.get_mouse_button_down(b)),
            None => Ok(false),
        }
    }

    /// Returns true on the frame the mouse button was released.
    fn mouse_button_up(&self, name: &str) -> PyResult<bool> {
        match parse_mouse_button(name) {
            Some(b) => with_engine(|e| e.input.get_mouse_button_up(b)),
            None => Ok(false),
        }
    }

    /// Current mouse position in window pixels [x, y].
    fn mouse_position(&self) -> PyResult<[f32; 2]> {
        with_engine(|e| {
            let pos = e.input.mouse_position();
            [pos.x, pos.y]
        })
    }

    /// Mouse movement delta since last frame [dx, dy].
    fn mouse_delta(&self) -> PyResult<[f32; 2]> {
        with_engine(|e| {
            let d = e.input.mouse_delta();
            [d.x, d.y]
        })
    }

    /// Mouse scroll wheel delta since last frame [dx, dy].
    fn scroll_delta(&self) -> PyResult<[f32; 2]> {
        with_engine(|e| {
            let d = e.input.scroll_delta();
            [d.x, d.y]
        })
    }

    fn __repr__(&self) -> String {
        "Input(...)".to_string()
    }
}
