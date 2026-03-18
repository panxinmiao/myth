"""
Myth Engine — PySide6 (Qt) Integration Demo
=============================================

Embeds a Myth 3D viewport inside a Qt application using a QWidget.
The same pattern works with PyQt6 — just change the imports.

Requirements:
    pip install PySide6

Usage:
    python examples/pyside_demo.py
"""

import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)

import myth


class MythViewport(QWidget):
    """A QWidget that hosts a Myth 3D renderer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Opaque background — we draw everything via wgpu
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setMinimumSize(640, 480)

        self.renderer = myth.Renderer(render_path="high_fidelity", vsync=True)
        self._initialized = False

        # Scene objects
        self.cube = None
        self.cam = None
        self.orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])

        # 60 fps timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(16)  # ~60 fps

    def _ensure_init(self):
        if self._initialized:
            return
        hwnd = int(self.winId())

        # get physical size for HiDPI support
        w, h = (
            int(self.width() * self.devicePixelRatio()),
            int(self.height() * self.devicePixelRatio()),
        )
        if w <= 0 or h <= 0:
            return

        self.renderer.init_with_handle(hwnd, w, h)
        self._setup_scene()
        self._initialized = True

    def _setup_scene(self):
        scene = self.renderer.create_scene()

        self.cube = scene.add_mesh(
            myth.BoxGeometry(1, 1, 1),
            myth.PhysicalMaterial(color="#ff8033", roughness=0.4, metalness=0.3),
        )
        self.cube.position = [0, 0.5, 0]

        ground = scene.add_mesh(
            myth.PlaneGeometry(width=20, height=20),
            myth.PhysicalMaterial(color="#666666", roughness=0.8),
        )
        ground.rotation_euler = [-90, 0, 0]

        self.cam = scene.add_camera(myth.PerspectiveCamera(fov=60, near=0.1))
        self.cam.position = [3, 3, 5]
        self.cam.look_at([0, 0.5, 0])
        scene.active_camera = self.cam

        sun = scene.add_light(
            myth.DirectionalLight(
                color=[1, 0.95, 0.9], intensity=2.0, cast_shadows=True
            )
        )
        sun.position = [5, 10, 5]
        sun.look_at([0, 0, 0])

        scene.set_background_color(0.1, 0.1, 0.15)
        scene.set_tone_mapping("aces")

    def _on_frame(self):
        self._ensure_init()
        if not self._initialized:
            return

        if self.cube:
            self.cube.rotate_y(0.016 * 0.5)
        if self.cam:
            self.orbit.update(self.cam, 0.016)

        self.renderer.frame()

    # ── Qt events → Myth input ────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._initialized:
            sz = event.size()
            self.renderer.resize(sz.width(), sz.height())

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.renderer.inject_mouse_move(pos.x(), pos.y())

    def mousePressEvent(self, event):
        btn_map = {
            Qt.MouseButton.LeftButton: 0,
            Qt.MouseButton.MiddleButton: 1,
            Qt.MouseButton.RightButton: 2,
        }
        btn = btn_map.get(event.button())
        if btn is not None:
            self.renderer.inject_mouse_down(btn)

    def mouseReleaseEvent(self, event):
        btn_map = {
            Qt.MouseButton.LeftButton: 0,
            Qt.MouseButton.MiddleButton: 1,
            Qt.MouseButton.RightButton: 2,
        }
        btn = btn_map.get(event.button())
        if btn is not None:
            self.renderer.inject_mouse_up(btn)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        self.renderer.inject_scroll(delta.x() / 120.0, delta.y() / 120.0)

    def closeEvent(self, event):
        self._timer.stop()
        self.renderer.dispose()
        super().closeEvent(event)

    def paintEngine(self):
        # Disable Qt painting — wgpu draws directly to the surface
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Myth Engine — PySide6 Integration")
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 3D viewport
        self.viewport = MythViewport()
        layout.addWidget(self.viewport, stretch=1)

        # Control bar
        bar = QHBoxLayout()
        layout.addLayout(bar)

        bar.addWidget(QLabel("This is a Qt UI control bar."))
        bar.addStretch()
        btn = QPushButton("Reset Camera")
        btn.clicked.connect(self._reset_camera)
        bar.addWidget(btn)

    def _reset_camera(self):
        if self.viewport.cam:
            self.viewport.cam.position = [3, 3, 5]
            self.viewport.cam.look_at([0, 0.5, 0])
            self.viewport.orbit = myth.OrbitControls(
                position=[3, 3, 5], target=[0, 0.5, 0]
            )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
