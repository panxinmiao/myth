//! [gallery]
//! name = "3D Gaussian Splatting Viewer"
//! category = "Gaussian Splatting"
//! description = "Interactive 3DGS viewer with a bundled startup cloud and custom NPZ/PLY/SPZ loading."
//! order = 520
//! features = ["3dgs", "gaussian-npz", "gaussian-spz"]
//!

use std::any::Any;
#[cfg(target_arch = "wasm32")]
use std::io::Cursor;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

use myth::GaussianCloudHandle;
use myth::prelude::*;
use myth_dev_utils::FpsCounter;
use myth_egui::{UiPass, UiPassNode, egui};
use winit::event::WindowEvent;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

const DEFAULT_CLOUD_NAME: &str = "point_cloud.npz";

#[derive(Debug, Clone)]
enum LoadingState {
    Idle,
    Loading(String),
    Error(String),
}

#[derive(Debug, Clone, Copy)]
enum CloudFormat {
    Npz,
    Ply,
    Spz,
}

impl CloudFormat {
    fn from_name(name: &str) -> Option<Self> {
        match name
            .rsplit('.')
            .next()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("npz") => Some(Self::Npz),
            Some("ply") => Some(Self::Ply),
            Some("spz") => Some(Self::Spz),
            _ => None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_name)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_from_path(self, assets: &AssetServer, path: String) -> GaussianCloudHandle {
        match self {
            Self::Npz => assets.load_gaussian_npz(path),
            Self::Ply => assets.load_gaussian_ply(path),
            Self::Spz => assets.load_gaussian_spz(path),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load_from_bytes(
        self,
        assets: &AssetServer,
        data: Vec<u8>,
    ) -> myth::Result<GaussianCloudHandle> {
        let cloud = match self {
            Self::Npz => myth::load_gaussian_npz(Cursor::new(data))?,
            Self::Ply => myth::load_gaussian_ply(Cursor::new(data))?,
            Self::Spz => myth::load_gaussian_spz(Cursor::new(data))?,
        };

        Ok(assets.gaussian_clouds.add(cloud))
    }
}

enum ViewerEvent {
    CloudQueued {
        name: String,
        handle: GaussianCloudHandle,
    },
    LoadError(String),
}

struct GaussianSplattingDemo {
    ui_pass: UiPass,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    cloud_node: NodeHandle,
    event_tx: Sender<ViewerEvent>,
    event_rx: Receiver<ViewerEvent>,
    pending_cloud: Option<(String, GaussianCloudHandle)>,
    current_cloud_name: Option<String>,
    loading_state: LoadingState,
}

impl AppHandler for GaussianSplattingDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("Renderer not initialized");
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let scene = engine.scene_manager.create_active();

        let default_path = format!("{}3dgs/{}", ASSET_PATH, DEFAULT_CLOUD_NAME);
        let cloud_handle = engine.assets.load_gaussian_npz(default_path);
        let cloud_node = scene.add_gaussian_cloud("gaussian_cloud", cloud_handle);

        // Camera — use the first camera from the training data as a starting view
        let camera_pos = Vec3::new(0.0, 2.0, 2.5);
        let target = Vec3::ZERO;

        let cam_node = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node)
            .set_position(camera_pos.x, camera_pos.y, camera_pos.z)
            .look_at(target);
        scene.active_camera = Some(cam_node);

        let (event_tx, event_rx) = channel();

        Self {
            ui_pass,
            controls: OrbitControls::new(camera_pos, target),
            fps_counter: FpsCounter::new(),
            cloud_node,
            event_tx,
            event_rx,
            pending_cloud: Some((DEFAULT_CLOUD_NAME.to_string(), cloud_handle)),
            current_cloud_name: None,
            loading_state: LoadingState::Loading(DEFAULT_CLOUD_NAME.to_string()),
        }
    }

    fn on_event(&mut self, _engine: &mut Engine, window: &dyn Window, event: &dyn Any) -> bool {
        let Some(event) = event.downcast_ref::<WindowEvent>() else {
            return false;
        };

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");

        if self.ui_pass.handle_input(winit_window, event) {
            return true;
        }

        if let WindowEvent::Resized(size) = event {
            self.ui_pass
                .resize(size.width, size.height, window.scale_factor());
        }

        false
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        self.process_load_results(scene, &engine.assets);

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "3D Gaussian Splatting - {} | FPS: {:.2}",
                self.display_name(),
                fps
            ));
        }

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        self.ui_pass.begin_frame(winit_window);
        let egui_ctx = self.ui_pass.context().clone();
        self.handle_drag_and_drop(&egui_ctx, engine.assets.clone());
        self.render_ui(&egui_ctx, engine.assets.clone());
        self.ui_pass.end_frame(winit_window);

        let _ = frame;
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        use myth::renderer::graph::core::{GraphBlackboard, HookStage};

        let Some(composer) = engine.compose_frame() else {
            return;
        };

        self.ui_pass
            .resolve_textures(composer.device(), composer.resource_manager());

        let ui_pass_ref = &mut self.ui_pass;

        composer
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, bb| {
                let new_surface = rdg.add_pass("UI_Pass", |builder| {
                    let out = builder.mutate_texture(bb.surface_out, "Surface_With_UI");
                    let node = UiPassNode {
                        pass: ui_pass_ref,
                        target_tex: out,
                    };
                    (node, out)
                });

                GraphBlackboard {
                    surface_out: new_surface,
                    ..bb
                }
            })
            .render();
    }
}

impl GaussianSplattingDemo {
    fn display_name(&self) -> &str {
        self.current_cloud_name
            .as_deref()
            .or_else(|| self.pending_cloud.as_ref().map(|(name, _)| name.as_str()))
            .unwrap_or(DEFAULT_CLOUD_NAME)
    }

    fn begin_loading_cloud(&mut self, name: String, handle: GaussianCloudHandle) {
        self.pending_cloud = Some((name.clone(), handle));
        self.loading_state = LoadingState::Loading(name);
    }

    fn process_load_results(&mut self, scene: &mut Scene, assets: &AssetServer) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                ViewerEvent::CloudQueued { name, handle } => {
                    self.begin_loading_cloud(name, handle);
                }
                ViewerEvent::LoadError(error) => {
                    log::error!("{error}");
                    self.pending_cloud = None;
                    self.loading_state = LoadingState::Error(error);
                }
            }
        }

        if let Some((name, handle)) = self.pending_cloud.clone() {
            if assets.gaussian_clouds.get(handle).is_some() {
                self.pending_cloud = None;
                self.loading_state = LoadingState::Idle;
                self.apply_loaded_cloud(scene, name, handle);
            } else if let Some(error) = assets.gaussian_clouds.get_error(handle) {
                self.pending_cloud = None;
                self.loading_state =
                    LoadingState::Error(format!("Failed to load '{name}': {error}"));
            }
        }
    }

    fn apply_loaded_cloud(&mut self, scene: &mut Scene, name: String, handle: GaussianCloudHandle) {
        scene.set_gaussian_cloud(self.cloud_node, handle);
        self.current_cloud_name = Some(name);
    }

    fn render_ui(&mut self, ctx: &egui::Context, assets: AssetServer) {
        egui::Window::new("3DGS Viewer")
            .default_pos([10.0, 10.0])
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.label("Bundled startup cloud: examples/assets/3dgs/point_cloud.npz");

                if ui.button("Open .npz / .ply / .spz...").clicked() {
                    self.open_cloud_file(assets.clone());
                }

                match &self.loading_state {
                    LoadingState::Idle => {
                        ui.label(format!("Current: {}", self.display_name()));
                    }
                    LoadingState::Loading(name) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Loading {}...", name));
                        });
                    }
                    LoadingState::Error(error) => {
                        ui.colored_label(egui::Color32::RED, error);
                        ui.label(format!("Current: {}", self.display_name()));
                    }
                }

                ui.separator();
                ui.label("Orbit: drag mouse");
                ui.label("Zoom: mouse wheel / touchpad");
                ui.label("You can also drag a .npz, .ply, or .spz file into the window.");

                #[cfg(target_arch = "wasm32")]
                ui.label("Web build supports file picker and in-window drop.");
            });
    }

    fn open_cloud_file(&mut self, assets: AssetServer) {
        let tx = self.event_tx.clone();

        execute_future(async move {
            let file = rfd::AsyncFileDialog::new()
                .add_filter("3D Gaussian Splatting", &["npz", "ply", "spz"])
                .pick_file()
                .await;

            if let Some(file_handle) = file {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    send_cloud_from_path(file_handle.path().to_path_buf(), assets, tx);
                }

                #[cfg(target_arch = "wasm32")]
                {
                    let name = file_handle.file_name();
                    let data = file_handle.read().await;
                    send_cloud_from_bytes(name, data, assets, tx);
                }
            }
        });
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context, assets: AssetServer) {
        if ctx.input(|input| !input.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("gaussian_cloud_drop_overlay"),
            ));
            let screen_rect = ctx.content_rect();

            painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(120));
            painter.text(
                screen_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop .npz / .ply / .spz file here",
                egui::FontId::proportional(28.0),
                egui::Color32::WHITE,
            );
        }

        let dropped_assets = assets.clone();
        ctx.input(|input| {
            if let Some(file) = input.raw.dropped_files.last() {
                self.process_dropped_file(file, dropped_assets.clone());
            }
        });
    }

    fn process_dropped_file(&mut self, file: &egui::DroppedFile, assets: AssetServer) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = &file.path {
            match CloudFormat::from_path(path) {
                Some(format) => {
                    let name = display_name_from_path(path);
                    let handle = format.load_from_path(&assets, path.to_string_lossy().to_string());
                    self.begin_loading_cloud(name, handle);
                }
                None => {
                    self.loading_state =
                        LoadingState::Error(format!("Unsupported file type: {}", path.display()));
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(bytes) = &file.bytes {
            let name = if file.name.is_empty() {
                "dropped_cloud".to_string()
            } else {
                file.name.clone()
            };
            let tx = self.event_tx.clone();
            let assets = assets.clone();
            let data = bytes.to_vec();
            self.loading_state = LoadingState::Loading(name.clone());

            execute_future(async move {
                send_cloud_from_bytes(name, data, assets, tx);
            });
        } else {
            self.loading_state = LoadingState::Error(
                "Dropped file did not include data in the current platform backend".to_string(),
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn display_name_from_path(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "cloud".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn send_cloud_from_path(path: PathBuf, assets: AssetServer, tx: Sender<ViewerEvent>) {
    let Some(format) = CloudFormat::from_path(&path) else {
        let _ = tx.send(ViewerEvent::LoadError(format!(
            "Unsupported file type: {}",
            path.display()
        )));
        return;
    };

    let name = display_name_from_path(&path);
    let handle = format.load_from_path(&assets, path.to_string_lossy().to_string());
    let _ = tx.send(ViewerEvent::CloudQueued { name, handle });
}

#[cfg(target_arch = "wasm32")]
fn send_cloud_from_bytes(
    name: String,
    data: Vec<u8>,
    assets: AssetServer,
    tx: Sender<ViewerEvent>,
) {
    let Some(format) = CloudFormat::from_name(&name) else {
        let _ = tx.send(ViewerEvent::LoadError(format!(
            "Unsupported file type: {name}"
        )));
        return;
    };

    match format.load_from_bytes(&assets, data) {
        Ok(handle) => {
            let _ = tx.send(ViewerEvent::CloudQueued { name, handle });
        }
        Err(error) => {
            let _ = tx.send(ViewerEvent::LoadError(format!(
                "Failed to load '{name}': {error}"
            )));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_future<F: std::future::Future<Output = ()> + Send + 'static>(future: F) {
    tokio::spawn(future);
}

#[cfg(target_arch = "wasm32")]
fn execute_future<F: std::future::Future<Output = ()> + 'static>(future: F) {
    wasm_bindgen_futures::spawn_local(future);
}

#[myth::main]
fn main() -> myth::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio Runtime");

    #[cfg(not(target_arch = "wasm32"))]
    let _enter = rt.enter();

    App::new()
        .with_title("Myth Engine — 3D Gaussian Splatting")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            vsync: false,
            ..Default::default()
        })
        .run::<GaussianSplattingDemo>()
}
