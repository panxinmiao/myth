use glam::Vec3;
use winit::event::MouseButton;
use crate::app::input::Input;
use crate::scene::transform::Transform;

pub struct OrbitControls {
    pub rotate_speed: f32,
    pub zoom_speed: f32,
    pub pan_speed: f32,
    pub damping_factor: f32,
    pub enable_damping: bool,
    pub min_distance: f32,
    pub max_distance: f32,

    target_center: Vec3,
    target_radius: f32,
    target_theta: f32,
    target_phi: f32,

    pub center: Vec3,
    pub radius: f32,
    pub theta: f32,
    pub phi: f32,
}

impl OrbitControls {
    pub fn new(center: Vec3, radius: f32) -> Self {
        let initial_theta = 0.0;
        let initial_phi = std::f32::consts::FRAC_PI_2;

        Self {
            rotate_speed: 0.005,
            zoom_speed: 0.05,
            pan_speed: 0.0015,
            damping_factor: 0.1,
            enable_damping: true,
            min_distance: 1.0,
            max_distance: 1000.0,

            target_center: center,
            target_radius: radius,
            target_theta: initial_theta,
            target_phi: initial_phi,

            center,
            radius,
            theta: initial_theta,
            phi: initial_phi,
        }
    }

    /// 每帧更新控制器
    /// 参数：`transform` 为相机的 `Transform` 组件，`input` 为输入状态，`fov_degrees` 为相机视野（度）
    pub fn update(&mut self, transform: &mut Transform, input: &Input, fov_degrees: f32) {
        let half_fov = fov_degrees.to_radians() / 2.0;
        let screen_height = input.screen_size.y.max(1.0);

        let distance = self.radius;
        let target_world_height = 2.0 * distance * half_fov.tan();
        let pixels_to_world_ratio = target_world_height / screen_height;

        if input.is_button_pressed(MouseButton::Left) {
            self.target_theta -= input.cursor_delta.x * self.rotate_speed;
            self.target_phi -= input.cursor_delta.y * self.rotate_speed;
            const EPS: f32 = 0.01;
            self.target_phi = self.target_phi.clamp(EPS, std::f32::consts::PI - EPS);
        }

        if input.scroll_delta.y != 0.0 {
            let zoom_scale = input.scroll_delta.y * self.zoom_speed;
            self.target_radius -= self.target_radius * zoom_scale;
            self.target_radius = self.target_radius.clamp(self.min_distance, self.max_distance);
        }

        if input.is_button_pressed(MouseButton::Right) {
            let sin_phi = self.phi.sin();
            let cos_phi = self.phi.cos();
            let sin_theta = self.theta.sin();
            let cos_theta = self.theta.cos();

            let offset = Vec3::new(
                sin_phi * sin_theta,
                cos_phi,
                sin_phi * cos_theta,
            );

            let forward = -offset.normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let up = right.cross(forward).normalize();

            let pan_delta_world = (right * -input.cursor_delta.x + up * input.cursor_delta.y) * pixels_to_world_ratio;
            self.target_center += pan_delta_world;
        }

        if self.enable_damping {
            self.theta  += (self.target_theta - self.theta) * self.damping_factor;
            self.phi    += (self.target_phi - self.phi) * self.damping_factor;
            self.radius += (self.target_radius - self.radius) * self.damping_factor;
            self.center += (self.target_center - self.center) * self.damping_factor;
        } else {
            self.theta = self.target_theta;
            self.phi = self.target_phi;
            self.radius = self.target_radius;
            self.center = self.target_center;
        }

        let sin_phi = self.phi.sin();
        let cos_phi = self.phi.cos();
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();

        let offset = Vec3::new(
            self.radius * sin_phi * sin_theta,
            self.radius * cos_phi,
            self.radius * sin_phi * cos_theta,
        );

        transform.position = self.center + offset;
        transform.look_at(self.center, Vec3::Y);
    }
}