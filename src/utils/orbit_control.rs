use glam::{Vec2, Vec3};
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

    pub center: Vec3,
    pub radius: f32,
    pub theta: f32,
    pub phi: f32,

    rotate_delta: Vec2, 
}

impl OrbitControls {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            rotate_speed: 1.0,
            zoom_speed: 0.05,
            pan_speed: 1.0,
            damping_factor: 0.05,
            enable_damping: true,
            min_distance: 1.0,
            max_distance: 1000.0,

            center,
            radius,
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_2,
            
            rotate_delta: Vec2::ZERO,
        }
    }

    pub fn update(&mut self, transform: &mut Transform, input: &Input, fov_degrees: f32, dt: f32) {
        let screen_height = input.screen_size.y.max(1.0);
        
        if input.is_button_pressed(MouseButton::Left) {
            let rotate_per_pixel = 2.0 * std::f32::consts::PI / screen_height;
            self.rotate_delta.x -= input.cursor_delta.x * rotate_per_pixel * self.rotate_speed;
            self.rotate_delta.y -= input.cursor_delta.y * rotate_per_pixel * self.rotate_speed;
        }

        if self.enable_damping {
            let target_fps = 60.0;
    
            let retention = (1.0 - self.damping_factor).powf(dt * target_fps);

            let delta_apply = self.rotate_delta * (1.0 - retention);
            
            self.theta += delta_apply.x;
            self.phi += delta_apply.y;
            
            self.rotate_delta *= retention;
        } else {
            self.theta += self.rotate_delta.x;
            self.phi += self.rotate_delta.y;
            self.rotate_delta = Vec2::ZERO;
        }

        const EPS: f32 = 0.0001;
        self.phi = self.phi.clamp(EPS, std::f32::consts::PI - EPS);

        if input.scroll_delta.y != 0.0 {
            let scale = (1.0 - self.zoom_speed).powf(input.scroll_delta.y.abs());
            if input.scroll_delta.y > 0.0 {
                self.radius *= scale;
            } else {
                self.radius /= scale;
            }
            self.radius = self.radius.clamp(self.min_distance, self.max_distance);
        }

        if input.is_button_pressed(MouseButton::Right) {
            let half_fov = fov_degrees.to_radians() / 2.0;
            let distance = self.radius;
            let target_world_height = 2.0 * distance * half_fov.tan();
            let pixels_to_world_ratio = target_world_height / screen_height;

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

            let pan_delta_world = (right * -input.cursor_delta.x + up * input.cursor_delta.y) 
                * pixels_to_world_ratio * self.pan_speed;
            
            self.center += pan_delta_world;
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