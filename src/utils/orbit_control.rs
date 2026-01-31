use glam::Vec3;
use crate::resources::input::{Input, MouseButton};
use crate::scene::transform::Transform;

#[derive(Clone, Copy, Debug)]
struct Spherical {
    pub radius: f32,
    pub phi: f32,
    pub theta: f32,
}

impl Spherical {
    fn new(radius: f32, phi: f32, theta: f32) -> Self {
        Self { radius, phi, theta }
    }

    fn set_from_vec3(&mut self, v: Vec3) {
        self.radius = v.length();
        if self.radius == 0.0 {
            self.theta = 0.0;
            self.phi = 0.0;
        } else {
            self.theta = v.x.atan2(v.z);
            self.phi = (v.y / self.radius).clamp(-1.0, 1.0).acos();
        }
    }

    fn make_safe(&mut self) {
        const EPS: f32 = 0.000001;
        self.phi = self.phi.clamp(EPS, std::f32::consts::PI - EPS);
    }
}

pub struct OrbitControls {

    pub enable_damping: bool,
    pub damping_factor: f32,
    pub zoom_damping_factor: f32,

    pub enable_zoom: bool,
    pub zoom_speed: f32,
    pub enable_rotate: bool,
    pub rotate_speed: f32,
    pub enable_pan: bool,
    pub pan_speed: f32,

    pub min_distance: f32,
    pub max_distance: f32,
    pub min_polar_angle: f32,
    pub max_polar_angle: f32,

 
    target: Vec3,
    spherical: Spherical,

    spherical_delta: Spherical,
    pan_offset: Vec3,   
    
    target_radius: f32,         
}

impl OrbitControls {
    pub fn new(camera_pos: Vec3, target: Vec3) -> Self {
        let mut spherical = Spherical::new(1.0, 0.0, 0.0);
        spherical.set_from_vec3(camera_pos - target);

        Self {
            enable_damping: true,
            damping_factor: 0.05,
            zoom_damping_factor: 0.1,

            enable_zoom: true,
            zoom_speed: 1.0,

            enable_rotate: true,
            rotate_speed: 0.1,

            enable_pan: true,
            pan_speed: 1.0,

            min_distance: 0.0,
            max_distance: f32::INFINITY,
            min_polar_angle: 0.0,
            max_polar_angle: std::f32::consts::PI,

            target,
            spherical,
            spherical_delta: Spherical::new(0.0, 0.0, 0.0),
            pan_offset: Vec3::ZERO,
            
            // Initialize target radius = current radius
            target_radius: spherical.radius, 
        }
    }

    pub fn update(&mut self, transform: &mut Transform, input: &Input, fov_degrees: f32, dt: f32) {
        let screen_height = input.screen_size().y.max(1.0);

        // Rotation
        if self.enable_rotate && input.get_mouse_button(MouseButton::Left) {
            let mouse_delta = input.mouse_delta();
            let rotate_angle_x = 2.0 * std::f32::consts::PI * mouse_delta.x / screen_height;
            let rotate_angle_y = 2.0 * std::f32::consts::PI * mouse_delta.y / screen_height;

            self.spherical_delta.theta -= rotate_angle_x * self.rotate_speed;
            self.spherical_delta.phi -= rotate_angle_y * self.rotate_speed;
        }

        // Zoom
        let scroll = input.scroll_delta();
        if self.enable_zoom && scroll.y != 0.0 {
            let zoom_scale = 0.95f32.powf(self.zoom_speed);

            if scroll.y > 0.0 {
                self.target_radius *= zoom_scale;
            } else {
                self.target_radius /= zoom_scale;
            }

            self.target_radius = self.target_radius.clamp(self.min_distance, self.max_distance);
        }

        // Panning
        if self.enable_pan && input.get_mouse_button(MouseButton::Right) {
            let mouse_delta = input.mouse_delta();
            let position = transform.position;
            let offset_dir = position - self.target;
            let target_distance = offset_dir.length();
            let target_world_height = 2.0 * target_distance * (fov_degrees.to_radians() * 0.5).tan();

            let pan_x = -mouse_delta.x * (target_world_height / screen_height) * self.pan_speed;
            let pan_y = mouse_delta.y * (target_world_height / screen_height) * self.pan_speed;

            let (right, up, _) = transform.rotation_basis();
            self.pan_offset += right * pan_x + up * pan_y;
        }

        let time_scale = dt * 60.0;

        // Apply panning
        self.target += self.pan_offset;
        self.pan_offset = Vec3::ZERO;

        // Apply rotation
        self.spherical.theta += self.spherical_delta.theta * time_scale;
        self.spherical.phi += self.spherical_delta.phi * time_scale;
        self.spherical.phi = self.spherical.phi.clamp(self.min_polar_angle, self.max_polar_angle);
        self.spherical.make_safe();

        // Apply zoom
        if self.enable_damping {
            let lerp_factor = 1.0 - (1.0 - self.zoom_damping_factor).powf(time_scale);
            self.spherical.radius += (self.target_radius - self.spherical.radius) * lerp_factor;
        } else {
            self.spherical.radius = self.target_radius;
        }

        self.spherical.radius = self.spherical.radius.clamp(self.min_distance, self.max_distance);

        // Calculate Transform
        let sin_phi_radius = self.spherical.phi.sin() * self.spherical.radius;
        let offset = Vec3::new(
            sin_phi_radius * self.spherical.theta.sin(),
            self.spherical.radius * self.spherical.phi.cos(),
            sin_phi_radius * self.spherical.theta.cos(),
        );

        transform.position = self.target + offset;
        transform.look_at(self.target, Vec3::Y);

        // Rotation damping decay
        if self.enable_damping {
            let damping = (1.0 - self.damping_factor).powf(time_scale);
            self.spherical_delta.theta *= damping;
            self.spherical_delta.phi *= damping;
        } else {
            self.spherical_delta.theta = 0.0;
            self.spherical_delta.phi = 0.0;
        }
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn set_position(&mut self, position: Vec3) {
        let offset = position - self.target;
        self.spherical.set_from_vec3(offset);
        self.target_radius = self.spherical.radius;
    }
}

impl Transform {
    pub fn rotation_basis(&self) -> (Vec3, Vec3, Vec3) {
        let right = self.rotation * Vec3::X;
        let up = self.rotation * Vec3::Y;
        let forward = self.rotation * Vec3::Z;
        (right, up, forward)
    }
}