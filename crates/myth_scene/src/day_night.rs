//! Day/night cycle scene logic.
//!
//! Keeps procedural sky parameters and optional directional light nodes in
//! sync from a shared notion of latitude and solar time.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

use myth_core::NodeHandle;
use myth_resources::Input;

use crate::scene::{Scene, SceneLogic};

/// Scene logic that drives a procedural sky and optional sun/moon lights.
///
/// The current implementation assumes an equinox sun path, which keeps the
/// API compact while still producing physically coherent daily motion.
#[derive(Debug, Clone)]
pub struct DayNightCycle {
    /// Local solar time in hours.
    pub time_of_day: f32,
    /// Observer latitude in degrees (`-90..=90`).
    pub latitude: f32,
    /// Whether time advances automatically in `update()`.
    pub auto_tick: bool,
    /// Hour progression speed, in in-world hours per real-time second.
    pub time_speed: f32,
    /// Optional directional light node driven as the sun.
    pub sun_light_handle: Option<NodeHandle>,
    /// Optional directional light node driven as the moon.
    pub moon_light_handle: Option<NodeHandle>,
    /// Virtual distance used to place the sun light node before orienting it.
    pub sun_distance: f32,
    /// Virtual distance used to place the moon light node before orienting it.
    pub moon_distance: f32,
    /// Peak intensity applied to the bound sun light.
    pub sun_day_intensity: f32,
    /// Peak intensity applied to the bound moon light.
    pub moon_night_intensity: f32,
}

impl DayNightCycle {
    /// Creates a new day/night controller.
    #[must_use]
    pub fn new(time_of_day: f32, latitude: f32) -> Self {
        Self {
            time_of_day: wrap_time_of_day(time_of_day),
            latitude: latitude.clamp(-90.0, 90.0),
            auto_tick: true,
            time_speed: 0.25,
            sun_light_handle: None,
            moon_light_handle: None,
            sun_distance: 10.0,
            moon_distance: 10.0,
            sun_day_intensity: 3.0,
            moon_night_intensity: 0.12,
        }
    }

    /// Binds the scene node used as the sun light.
    #[must_use]
    pub fn with_sun(mut self, handle: NodeHandle) -> Self {
        self.sun_light_handle = Some(handle);
        self
    }

    /// Binds the scene node used as the moon light.
    #[must_use]
    pub fn with_moon(mut self, handle: NodeHandle) -> Self {
        self.moon_light_handle = Some(handle);
        self
    }

    /// Sets the simulation speed in hours per second.
    #[must_use]
    pub fn with_time_speed(mut self, time_speed: f32) -> Self {
        self.time_speed = time_speed.max(0.0);
        self
    }

    /// Enables or disables automatic time stepping.
    #[must_use]
    pub fn with_auto_tick(mut self, auto_tick: bool) -> Self {
        self.auto_tick = auto_tick;
        self
    }

    /// Computes the normalized world-space direction toward the sun.
    #[must_use]
    pub fn compute_sun_direction(&self) -> Vec3 {
        let latitude = self.latitude.to_radians();
        let hour_angle = self.solar_hour_angle();
        Vec3::new(
            -hour_angle.sin(),
            latitude.cos() * hour_angle.cos(),
            latitude.sin() * hour_angle.cos(),
        )
        .normalize_or_zero()
    }

    /// Computes the normalized world-space direction toward the moon.
    #[must_use]
    pub fn compute_moon_direction(&self) -> Vec3 {
        (-self.compute_sun_direction()).normalize_or_zero()
    }

    /// Computes the celestial pole axis used to rotate the star field.
    #[must_use]
    pub fn compute_star_axis(&self) -> Vec3 {
        let latitude = self.latitude.to_radians();
        Vec3::new(0.0, latitude.sin(), -latitude.cos()).normalize_or_zero()
    }

    /// Computes the star-field rotation angle in radians.
    #[must_use]
    pub fn compute_star_rotation_angle(&self) -> f32 {
        wrap_time_of_day(self.time_of_day) / 24.0 * TAU
    }

    #[must_use]
    fn solar_hour_angle(&self) -> f32 {
        wrap_time_of_day(self.time_of_day) / 24.0 * TAU - PI
    }

    #[must_use]
    fn sun_light_intensity(&self, sun_direction: Vec3) -> f32 {
        self.sun_day_intensity * smoothstep(-0.08, 0.04, sun_direction.y)
    }

    #[must_use]
    fn moon_light_intensity(&self, sun_direction: Vec3, moon_direction: Vec3) -> f32 {
        let moon_above_horizon = smoothstep(-0.08, 0.04, moon_direction.y);
        let night_factor = 1.0 - smoothstep(-0.12, 0.04, sun_direction.y);
        self.moon_night_intensity * moon_above_horizon * night_factor
    }

    fn update_bound_light(
        scene: &mut Scene,
        handle: Option<NodeHandle>,
        direction_to_body: Vec3,
        distance: f32,
        intensity: f32,
    ) {
        let Some(handle) = handle else {
            return;
        };

        if let Some((light, node)) = scene.get_light_bundle(handle) {
            node.transform.position = direction_to_body * distance.max(0.001);
            let up = if direction_to_body.y.abs() > 0.999 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            node.transform.look_at(Vec3::ZERO, up);
            light.intensity = intensity.max(0.0);
        }
    }
}

impl SceneLogic for DayNightCycle {
    fn update(&mut self, scene: &mut Scene, _input: &Input, dt: f32) {
        if self.auto_tick {
            self.time_of_day = wrap_time_of_day(self.time_of_day + dt * self.time_speed);
        }

        let sun_direction = self.compute_sun_direction();
        let moon_direction = self.compute_moon_direction();
        let star_axis = self.compute_star_axis();
        let star_rotation = self.compute_star_rotation_angle();

        if let Some(params) = scene.background.procedural_sky_params_mut() {
            params.set_sun_direction(sun_direction);
            params.set_moon_direction(moon_direction);
            params.set_star_axis(star_axis);
            params.set_star_rotation(star_rotation);
        }

        Self::update_bound_light(
            scene,
            self.sun_light_handle,
            sun_direction,
            self.sun_distance,
            self.sun_light_intensity(sun_direction),
        );
        Self::update_bound_light(
            scene,
            self.moon_light_handle,
            moon_direction,
            self.moon_distance,
            self.moon_light_intensity(sun_direction, moon_direction),
        );
    }
}

fn wrap_time_of_day(time_of_day: f32) -> f32 {
    time_of_day.rem_euclid(24.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::background::BackgroundMode;
    use crate::light::Light;
    use crate::scene::Scene;

    #[test]
    fn equatorial_noon_points_to_zenith() {
        let cycle = DayNightCycle::new(12.0, 0.0);
        let sun = cycle.compute_sun_direction();
        assert!(sun.y > 0.999);
    }

    #[test]
    fn mid_latitude_noon_points_south() {
        let cycle = DayNightCycle::new(12.0, 45.0);
        let sun = cycle.compute_sun_direction();
        assert!(sun.y > 0.0);
        assert!(sun.z < 0.0);
    }

    #[test]
    fn north_pole_star_axis_points_up() {
        let cycle = DayNightCycle::new(0.0, 90.0);
        let axis = cycle.compute_star_axis();
        assert!(axis.y > 0.999);
        assert!(axis.z.abs() < 1e-4);
    }

    #[test]
    fn update_syncs_procedural_background_and_lights() {
        let mut scene = Scene::new();
        scene.background.set_mode(BackgroundMode::procedural());

        let sun = scene.add_light(Light::new_directional(Vec3::ONE, 0.0));
        let moon = scene.add_light(Light::new_directional(Vec3::ONE, 0.0));

        let mut cycle = DayNightCycle::new(18.0, 35.0).with_sun(sun).with_moon(moon);
        cycle.update(&mut scene, &Input::default(), 0.0);

        let params = scene
            .background
            .procedural_sky_params()
            .expect("procedural background expected");
        assert!(params.star_axis.length_squared() > 0.99);
        assert!(params.sun_direction.length_squared() > 0.99);
        assert!(params.moon_direction.length_squared() > 0.99);

        let (sun_light, sun_node) = scene.get_light_bundle(sun).expect("sun light missing");
        assert!(sun_node.transform.position.length() > 0.0);
        assert!(sun_light.intensity >= 0.0);
    }
}
