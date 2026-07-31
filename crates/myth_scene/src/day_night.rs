//! Day/night cycle scene logic.
//!
//! Keeps procedural sky parameters and optional directional light nodes in
//! sync from a shared notion of latitude and solar time.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

use myth_core::NodeHandle;
use myth_resources::Input;

use crate::light::{LIGHT_FLAG_IS_MOON, LIGHT_FLAG_IS_SUN};
use crate::scene::{Scene, SceneLogic};

/// Scene logic that drives a procedural sky and optional sun/moon lights.
///
/// The current implementation assumes an equinox sun path, which keeps the
/// API compact while still producing physically coherent daily motion.
/// Its local astronomical frame maps north to world `-Z` and south to world
/// `+Z`; world `+X` points east and `+Y` points toward the zenith.
#[derive(Debug, Clone)]
pub struct DayNightCycle {
    /// Local solar time in hours.
    pub time_of_day: f32,
    /// Total days elapsed since the start of the simulation.
    pub day_count: f32,
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
            day_count: 14.75, // Start at a full moon phase by default
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
        self.time_speed = time_speed;
        self
    }

    /// Sets the current day count, which drives moon phase and sidereal drift.
    #[must_use]
    pub fn with_day_count(mut self, day_count: f32) -> Self {
        self.day_count = day_count.max(0.0);
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
        let latitude = self.latitude.to_radians();

        let lunar_progress = self.day_count / 29.5;

        let moon_hour_angle = self.solar_hour_angle() - (lunar_progress * std::f32::consts::TAU);

        Vec3::new(
            -moon_hour_angle.sin(),
            latitude.cos() * moon_hour_angle.cos(),
            latitude.sin() * moon_hour_angle.cos(),
        )
        .normalize_or_zero()
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
        let solar_rotation = wrap_time_of_day(self.time_of_day) / 24.0 * TAU;
        let sidereal_drift = (self.day_count / 365.25) * TAU;

        solar_rotation + sidereal_drift
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
    fn moon_phase_fraction(sun_direction: Vec3, moon_direction: Vec3) -> f32 {
        (1.0 - sun_direction.dot(moon_direction).clamp(-1.0, 1.0)) * 0.5
    }

    #[must_use]
    fn moon_light_intensity(&self, sun_direction: Vec3, moon_direction: Vec3) -> f32 {
        let phase_fraction = Self::moon_phase_fraction(sun_direction, moon_direction);
        let moon_above_horizon = smoothstep(-0.08, 0.04, moon_direction.y);
        let night_factor = 1.0 - smoothstep(-0.12, 0.04, sun_direction.y);
        self.moon_night_intensity * phase_fraction * moon_above_horizon * night_factor
    }

    fn update_bound_light(
        scene: &mut Scene,
        handle: Option<NodeHandle>,
        direction_to_body: Vec3,
        distance: f32,
        intensity: f32,
        celestial_flag: u32,
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
            light.flags &= !(LIGHT_FLAG_IS_SUN | LIGHT_FLAG_IS_MOON);
            light.flags |= celestial_flag;
            light.cast_shadows = light.intensity > 0.001;
        }
    }
}

impl SceneLogic for DayNightCycle {
    fn update(&mut self, scene: &mut Scene, _input: &Input, dt: f32) {
        if self.auto_tick {
            let delta_hours = dt * self.time_speed;
            self.day_count += delta_hours / 24.0;

            self.time_of_day = wrap_time_of_day(self.time_of_day + delta_hours);
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
            LIGHT_FLAG_IS_SUN,
        );
        Self::update_bound_light(
            scene,
            self.moon_light_handle,
            moon_direction,
            self.moon_distance,
            self.moon_light_intensity(sun_direction, moon_direction),
            LIGHT_FLAG_IS_MOON,
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
        assert!(sun.z > 0.0);
    }

    #[test]
    fn north_pole_star_axis_points_up() {
        let cycle = DayNightCycle::new(0.0, 90.0);
        let axis = cycle.compute_star_axis();
        assert!(axis.y > 0.999);
        assert!(axis.z.abs() < 1e-4);
    }

    #[test]
    fn moon_phase_fraction_tracks_new_quarter_and_full_moon() {
        let sun_direction = Vec3::Y;

        let new_moon = DayNightCycle::moon_phase_fraction(sun_direction, sun_direction);
        let quarter_moon = DayNightCycle::moon_phase_fraction(sun_direction, Vec3::X);
        let full_moon = DayNightCycle::moon_phase_fraction(sun_direction, -sun_direction);

        assert!(new_moon <= 1e-6);
        assert!((quarter_moon - 0.5).abs() <= 1e-6);
        assert!((full_moon - 1.0).abs() <= 1e-6);
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

        let (sun_position_len, sun_intensity, sun_flags, sun_cast_shadows) = {
            let (sun_light, sun_node) = scene.get_light_bundle(sun).expect("sun light missing");
            (
                sun_node.transform.position.length(),
                sun_light.intensity,
                sun_light.flags,
                sun_light.cast_shadows,
            )
        };
        let (moon_position_len, moon_intensity, moon_flags, moon_cast_shadows) = {
            let (moon_light, moon_node) = scene.get_light_bundle(moon).expect("moon light missing");
            (
                moon_node.transform.position.length(),
                moon_light.intensity,
                moon_light.flags,
                moon_light.cast_shadows,
            )
        };

        assert!(sun_position_len > 0.0);
        assert!(moon_position_len > 0.0);
        assert!(sun_intensity >= 0.0);
        assert!(moon_intensity >= 0.0);
        assert_ne!(sun_flags & LIGHT_FLAG_IS_SUN, 0);
        assert_eq!(sun_flags & LIGHT_FLAG_IS_MOON, 0);
        assert_ne!(moon_flags & LIGHT_FLAG_IS_MOON, 0);
        assert_eq!(moon_flags & LIGHT_FLAG_IS_SUN, 0);
        assert_eq!(sun_cast_shadows, sun_intensity > 0.001);
        assert_eq!(moon_cast_shadows, moon_intensity > 0.001);
    }
}
