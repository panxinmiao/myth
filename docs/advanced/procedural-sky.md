# Procedural Sky & Atmosphere

Myth includes a **physically-based procedural sky system** using the widely-adopted **Hillaire 2020** atmospheric scattering model. It generates not just realistic sky gradients, but also a procedural sun, moon, and starfield, plus a ready-to-use **DayNightCycle** component.

## 1. Enabling the Procedural Sky

The sky is configured as the scene's **background mode**. The engine provides several presets to quickly get the mood you want:

```rust
use myth::prelude::*;

// Start from the "golden hour" preset
let sky = ProceduralSkyParams::golden_hour();

scene.background.set_mode(BackgroundMode::procedural_with(sky));
```

Once set, the sky acts as an environment lighting source, automatically contributing to PBR material reflection and diffuse — no separate HDR map required.

## 2. Celestial Textures: Sun, Moon & Stars

You can give the moon an albedo texture and overlay a Milky Way panorama on the night sky for added realism:

```rust
let starbox = engine.assets.load_texture(
    "assets/envs/Milky_Way_panorama.jpg",
    ColorSpace::Srgb,
    true,
);
let moon_albedo = engine.assets.load_texture(
    "assets/moon.jpg",
    ColorSpace::Srgb,
    true,
);

let mut sky = ProceduralSkyParams::golden_hour();
sky.set_starbox_texture(starbox);
sky.set_moon_texture(moon_albedo);

scene.background.set_mode(BackgroundMode::procedural_with(sky));
```

## 3. DayNightCycle

The `DayNightCycle` component automatically advances time and synchronizes the trajectories and light directions of the sun, moon, and stars. You just bind directional-light nodes to it:

```rust
// Sun: directional light casting the main shadow
let mut sun_light = Light::new_directional(Vec3::new(1.0, 0.95, 0.8), 3.0);
sun_light.cast_shadows = true;
let sun_node = scene.add_light(sun_light);

// Moon: faint fill light at night
let moon_light = Light::new_directional(Vec3::new(0.62, 0.72, 1.0), 0.12);
let moon_node = scene.add_light(moon_light);

// Build the cycle: start at 16.5h, full day length 35s (for the demo)
let cycle = DayNightCycle::new(16.5, 35.0)
    .with_sun(sun_node)
    .with_moon(moon_node)
    .with_time_speed(0.35); // time flow speed

scene.add_logic(cycle);
```

Once bound, each frame the engine will:

- Compute the azimuth and elevation of the sun / moon from the current time;
- Automatically update the orientation and intensity of both directional lights (sun dims and moon brightens at sunset);
- Sync atmospheric scattering parameters so the sky color transitions naturally over time (blue daytime sky → warm dusk → starry night).

## 4. Cooperating With Other Systems

The procedural sky is designed to integrate seamlessly with the rest of the engine:

- **IBL Environment Lighting:** The sky automatically serves as the environment irradiance source, so PBR reflections faithfully reflect the sky's tone.
- **Shadows:** The sun directional light drives Cascaded Shadow Maps (CSM); shadow direction shifts naturally with the day-night cycle.
- **Post-Processing:** The sky outputs HDR color, handled uniformly by Bloom and Tone Mapping, giving sun glare and other highlights a physically correct look.

## Next Steps

- Configure shadows and lighting → [GPU-Driven Clustered Lighting](/advanced/clustered-shading)
- Tune Bloom and other post effects → [Post-Processing & Screen-Space FX](/advanced/post-processing)
