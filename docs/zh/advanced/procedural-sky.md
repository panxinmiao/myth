# 程序化天空与大气

Myth 内置了一套**基于物理的程序化天空系统**，采用业界广泛使用的 **Hillaire 2020** 大气散射模型。它不仅能生成逼真的天空渐变，还提供程序化的太阳、月亮与星空，以及开箱即用的**昼夜循环 (DayNightCycle)** 组件。

## 1. 启用程序化天空

天空作为场景的**背景模式 (Background Mode)** 进行配置。引擎提供了若干预设，方便你快速获得理想的氛围：

```rust
use myth::prelude::*;

// 从“黄金时刻”预设出发
let sky = ProceduralSkyParams::golden_hour();

scene.background.set_mode(BackgroundMode::procedural_with(sky));
```

设置后，天空会作为环境光照来源，自动参与 PBR 材质的反射与漫反射，无需额外配置 HDR 贴图。

## 2. 天体纹理：太阳、月亮与星空

你可以为月亮贴上反照率纹理、为夜空叠加银河全景图，让天体更加真实：

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

## 3. 昼夜循环 DayNightCycle

`DayNightCycle` 组件会自动推进时间，并同步太阳 / 月亮 / 星辰的轨迹与光照方向。你只需把方向光节点绑定到它：

```rust
// 太阳：投射主阴影的方向光
let mut sun_light = Light::new_directional(Vec3::new(1.0, 0.95, 0.8), 3.0);
sun_light.cast_shadows = true;
let sun_node = scene.add_light(sun_light);

// 月亮：夜间的微弱补光
let moon_light = Light::new_directional(Vec3::new(0.62, 0.72, 1.0), 0.12);
let moon_node = scene.add_light(moon_light);

// 构建昼夜循环：起始 16.5 时，一天总时长 35 秒（演示用）
let cycle = DayNightCycle::new(16.5, 35.0)
    .with_sun(sun_node)
    .with_moon(moon_node)
    .with_time_speed(0.35); // 时间流速

scene.add_logic(cycle);
```

绑定后，引擎会在每帧：

- 根据当前时刻计算太阳 / 月亮的方位角与高度角；
- 自动更新两盏方向光的朝向与强度（日落时太阳渐弱、月亮渐亮）；
- 同步大气散射参数，使天空颜色随时间自然过渡（白昼蓝天 → 黄昏暖色 → 夜晚星空）。

## 4. 与其他系统的协作

程序化天空被设计为与引擎其余部分无缝协作：

- **IBL 环境光照：** 天空自动作为环境辐照度来源，PBR 材质的反射会真实反映天空色调。
- **阴影：** 太阳方向光驱动级联阴影 (CSM)，阴影方向随昼夜自然变化。
- **后处理：** 天空输出 HDR 颜色，统一交由 Bloom 与 Tone Mapping 处理，太阳光晕等高光得到物理正确的呈现。

## 下一步

- 配置阴影与光照 → [GPU-Driven 与聚类光照](/zh/advanced/clustered-shading)
- 调整 Bloom 等后处理 → [后处理与屏幕空间特效](/zh/advanced/post-processing)
