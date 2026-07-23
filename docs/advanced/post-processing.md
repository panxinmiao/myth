# Post-Processing & Screen-Space FX

Myth's `HighFidelity` render path includes a full suite of modern post-processing and screen-space effects. Every effect participates in Render Graph compilation as a node, and **effects that aren't enabled are automatically culled with zero overhead**.

Most post-processing effects hang off `scene` and can be configured at init time or at runtime.

::: warning Prerequisite
All effects on this page depend on `RenderPath::HighFidelity`. Make sure you selected the high-fidelity render path when creating the `App` — see [Render Paths & Frame Composer](/architecture/rendering-pipeline).
:::

## 1. Bloom

Bloom extracts bright regions of the image and bleeds them outward to create a glow. It works best with emissive PBR materials:

```rust
scene.bloom.set_enabled(true);
scene.bloom.set_strength(0.04);        // strength
scene.bloom.set_radius(0.005);         // bleed radius
scene.bloom.set_karis_average(true);   // Karis average, suppresses fireflies
```

Bloom uses a classic multi-level downsample / upsample pyramid (see the topology in [Render Graph](/architecture/render-graph)), balancing quality and performance.

## 2. SSAO (Screen-Space Ambient Occlusion)

SSAO adds soft darkening to geometry crevices and contact regions, significantly improving the sense of volume and realism:

```rust
scene.ssao.enabled = true;
```

SSAO reuses the depth and normal buffers produced by the Pre Pass, and is blurred before use to remove noise.

## 3. SSR (Screen-Space Reflections)

SSR traces rays in screen space to provide real-time reflections for wet floors, polished metal, and similar surfaces:

```rust
scene.ssr.set_enabled(true);
scene.ssr.set_quality(SsrQuality::Ultra); // Low / Medium / High / Ultra
scene.ssr.set_thickness(0.01);            // thickness threshold, affects intersection
```

::: tip Quality vs Performance
SSR offers multiple quality presets. `Ultra` chases the highest visual fidelity, while `Low` keeps things smooth on constrained platforms like mobile. You can switch at runtime.
:::

## 4. SSGI (Screen-Space Global Illumination)

SSGI approximates indirect lighting (light bounces) in screen space, letting colors bleed between objects (color bleeding) and greatly enhancing realism:

```rust
scene.ssgi.set_enabled(true);
scene.ssgi.set_quality(SsgiQuality::Ultra);
```

## 5. SSSS (Screen-Space Subsurface Scattering)

SSSS simulates light scattering inside translucent materials like skin, wax, and marble — critical for character skin rendering. It depends on Pre Pass normals and Feature ID, performing horizontal / vertical blur passes after opaque shading.

```rust
scene.ssss.set_enabled(true);
```

## 6. Anti-Aliasing: TAA / FXAA / MSAA

The engine provides multiple anti-aliasing options:

| Method | Characteristics |
| :--- | :--- |
| **TAA** (Temporal) | Uses history frames + velocity buffer for the highest quality while suppressing high-frequency flicker; paired with CAS sharpening to restore clarity |
| **FXAA** (Fast Approximate) | Single-frame post-process, extremely cheap, used as a final edge smoothing |
| **MSAA** (Multi-Sample) | Hardware-level geometric edge anti-aliasing |

TAA is the recommended method for the high-fidelity path. It reads/writes history color/depth buffers in its resolve stage and reprojects via the velocity buffer.

## 7. HDR, Tone Mapping & Color Grading

The entire pipeline runs in **HDR linear color space**. Before final output, the Tone Mapping stage compresses the high dynamic range into the display's presentable range and applies color grading. This ensures HDR effects like Bloom, emissive, and transmission blend in a physically correct way.

## Performance Philosophy

Thanks to the SSA Render Graph, these effects are **pay-as-you-go**:

- Turn off an effect → the compiler automatically culls its node and any predecessor that exists solely to serve it (dead-pass elimination).
- Intermediate textures shared across effects → the compiler aliases them automatically, with zero VRAM waste.

So you can freely toggle effects per scene without worrying about residual overhead. See [Render Graph](/architecture/render-graph).

## Next Steps

- Want to insert fully custom post-processing? → [Custom Shaders & Post FX](/advanced/custom-shader)
- Configure emissive materials → [PBR Materials](/advanced/pbr-materials)
