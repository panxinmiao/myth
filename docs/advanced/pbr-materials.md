# PBR Materials

Myth ships with a robust **Physically Based Rendering (PBR)** material pipeline. `PhysicalMaterial` supports not only the standard metallic-roughness workflow, but also several advanced surface features from glTF extensions: Clearcoat, Iridescence, Transmission, Sheen, and Anisotropy.

## 1. The Basic Workflow

`PhysicalMaterial` uses a chainable builder API. Start from a base color and layer on physical properties:

```rust
use myth::prelude::*;

let material = engine.assets.materials.add(
    PhysicalMaterial::new(Vec4::new(0.8, 0.1, 0.1, 1.0)) // base color (RGBA)
        .with_metallic(1.0)      // metallic [0, 1]
        .with_roughness(0.25)    // roughness [0, 1]
);
```

::: tip Intuition for Metallic / Roughness
- **Metallic:** 0 means dielectric (plastic, wood, skin); 1 means pure metal.
- **Roughness:** 0 means mirror-smooth; 1 means fully diffuse.
:::

## 2. Emissive & Texture Maps

```rust
let material = PhysicalMaterial::new(Vec4::new(0.1, 0.1, 0.1, 1.0))
    .with_roughness(0.18)
    // Emissive: color + intensity, participates in Bloom
    .with_emissive(Vec3::new(0.25, 0.86, 1.0), 2.2)
    // Albedo map
    .with_map(albedo_tex);
```

Emissive materials combined with [Bloom](/advanced/post-processing) produce glowing effects like neon signs and energy cores.

## 3. Advanced Surface Features

The following features correspond to official glTF 2.0 extensions and combine freely with the base properties:

### Clearcoat

Adds a transparent coating layer on top of the base material, simulating car paint, varnished wood, etc.:

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.22)
    .with_clearcoat(1.0, 0.04); // (clearcoat strength, clearcoat roughness)
```

### Sheen

Simulates the soft rim highlight of fabrics like velvet and silk:

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.92)
    .with_sheen(Vec3::new(0.85, 0.42, 0.96), 0.38); // (sheen color, sheen roughness)
```

### Iridescence

Simulates thin-film interference colors seen in soap bubbles, beetle shells, and oil films:

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.08)
    .with_iridescence(1.0, 1.3, 120.0, 900.0);
    //               (strength, film IOR, min thickness nm, max thickness nm)
```

### Anisotropy

Simulates directional highlights on brushed metal, hair, etc.:

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.18)
    .with_anisotropy(0.95, std::f32::consts::PI * 0.35); // (strength, rotation)
```

### Transmission & Dispersion

Implements refractive transparent materials like glass, gems, and liquids:

```rust
PhysicalMaterial::new(base_color)
    .with_ior(1.52)          // index of refraction
    .with_roughness(0.03)
    .with_transmission(1.0, 0.55, 4.0, Vec3::new(0.96, 0.98, 1.0));
    //                (transmission, thickness, dispersion, absorption/attenuation color)
```

::: warning Transmission Requires the High-Fidelity Path
Transmissive materials read a refraction background buffer of scene color, depending on the `HighFidelity` render path and its transmission copy/mipmap stage. See [Render Paths & Frame Composer](/architecture/rendering-pipeline).
:::

## 4. Other Built-in Materials

Besides `PhysicalMaterial`, the engine provides lighter-weight material types:

| Material | Purpose |
| :--- | :--- |
| `PhysicalMaterial` | Full PBR, the go-to for production rendering |
| `PhongMaterial` | Classic Phong shading, lower cost, good for prototypes / low-end devices |
| `UnlitMaterial` | No lighting, outputs color / texture directly, ideal for UI, debugging, skyboxes |

```rust
// Unaffected by lighting, displays the texture directly
let mat = UnlitMaterial::new(Vec4::ONE).with_map(tex_handle);
```

## Next Steps

- See materials under environment lighting → [Procedural Sky & Atmosphere](/advanced/procedural-sky)
- Want fully custom materials? → [Custom Shaders & Post FX](/advanced/custom-shader)
- Understand the underlying strongly-typed design → [Material System](/architecture/material-system)
