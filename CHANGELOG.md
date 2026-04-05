# Changelog

## Unreleased

### Major Changes
- Add a dedicated `myth_macros` crate for generating Material and GPU data structures. The previous complex declarative macros have been removed, making the creation of Materials and GPU data structs simpler and more ergonomic, with a more user-friendly API.
- Refactored the asynchronous asset loading system, introducing a “fire-and-forget” style, ergonomic API. All asynchronous loading logic is now fully handled internally by the engine.
- **Headless Rendering Mode**: Added support for offscreen rendering without a window surface (`Renderer.init_headless`). Ideal for server-side rendering, CI/CD testing, and offline video/image generation.
  - **Synchronous GPU Readback**: Introduced `Renderer.readback_pixels()` for simple, one-shot synchronous GPU-to-CPU pixel data extraction.
  - **High-Throughput Asynchronous Readback Stream**: Implemented `ReadbackStream`, a non-blocking ring-buffer pipeline for continuous frame readback. Designed for extreme performance in video recording and AI training data generation without stalling the GPU pipeline.

### Added
- Added point light shadows, completing the final piece of the basic lighting system.
- Added a “debug_view” feature, enabling real-time inspection of material base textures (albedo, metalness, roughness) as well as in-frame buffers (depth, normal, SSAO, velocity, and more).
- `AssetServer::load_lut_texture` now supports both `.cube` and `.bin` LUT files, with automatic format detection based on file extension.

### Changes
- Updated egui to 0.34.1.

## v0.1.1

Released 2021-03-26

#### Changes
- Use `ehttp` instead of `reqwest`.
- Release python bindings package (`myth-py`) on PyPI.
- Update documentation.

## v0.1.0

Released 2026-03-25

#### Added
- First release of `Myth Engine`.

## Diffs

- [Unreleased](https://github.com/panxinmiao/myth/compare/0.1.1...HEAD)
- [v0.1.1](https://github.com/panxinmiao/myth/compare/0.1.0...0.1.1)
- [v0.1.0](https://github.com/panxinmiao/myth/compare/0.0.1...0.1.0)