# Changelog

## Unreleased

### Major Changes
- Add a dedicated `myth_macros` crate for generating Material and GPU data structures. The previous complex declarative macros have been removed, making the creation of Materials and GPU data structs simpler and more ergonomic, with a more user-friendly API.
- Added point light shadows, completing the final piece of the basic lighting system.
- Refactored the asynchronous asset loading system, introducing a “fire-and-forget” style, ergonomic API. All asynchronous loading logic is now fully handled internally by the engine.

### Added
- Added a “debug_view” feature, enabling real-time inspection of material base textures (albedo, metalness, roughness) as well as in-frame buffers (depth, normal, SSAO, velocity, and more).

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