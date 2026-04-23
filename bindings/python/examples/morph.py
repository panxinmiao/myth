"""
Myth Engine - Morph Targets
===========================

Load a glTF model with morph targets, drive its blend-shape weights from Python,
and use the example as a quick runtime check for the Myth Python bindings.

Usage:
    python examples/morph.py [path/to/model.glb] [mesh_node_name]

By default this example loads examples/assets/facecap.glb and animates the first
named mesh node that contains morph targets.
"""

import json
import math
import struct
import sys
from pathlib import Path

import myth

from __asset_utils import get_asset

DEFAULT_MODEL_PATH = get_asset("facecap.glb")
DEFAULT_ENV_PATH = get_asset("envs/royal_esplanade_2k.hdr.jpg")


def load_gltf_document(path: Path) -> dict:
    if path.suffix.lower() == ".glb":
        with path.open("rb") as handle:
            magic, _version, length = struct.unpack("<4sII", handle.read(12))
            if magic != b"glTF":
                raise RuntimeError(f"Not a valid GLB file: {path}")

            while handle.tell() < length:
                chunk_length, chunk_type = struct.unpack("<I4s", handle.read(8))
                chunk_data = handle.read(chunk_length)
                if chunk_type == b"JSON":
                    return json.loads(chunk_data.decode("utf-8"))

        raise RuntimeError(f"GLB file does not contain a JSON chunk: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_morph_mesh_info(model_path: str, preferred_node_name: str | None = None):
    document = load_gltf_document(Path(model_path))
    meshes = document.get("meshes", [])
    candidates = []

    for node in document.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None or mesh_index >= len(meshes):
            continue

        mesh = meshes[mesh_index]
        primitives = mesh.get("primitives", [])
        if not primitives:
            continue

        target_count = len(primitives[0].get("targets", []))
        if target_count == 0:
            continue

        target_names = mesh.get("extras", {}).get("targetNames") or mesh.get(
            "targetNames"
        )
        if not target_names:
            target_names = [f"target_{index}" for index in range(target_count)]

        candidates.append((node.get("name"), list(target_names)))

    if preferred_node_name is not None:
        for node_name, target_names in candidates:
            if node_name == preferred_node_name:
                return node_name, target_names
        raise RuntimeError(
            f"Could not find morph-target mesh node named '{preferred_node_name}' in {model_path}"
        )

    for node_name, target_names in candidates:
        if node_name:
            return node_name, target_names

    if candidates:
        raise RuntimeError(
            "Found morph-target meshes in the glTF asset, but none of them have a node name. "
            "Pass an explicit mesh node name once the binding exposes a traversal API."
        )

    raise RuntimeError(f"No morph-target mesh found in {model_path}")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
REQUESTED_MESH_NODE_NAME = sys.argv[2] if len(sys.argv) > 2 else None
MESH_NODE_NAME, TARGET_NAMES = find_morph_mesh_info(MODEL_PATH, REQUESTED_MESH_NODE_NAME)
TARGET_INDICES = {name: index for index, name in enumerate(TARGET_NAMES)}

app = myth.App(
    title="Myth Engine - Morph Targets",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=True,
)
orbit = myth.OrbitControls(position=[0.0, 0.2, 4.0], target=[0.0, 0.15, 0.0])

model_root = None
morph_node = None
camera = None
fps_accum_time = 0.0
fps_accum_frames = 0


def set_weight(weights: list[float], name: str, value: float) -> None:
    index = TARGET_INDICES.get(name)
    if index is not None and index < len(weights):
        weights[index] = clamp01(value)


@app.init
def on_init(ctx: myth.Engine):
    global model_root, morph_node, camera

    scene = ctx.create_scene()
    model_root = ctx.load_gltf(MODEL_PATH)
    morph_node = scene.find_node_by_name(MESH_NODE_NAME)
    if morph_node is None or morph_node.mesh is None:
        raise RuntimeError(
            f"Loaded '{MODEL_PATH}', but could not resolve morph mesh node '{MESH_NODE_NAME}'"
        )

    if not hasattr(morph_node.mesh, "morph_target_influences"):
        raise RuntimeError(
            "MeshComponent.morph_target_influences is missing at runtime. "
            "Rebuild the Python extension first, for example with: maturin develop --release"
        )

    weights = list(morph_node.mesh.morph_target_influences)
    if not weights:
        raise RuntimeError(f"Mesh node '{MESH_NODE_NAME}' does not expose any morph weights")

    camera = scene.add_camera(
        myth.PerspectiveCamera(
            fov=45,
            near=0.01,
            anti_aliasing=myth.AntiAliasing.fxaa(),
        )
    )
    camera.position = [0.0, 0.2, 4.0]
    camera.look_at([0.0, 0.15, 0.0])
    scene.active_camera = camera

    key = scene.add_light(
        myth.DirectionalLight(
            color=[1.0, 0.97, 0.92],
            intensity=3.0,
            cast_shadows=True,
        )
    )
    key.position = [2.5, 3.5, 3.0]
    key.look_at([0.0, 0.1, 0.0])

    fill = scene.add_light(
        myth.DirectionalLight(
            color=[0.55, 0.65, 0.9],
            intensity=1.1,
        )
    )
    fill.position = [-2.5, 1.5, 1.0]
    fill.look_at([0.0, 0.0, 0.0])

    scene.set_environment_map(ctx.load_texture(DEFAULT_ENV_PATH, color_space="srgb"))
    scene.set_environment_intensity(1.2)
    scene.set_ambient_light(0.02, 0.02, 0.02)
    scene.set_background_color(0.07, 0.08, 0.1)
    scene.set_tone_mapping("aces", exposure=1.0)
    scene.set_bloom(True, strength=0.02)

    animations = scene.list_animations(model_root)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Morph mesh node: {MESH_NODE_NAME}")
    print(f"Morph target count: {len(weights)}")
    if animations:
        print(f"Available animations: {animations}")
        print("This demo leaves clip playback disabled so Python-driven morph weights stay visible.")

    preview = ", ".join(TARGET_NAMES[:8])
    suffix = "..." if len(TARGET_NAMES) > 8 else ""
    print(f"Morph targets: {preview}{suffix}")


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global fps_accum_time, fps_accum_frames

    orbit.update(camera, frame.dt)

    mesh = morph_node.mesh if morph_node is not None else None
    if mesh is None:
        return

    weights = [0.0] * len(TARGET_NAMES)
    t = frame.time

    smile = 0.5 + 0.5 * math.sin(t * 1.1 + 0.5)
    jaw = 0.5 + 0.5 * math.sin(t * 1.7)
    brow = 0.5 + 0.5 * math.sin(t * 0.8 + 1.2)
    blink = clamp01(math.sin(t * 3.4 - 0.3)) ** 10

    set_weight(weights, "jawOpen", jaw)
    set_weight(weights, "mouthSmile_L", smile)
    set_weight(weights, "mouthSmile_R", smile)
    set_weight(weights, "browInnerUp", brow * 0.6)
    set_weight(weights, "eyeBlink_L", blink)
    set_weight(weights, "eyeBlink_R", blink * 0.95)

    if not TARGET_INDICES:
        # If the glTF asset doesn't provide target names, just animate the first few morph targets in a wave pattern.
        for index in range(min(3, len(weights))):
            weights[index] = 0.5 + 0.5 * math.sin(t * (1.0 + index * 0.35) + index)

    # print(f"Updating morph weights: {weights}")
    mesh.morph_target_influences = weights

    fps_accum_time += frame.dt
    fps_accum_frames += 1
    if fps_accum_time >= 0.5:
        fps = fps_accum_frames / fps_accum_time
        ctx.set_title(
            f"Myth Engine - Morph Targets | FPS: {fps:.1f} | Node: {MESH_NODE_NAME}"
        )
        fps_accum_time = 0.0
        fps_accum_frames = 0


app.run()
