from pathlib import Path

_HERE = Path(__file__).resolve().parent
ASSETS_ROOT = _HERE.parent.parent.parent / "examples" / "assets"


def get_asset(relative_path: str) -> str:
    """Get the absolute path of a file in the main repository's assets directory."""
    asset_path = ASSETS_ROOT / relative_path

    if not asset_path.exists():
        raise FileNotFoundError(
            f"Asset not found: {asset_path} \nPlease ensure you have cloned the full myth-engine repository."
        )

    return str(asset_path)
