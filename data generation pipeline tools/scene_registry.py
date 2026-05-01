"""Scene registry utilities for batch rendering hand-authored Blender scenes.

The registry file intentionally uses a tiny YAML subset so the tools do not
depend on PyYAML inside Blender or the project virtual environment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES_DIR = REPO_ROOT / "Blender files" / "Scenes"
DEFAULT_REGISTRY_PATH = DEFAULT_SCENES_DIR / "scenes.yaml"
SCENE_ID_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class SceneEntry:
    id: str
    blend: str
    bike: str
    rider: str
    frame_range: tuple[int, int] | None = None
    camera_target: str = "k_handlebar_middle"
    weight: float = 1.0
    notes: str = ""

    @property
    def blend_path(self) -> Path:
        return DEFAULT_SCENES_DIR / self.blend

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["blend_path"] = str(self.blend_path)
        return data


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    result: list[str] = []
    for char in line:
        if char in {"'", '"'}:
            in_quote = None if in_quote == char else char
        if char == "#" and in_quote is None:
            break
        result.append(char)
    return "".join(result).rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_minimal_yaml(path: Path) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_scenes = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped == "scenes:":
            in_scenes = True
            continue
        if not in_scenes:
            continue

        if stripped.startswith("- "):
            if current is not None:
                scenes.append(current)
            current = {}
            item = stripped[2:].strip()
            if item:
                key, value = item.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue

        if current is None:
            raise ValueError(f"Malformed registry line before first scene item: {raw_line}")
        if ":" not in stripped:
            raise ValueError(f"Malformed registry line: {raw_line}")
        key, value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(value)

    if current is not None:
        scenes.append(current)

    return {"scenes": scenes}


def _frame_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("frame_range must be null or a two-element list, e.g. [0, 600]")
    start, end = int(value[0]), int(value[1])
    if end < start:
        raise ValueError("frame_range end must be >= start")
    return (start, end)


def load_scene_registry(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    *,
    validate_files: bool = True,
) -> list[SceneEntry]:
    data = _load_minimal_yaml(registry_path)
    entries: list[SceneEntry] = []
    seen_ids: set[str] = set()

    for raw in data.get("scenes", []):
        missing = {"id", "blend", "bike", "rider"} - set(raw)
        if missing:
            raise ValueError(f"Scene entry is missing required fields: {sorted(missing)}")

        scene_id = str(raw["id"])
        if not SCENE_ID_RE.match(scene_id):
            raise ValueError(f"Invalid scene id '{scene_id}'. Use only lowercase a-z, 0-9, _.")
        if scene_id in seen_ids:
            raise ValueError(f"Duplicate scene id: {scene_id}")
        seen_ids.add(scene_id)

        entry = SceneEntry(
            id=scene_id,
            blend=str(raw["blend"]),
            bike=str(raw["bike"]),
            rider=str(raw["rider"]),
            frame_range=_frame_range(raw.get("frame_range")),
            camera_target=str(raw.get("camera_target") or "k_handlebar_middle"),
            weight=float(raw.get("weight") or 1.0),
            notes=str(raw.get("notes") or ""),
        )
        if entry.weight <= 0:
            raise ValueError(f"Scene '{entry.id}' has non-positive weight: {entry.weight}")
        if validate_files and not entry.blend_path.exists():
            raise FileNotFoundError(f"Scene '{entry.id}' blend file not found: {entry.blend_path}")
        entries.append(entry)

    if not entries:
        raise ValueError(f"No scenes found in registry: {registry_path}")
    return entries


def sample_scenes(
    entries: list[SceneEntry],
    count: int,
    seed: int,
) -> list[tuple[SceneEntry, int]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    weights = [entry.weight for entry in entries]
    sampled: list[tuple[SceneEntry, int]] = []
    for _ in range(count):
        entry = rng.choices(entries, weights=weights, k=1)[0]
        camera_seed = rng.randrange(0, 2**31)
        sampled.append((entry, camera_seed))
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or sample Blender scene registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--sample", type=int, default=0, help="Print N sampled scenes as JSON.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    entries = load_scene_registry(args.registry)
    if args.sample:
        payload = [
            {"scene": entry.to_dict(), "camera_seed": camera_seed}
            for entry, camera_seed in sample_scenes(entries, args.sample, args.seed)
        ]
    else:
        payload = [entry.to_dict() for entry in entries]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
