from __future__ import annotations

from pathlib import Path

_PRESET_BASE_DIR = Path(__file__).resolve().parent

PREDEFINED_CITY_MAPS = {
    "classic_test_city": {
        "label": "Classic Block City",
        "path": "tests/test_city.txt",
        "description": "Baseline compact city used in tests.",
    },
    "downtown_grid": {
        "label": "Downtown Grid",
        "path": "maps/downtown_grid.txt",
        "description": "Dense mixed downtown blocks with alternating zones.",
    },
    "suburban_ring": {
        "label": "Suburban Ring",
        "path": "maps/suburban_ring.txt",
        "description": "Outer residential ring with separated inner activity zones.",
    },
    "campus_hub": {
        "label": "Campus Hub",
        "path": "maps/campus_hub.txt",
        "description": "Centralized education and public-space hubs with radial movement.",
    },
    "industrial_corridor": {
        "label": "Industrial Corridor",
        "path": "maps/industrial_corridor.txt",
        "description": "Long commuting corridors between homes and work-heavy districts.",
    },
}


def get_map_preset_keys() -> list[str]:
    return list(PREDEFINED_CITY_MAPS.keys())


def get_map_label(preset_key: str | None, city_map_path: str | None = None) -> str:
    if preset_key and preset_key in PREDEFINED_CITY_MAPS:
        return PREDEFINED_CITY_MAPS[preset_key]["label"]
    return city_map_path or "Custom Map"


def resolve_city_map_path(city_map_path: str, city_map_preset: str | None = None) -> str:
    if city_map_preset:
        preset = PREDEFINED_CITY_MAPS.get(city_map_preset)
        if preset is None:
            raise ValueError(f"Unknown city_map_preset: {city_map_preset}")
        return str((_PRESET_BASE_DIR / preset["path"]).resolve())
    return city_map_path
