from __future__ import annotations

from pathlib import Path

_PRESET_BASE_DIR = Path(__file__).resolve().parent

PREDEFINED_CITY_MAPS = {
    "classic_city": {
        "label": "Classic City",
        "path": "maps/city.txt",
        "description": "Baseline compact city used in tests.",
    },
    "metropolis": {
        "label": "Metropolis",
        "path": "maps/metropolis.txt",
        "description": "Large urban area with diverse districts and high population density.",
    },
    "campus_hub": {
        "label": "Campus Hub",
        "path": "maps/campus_hub.txt",
        "description": "Centralized education and public-space hubs with radial movement.",
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
