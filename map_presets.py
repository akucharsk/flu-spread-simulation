from __future__ import annotations

from pathlib import Path

_PRESET_BASE_DIR = Path(__file__).resolve().parent

PREDEFINED_CITY_MAPS = {
    "classic_test_city": {
        "label": "Classic Block City",
        "path": "tests/test_city.txt",
        "description": "Baseline compact city used by the test suite.",
    },
    "campus_30": {
        "label": "Academic Campus (30x30)",
        "path": "maps/campus_30.txt",
        "description": "Small university campus: two big lecture halls, a school and dormitories.",
    },
    "suburban_town_60": {
        "label": "Suburban Town (60x60)",
        "path": "maps/suburban_town_60.txt",
        "description": "Quiet town with scattered houses, two schools, one university and a few workplaces.",
    },
    "mixed_district_100": {
        "label": "Mixed District (100x100)",
        "path": "maps/mixed_district_100.txt",
        "description": "Balanced urban district: residential blocks, offices, schools and parks intermixed.",
    },
    "industrial_corridor_150": {
        "label": "Industrial Corridor (150x150)",
        "path": "maps/industrial_corridor_150.txt",
        "description": "Work-heavy belt of large industrial sites, worker housing and a couple of schools.",
    },
    "megacity_200": {
        "label": "Megacity (200x200)",
        "path": "maps/megacity_200.txt",
        "description": "Full metropolitan area with every infrastructure type and broad arterial roads.",
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
