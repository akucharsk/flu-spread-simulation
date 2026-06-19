"""Procedural generator for the city maps shipped in this folder.

Run from the repo root::

    python maps/_generate_maps.py

The script overwrites the .txt files for every preset in PRESETS, producing
square maps with irregular blob-shaped buildings (no boring 'rows of equal
rectangles') and varying street widths.

Cell encoding matches ``CellType``:
    0  DEFAULT       (asphalt / streets / unused)
    1  HOUSEHOLD
    2  WORKPLACE
    3  PUBLIC_SPACE  (parks, plazas, sidewalks people may roam)
    4  UNIVERSITY    (STUDENT day destination)
    5  SCHOOL        (CHILDREN day destination)

The generator deliberately:
  * separates buildings of different types with at least 1 street cell,
  * gives blob-grown irregular outlines instead of clean rectangles,
  * carves a few wide arteries on bigger maps so visualization shows variety,
  * leaves a 1-cell street border around the whole map for free movement.

Re-running with the same seeds reproduces the exact same maps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

GRID_DIR = Path(__file__).resolve().parent

# Cell-type codes used in the map text files.
DEFAULT = "0"
HOUSEHOLD = "1"
WORKPLACE = "2"
PUBLIC_SPACE = "3"
UNIVERSITY = "4"
SCHOOL = "5"


# ------------------------------------------------------------------
# Generic procedural primitives
# ------------------------------------------------------------------

def make_grid(width: int, height: int) -> list[list[str]]:
    """All-DEFAULT (street) grid."""
    return [[DEFAULT for _ in range(width)] for _ in range(height)]


def _in_bounds(grid: list[list[str]], x: int, y: int) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def _moore_has_other_type(grid: list[list[str]], x: int, y: int, own_type: str) -> bool:
    """True iff one of the 8-neighbours is a building of a *different* type.

    Used to keep distinct buildings separated by at least one street cell.
    """
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nx, ny = x + dx, y + dy
            if _in_bounds(grid, nx, ny):
                v = grid[ny][nx]
                if v != DEFAULT and v != own_type:
                    return True
    return False


def _bfs_grow(
    grid: list[list[str]],
    sx: int, sy: int,
    target_size: int,
    own_type: str,
    cohesion: float = 0.78,
) -> list[tuple[int, int]]:
    """Randomised BFS from (sx, sy) that paints an irregular blob.

    Each candidate neighbour is admitted with probability ``cohesion``, which
    yields organic shapes (sometimes long limbs, sometimes compact lobes).
    Cells that would touch a building of another type are skipped to keep
    streets between distinct buildings.
    """
    cells = [(sx, sy)]
    visited = {(sx, sy)}
    frontier = [(sx, sy)]
    while len(cells) < target_size and frontier:
        idx = random.randrange(len(frontier))
        cx, cy = frontier.pop(idx)
        candidates = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        random.shuffle(candidates)
        for nx, ny in candidates:
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if not _in_bounds(grid, nx, ny):
                continue
            if grid[ny][nx] != DEFAULT:
                continue
            if _moore_has_other_type(grid, nx, ny, own_type):
                continue
            if random.random() < cohesion:
                cells.append((nx, ny))
                frontier.append((nx, ny))
    return cells


def place_building(
    grid: list[list[str]],
    cell_type: str,
    target_size: int,
    *,
    cohesion: float = 0.78,
    min_quality: float = 0.55,
    border: int = 1,
    attempts: int = 250,
) -> bool:
    """Pick a random seed and grow an irregular blob of ``cell_type``.

    Returns True iff a building of at least ``target_size * min_quality``
    cells could be placed.  ``border`` is the minimum distance from the map
    edge for the seed (so the outer ring remains a road).
    """
    height = len(grid)
    width = len(grid[0])
    if height <= 2 * border or width <= 2 * border:
        return False

    for _ in range(attempts):
        sx = random.randrange(border, width - border)
        sy = random.randrange(border, height - border)
        if grid[sy][sx] != DEFAULT:
            continue
        if _moore_has_other_type(grid, sx, sy, cell_type):
            continue
        cells = _bfs_grow(grid, sx, sy, target_size, cell_type, cohesion=cohesion)
        if len(cells) >= max(1, int(target_size * min_quality)):
            for x, y in cells:
                grid[y][x] = cell_type
            return True
    return False


def carve_avenue(
    grid: list[list[str]],
    *,
    horizontal: bool,
    position: int,
    width: int = 2,
) -> None:
    """Erase a straight wide road through the map (sets cells to DEFAULT)."""
    h, w = len(grid), len(grid[0])
    half_lo = width // 2
    half_hi = width - half_lo
    if horizontal:
        for y in range(position - half_lo, position + half_hi):
            if 0 <= y < h:
                for x in range(w):
                    grid[y][x] = DEFAULT
    else:
        for x in range(position - half_lo, position + half_hi):
            if 0 <= x < w:
                for y in range(h):
                    grid[y][x] = DEFAULT


def carve_border(grid: list[list[str]], thickness: int = 1) -> None:
    """Ensure the outer ring is a road, so agents can always walk around."""
    h, w = len(grid), len(grid[0])
    for y in range(h):
        for x in range(w):
            if (
                y < thickness
                or x < thickness
                or y >= h - thickness
                or x >= w - thickness
            ):
                grid[y][x] = DEFAULT


def to_text(grid: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in grid) + "\n"


def write_map(grid: list[list[str]], filename: str) -> None:
    (GRID_DIR / filename).write_text(to_text(grid), encoding="utf-8")


# ------------------------------------------------------------------
# Map presets
# ------------------------------------------------------------------

@dataclass
class BuildingSpec:
    cell_type: str
    count: int
    size_min: int
    size_max: int
    cohesion: float = 0.78


@dataclass
class MapSpec:
    filename: str
    size: int
    seed: int
    buildings: list[BuildingSpec] = field(default_factory=list)
    avenues_h: list[tuple[int, int]] = field(default_factory=list)  # (y, width)
    avenues_v: list[tuple[int, int]] = field(default_factory=list)  # (x, width)


PRESETS: list[MapSpec] = [
    # 1. 30x30 - small academic campus
    MapSpec(
        filename="campus_30.txt",
        size=30,
        seed=101,
        buildings=[
            BuildingSpec(UNIVERSITY,  2, 35, 55, cohesion=0.82),
            BuildingSpec(SCHOOL,      1, 12, 18, cohesion=0.78),
            BuildingSpec(HOUSEHOLD,   6,  6, 11, cohesion=0.75),
            BuildingSpec(WORKPLACE,   2,  9, 14, cohesion=0.78),
            BuildingSpec(PUBLIC_SPACE, 4, 8, 18, cohesion=0.70),
        ],
        avenues_h=[(15, 2)],
    ),

    # 2. 60x60 - suburban town
    MapSpec(
        filename="suburban_town_60.txt",
        size=60,
        seed=202,
        buildings=[
            BuildingSpec(UNIVERSITY,  1, 50, 80, cohesion=0.82),
            BuildingSpec(SCHOOL,      2, 20, 35, cohesion=0.80),
            BuildingSpec(HOUSEHOLD,  30,  5, 12, cohesion=0.78),
            BuildingSpec(WORKPLACE,   6, 12, 25, cohesion=0.78),
            BuildingSpec(PUBLIC_SPACE, 8, 10, 30, cohesion=0.72),
        ],
        avenues_h=[(20, 2), (40, 2)],
        avenues_v=[(30, 2)],
    ),

    # 3. 100x100 - mixed district
    MapSpec(
        filename="mixed_district_100.txt",
        size=100,
        seed=303,
        buildings=[
            BuildingSpec(UNIVERSITY,  2,  80, 130, cohesion=0.84),
            BuildingSpec(SCHOOL,      3,  35,  60, cohesion=0.80),
            BuildingSpec(HOUSEHOLD,  55,   8,  20, cohesion=0.78),
            BuildingSpec(WORKPLACE,  18,  20,  50, cohesion=0.78),
            BuildingSpec(PUBLIC_SPACE, 14, 20,  60, cohesion=0.70),
        ],
        avenues_h=[(33, 2), (66, 2)],
        avenues_v=[(33, 2), (66, 2)],
    ),

    # 4. 150x150 - industrial corridor (lots of work, fewer schools)
    MapSpec(
        filename="industrial_corridor_150.txt",
        size=150,
        seed=404,
        buildings=[
            BuildingSpec(WORKPLACE,  30,  60, 130, cohesion=0.82),
            BuildingSpec(HOUSEHOLD,  90,  10,  22, cohesion=0.78),
            BuildingSpec(UNIVERSITY,  2,  90, 140, cohesion=0.84),
            BuildingSpec(SCHOOL,      4,  40,  70, cohesion=0.80),
            BuildingSpec(PUBLIC_SPACE, 20, 30,  90, cohesion=0.72),
        ],
        avenues_h=[(40, 3), (80, 3), (120, 3)],
        avenues_v=[(50, 3), (100, 3)],
    ),

    # 5. 200x200 - full metropolitan city
    MapSpec(
        filename="megacity_200.txt",
        size=200,
        seed=505,
        buildings=[
            BuildingSpec(UNIVERSITY,   5, 120, 200, cohesion=0.84),
            BuildingSpec(SCHOOL,       8,  50,  90, cohesion=0.80),
            BuildingSpec(HOUSEHOLD,  180,  10,  25, cohesion=0.78),
            BuildingSpec(WORKPLACE,   45,  40, 110, cohesion=0.80),
            BuildingSpec(PUBLIC_SPACE, 30, 30, 120, cohesion=0.72),
        ],
        avenues_h=[(40, 3), (100, 4), (160, 3)],
        avenues_v=[(40, 3), (100, 4), (160, 3)],
    ),
]


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def generate_map(spec: MapSpec) -> list[list[str]]:
    random.seed(spec.seed)
    grid = make_grid(spec.size, spec.size)

    for y, w in spec.avenues_h:
        carve_avenue(grid, horizontal=True, position=y, width=w)
    for x, w in spec.avenues_v:
        carve_avenue(grid, horizontal=False, position=x, width=w)

    for spec_b in spec.buildings:
        for _ in range(spec_b.count):
            target = random.randint(spec_b.size_min, spec_b.size_max)
            place_building(
                grid,
                cell_type=spec_b.cell_type,
                target_size=target,
                cohesion=spec_b.cohesion,
            )

    carve_border(grid, thickness=1)
    return grid


def main() -> None:
    for spec in PRESETS:
        grid = generate_map(spec)
        write_map(grid, spec.filename)
        cells = sum(1 for row in grid for v in row if v != DEFAULT)
        total = spec.size * spec.size
        coverage = cells / total * 100
        print(
            f"  {spec.filename:>30s}  {spec.size}x{spec.size}  "
            f"built coverage = {coverage:5.1f}%"
        )


if __name__ == "__main__":
    main()
