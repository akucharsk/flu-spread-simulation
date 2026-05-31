from mesa.space import MultiGrid
import random

def load_city_map(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    grid = [list(line) for line in lines]

    width = len(grid[0])
    height = len(grid)

    return grid, width, height


def build_city_grid(model, city_map):
    grid_data, width, height = city_map

    grid = MultiGrid(width, height, torus=False)

    model.cell_types = {
        "1": "household",
        "2": "workplace",
        "3": "public",
        "0": "empty",
    }
    model.cell_type_ids = {
        t: id for id, t in model.cell_types.items()
    }

    model.location_types = {}

    for y, row in enumerate(grid_data):
        for x, cell in enumerate(row):
            cell_type = model.cell_types[cell]

            model.location_types[(x, y)] = cell_type

    return grid
