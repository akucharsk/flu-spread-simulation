import random
import argparse

def create_empty_grid(width, height):
    return [["0" for _ in range(width)] for _ in range(height)]


def in_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


def add_cluster(grid, x, y, size, value, width, height):
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            if random.random() < 0.7:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny, width, height):
                    grid[ny][nx] = value


def generate_city(width, height, household_clusters, workplace_clusters, public_clusters):
    grid = create_empty_grid(width, height)

    for _ in range(household_clusters):
        x = random.randint(5, width - 6)
        y = random.randint(5, height - 6)
        add_cluster(grid, x, y, size=2, value="1", width=width, height=height)

    for _ in range(workplace_clusters):
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)
        add_cluster(grid, x, y, size=1, value="2", width=width, height=height)

    for _ in range(public_clusters):
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)
        add_cluster(grid, x, y, size=2, value="3", width=width, height=height)

    return grid


def save_city(grid, filename):
    with open(filename, "w") as f:
        for row in grid:
            f.write("".join(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a procedural city map")

    parser.add_argument("--width", type=int, default=100, help="Grid width")
    parser.add_argument("--height", type=int, default=60, help="Grid height")

    parser.add_argument("--household_clusters", type=int, default=25, help="Number of cluster type 1")
    parser.add_argument("--workplace_clusters", type=int, default=12, help="Number of cluster type 2")
    parser.add_argument("--public_clusters", type=int, default=10, help="Number of cluster type 3")

    parser.add_argument("--output", type=str, default="city.txt",
                        help="Output filename")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    city = generate_city(args.width, args.height, args.household_clusters, args.workplace_clusters, args.public_clusters)
    save_city(city, args.output)

    print(f"City map generated: {args.output}")
