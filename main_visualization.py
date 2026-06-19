"""Interactive entry point for the flu spread simulation.

Launches a local Pygame window with the model running live.
"""
import argparse
import json

from model import EpidemicModel
from pygame_visualizer import PygameVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Run flu spread simulation with a live Pygame visualization",
    )
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args, _ = parser.parse_known_args()

    with open(args.config) as file:
        config = json.load(file)

    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        city_map_preset=config.get("cityMapPreset"),
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
        verbose=config.get("verbose", False),
    )

    window_size = tuple(config.get("windowSize", (1600, 900)))
    visualizer = PygameVisualizer(
        model=model,
        figsize=tuple(config.get("figsize", (12, 9))),
        agent_size=config.get("agentSize", 6),
        window_size=window_size,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
