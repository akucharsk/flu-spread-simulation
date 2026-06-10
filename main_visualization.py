import argparse
import json

from mesa_visualizer import MesaVisualizer
from model import EpidemicModel


def main():
    parser = argparse.ArgumentParser(description="Run flu spread simulation with live visualization")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args, _ = parser.parse_known_args()

    with open(args.config) as file:
        config = json.load(file)

    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
        verbose=config.get("verbose", False),
    )

    visualizer = MesaVisualizer(
        model=model,
        figsize=tuple(config.get("figsize", (12, 9))),
        agent_size=config.get("agentSize", 20),
    )
    return visualizer


visualizer = main()
Page = visualizer.run()
