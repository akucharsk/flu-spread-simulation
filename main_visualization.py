import argparse
import json

from mesa_visualizer import MesaVisualizer
from model import EpidemicModel


def main():

    with open("config.json") as file:
        config = json.load(file)

    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
    )

    visualizer = MesaVisualizer(
        model=model,
        figsize=tuple(config.get("figsize", (12, 9))),
        agent_size=config.get("agentSize", 20),
    )
    return visualizer


if __name__ == "__main__":
    visualizer = main()
    page = visualizer.run()
    page
