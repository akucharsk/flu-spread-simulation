import argparse
import json

from model import EpidemicModel
from states import HealthState


def main():
    parser = argparse.ArgumentParser(description="Run flu spread simulation headless")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to simulate")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
    )

    print(f"Running simulation for {args.steps} steps...")
    for i in range(args.steps):
        model.step()

        if (i + 1) % max(1, args.steps // 10) == 0:
            infected = sum(1 for a in model.agents if a.health_state == HealthState.INFECTIOUS)
            exposed = sum(1 for a in model.agents if a.health_state == HealthState.EXPOSED)
            recovered = sum(1 for a in model.agents if a.health_state == HealthState.RECOVERED)
            print(f"  Step {i + 1}: Exposed={exposed}, Infected={infected}, Recovered={recovered}")

    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
