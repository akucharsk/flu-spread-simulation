import json

from mesa.visualization import SolaraViz
from mesa_visualizer import MesaVisualizer
from model import EpidemicModel
from states import HealthState


def main():
    with open("config.json") as file:
        config = json.load(file)
    
    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
    )
    
    if config.get("steps"):
        # Run simulation for N steps without visualization
        print(f"Running simulation for {config['steps']} steps...")
        for i in range(config["steps"]):
            model.step()
            
            if (i + 1) % max(1, config["steps"] // 10) == 0:
                infected = sum(
                    1 for agent in model.agents
                    if agent.health_state == HealthState.INFECTIOUS
                )
                exposed = sum(
                    1 for agent in model.agents
                    if agent.health_state == HealthState.EXPOSED
                )
                recovered = sum(
                    1 for agent in model.agents
                    if agent.health_state == HealthState.RECOVERED
                )
                print(f"  Step {i+1}: Exposed={exposed}, Infected={infected}, Recovered={recovered}")
        
        print("✓ Simulation completed successfully")
    else:
        # Run with visualization
        visualizer = MesaVisualizer(
            model=model,
            figsize=tuple(config.get("figsize", (12, 9))),
            agent_size=config.get("agentSize", 20)
        )
        return visualizer


if __name__ == "__main__":
    visualizer = main()
    if visualizer:
        page = visualizer.run()
        page
