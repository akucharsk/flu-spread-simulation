import sys
import argparse
from model import EpidemicModel
from visualization import Visualizer
from states import HealthState


def main():
    parser = argparse.ArgumentParser(description='Flu Spread Simulation')
    parser.add_argument('--steps', type=int, default=None,
                        help='Run simulation for N steps without visualization')
    parser.add_argument('--population', type=int, default=1000,
                        help='Number of agents (default: 1000)')
    parser.add_argument('--width', type=int, default=40,
                        help='Grid width (default: 40)')
    parser.add_argument('--height', type=int, default=40,
                        help='Grid height (default: 40)')
    parser.add_argument('--city-map', type=str, default=None,
                        help='Path to city map file', required=True)
    
    args = parser.parse_args()
    
    model = EpidemicModel(population=args.population, width=args.width, height=args.height)
    
    if args.steps:
        # Run simulation for N steps without visualization
        print(f"Running simulation for {args.steps} steps...")
        for i in range(args.steps):
            model.step()
            
            if (i + 1) % max(1, args.steps // 10) == 0:
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
        visualizer = Visualizer(model, fps=10, radius=10)
        visualizer.run()


if __name__ == "__main__":
    main()
