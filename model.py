from mesa import Model
from mesa.space import MultiGrid

from agents import PersonAgent
import random


class EpidemicModel(Model):

    def __init__(
        self,
        width=20,
        height=20,
        population=100
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)

        self.running = True

        # Create agents
        for _ in range(population):

            agent_type = random.choice([
                "student",
                "worker",
                "senior",
                "healthcare"
            ])

            mobility_map = {
                "student": 0.9,
                "worker": 0.8,
                "senior": 0.3,
                "healthcare": 1.0
            }

            infection_map = {
                "student": 0.2,
                "worker": 0.25,
                "senior": 0.35,
                "healthcare": 0.15
            }

            agent = PersonAgent(
                self,
                agent_type=agent_type,
                infection_rate=infection_map[agent_type],
                mobility=mobility_map[agent_type]
            )

            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)

            self.grid.place_agent(agent, (x, y))

        # Infect one initial agent
        patient_zero = random.choice(list(self.agents))
        patient_zero.health_state = patient_zero.health_state.INFECTIOUS

    def step(self):
        self.agents.shuffle_do("step")
