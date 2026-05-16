from mesa import Agent
from states import HealthState
import random


class PersonAgent(Agent):

    def __init__(
        self,
        model,
        agent_type,
        infection_rate,
        mobility,
    ):
        super().__init__(model)

        self.agent_type = agent_type
        self.health_state = HealthState.SUSCEPTIBLE

        self.infection_rate = infection_rate
        self.mobility = mobility

        self.days_infected = 0
        
    @property
    def x(self):
        return self.pos[0]
      
    @property
    def y(self):
        return self.pos[1]

    def move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        new_position = random.choice(neighbors)
        self.model.grid.move_agent(self, new_position)

    def interact(self):

        if self.health_state != HealthState.INFECTIOUS:
            return

        nearby_agents = self.model.grid.get_cell_list_contents(
            [self.pos]
        )

        for agent in nearby_agents:

            if agent.health_state == HealthState.SUSCEPTIBLE:

                if random.random() < self.infection_rate:
                    agent.health_state = HealthState.EXPOSED

    def update_health(self):

        if self.health_state == HealthState.EXPOSED:
            if random.random() < 0.2:
                self.health_state = HealthState.INFECTIOUS

        elif self.health_state == HealthState.INFECTIOUS:

            self.days_infected += 1

            if self.days_infected > 10:
                self.health_state = HealthState.RECOVERED

    def step(self):

        if random.random() < self.mobility:
            self.move()

        self.interact()

        self.update_health()