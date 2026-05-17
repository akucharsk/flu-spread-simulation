from mesa import Model
from mesa.space import MultiGrid

from agents import PersonAgent
from agent_types import AgentType, AGENT_CONFIG
import random


class EpidemicModel(Model):

    def __init__(
        self,
        width=20,
        height=20,
        population=100,
        avg_household_size=3
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)
        self.avg_household_size = avg_household_size

        self.running = True

        # Create agents
        agents_list = []
        households = []  # List to track household groups
        
        for i in range(population):
            agent_type = random.choice(list(AgentType))
            params = AGENT_CONFIG[agent_type]
            
            agent = PersonAgent(
                self,
                agent_type=agent_type,
                infection_rate=params.infection_rate,
                mobility=params.mobility
            )
            
            agents_list.append(agent)
            
            # Randomly place agent on grid
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Create family groups (social network - static links)
        self._create_households(agents_list)

        # Infect one initial agent
        patient_zero = random.choice(agents_list)
        patient_zero.health_state = patient_zero.health_state.INFECTIOUS

    def _create_households(self, agents_list):
        """
        Create family groups (households) where members have higher
        interaction risk (static social network links).
        """
        household_id = 0
        agents_copy = agents_list.copy()
        random.shuffle(agents_copy)
        
        while agents_copy:
            # Random household size around average
            household_size = max(1, int(random.gauss(self.avg_household_size, 1)))
            household_size = min(household_size, len(agents_copy))
            
            # Create household
            household = agents_copy[:household_size]
            agents_copy = agents_copy[household_size:]
            
            # Link household members as family
            for agent in household:
                agent.household_id = household_id
                agent.family_members = [a for a in household if a != agent]
            
            household_id += 1

    def step(self):
        self.agents.shuffle_do("step")
