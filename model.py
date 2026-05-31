from mesa import Model
from mesa.space import MultiGrid, PropertyLayer

from agents import PersonAgent
from agent_types import AgentType, AGENT_CONFIG
import random

from city_utils import build_city_grid, load_city_map

class EpidemicModel(Model):

    def __init__(
        self,
        city_map_path="city1.txt",
        population=2000,
        avg_household_size=3,
        time_of_day=10, # 10:00
        timestep=0.5 # hour
    ):
        super().__init__()
        self.grid = build_city_grid(self, load_city_map(city_map_path))
        self.width = self.grid.width
        self.height = self.grid.height
        self.avg_household_size = avg_household_size

        self.running = True
        self.time_of_day = time_of_day
        self.timestep = timestep

        # Create agents
        agents_list = []
        
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
            
            self.place_agent_in_city(agent)

        # Create family groups (social network - static links)
        self._create_households(agents_list)

        # Infect one initial agent
        patient_zero = random.choice(agents_list)
        patient_zero.health_state = patient_zero.health_state.INFECTIOUS
        
        property_layer = PropertyLayer(
            "cell_type",
            self.width,
            self.height,
            "0"
        )
        
        for pos, t in self.location_types.items():
            property_layer.set_cell(pos, self.cell_type_ids[t])
        self.grid.add_property_layer(property_layer)

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
            
    def place_agent_in_city(self, agent):
        household_cells = [
            pos for pos, t in self.location_types.items()
            if t == "household"
        ]

        workplace_cells = [
            pos for pos, t in self.location_types.items()
            if t == "workplace"
        ]

        public_cells = [
            pos for pos, t in self.location_types.items()
            if t == "public"
        ]

        r = random.random()

        if r < 0.6:
            pos = random.choice(household_cells)
        elif r < 0.9:
            pos = random.choice(workplace_cells)
        else:
            pos = random.choice(public_cells)

        self.grid.place_agent(agent, pos)
    

    def step(self):
        self.agents.shuffle_do("step")
