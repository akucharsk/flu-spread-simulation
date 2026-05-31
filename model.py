from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
import numpy as np

from agents import PersonAgent
from agent_types import AgentType, AGENT_CONFIG
import random

from city_utils import build_city_grid, load_city_map
from states import CellType

class EpidemicModel(Model):

    def __init__(
        self,
        city_map_path="city2.txt",
        population=2000,
        avg_household_size=3,
        time_of_day=10, # 10:00
        timestep=0.5 # hour
    ):
        super().__init__()
        self.location_data = {}
        self.grid = build_city_grid(self, load_city_map(city_map_path))
        self.width = self.grid.width
        self.height = self.grid.height
        self.avg_household_size = avg_household_size

        self.running = True
        self.time_of_day = time_of_day
        self.timestep = timestep

        # Create agents
        agents_list = []

        self._create_property_layers()
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
        self._create_workplaces(
            [agent for agent in agents_list if agent.agent_type in [AgentType.WORKER, AgentType.HEALTHCARE]]
        )

        # Infect one initial agent
        patient_zero = random.choice(agents_list)
        patient_zero.health_state = patient_zero.health_state.INFECTIOUS
        
    def _create_property_layers(self):
        self.type_property_layer = PropertyLayer(
            "cell_type",
            width=self.width,
            height=self.height,
            default_value=CellType.DEFAULT.value,
        )
        household_id = 1
        workplace_id = 1
        publicspace_id = 1
        for pos, (t, _) in self.location_data.items():
            value = 0
            self.type_property_layer.set_cell(pos, t.value)
            match t:
                case CellType.HOUSEHOLD:
                    value = household_id
                    household_id += 1
                case CellType.WORKPLACE:
                    value = workplace_id
                    workplace_id += 1
                case CellType.PUBLIC_SPACE:
                    value = publicspace_id
                    publicspace_id += 1
            self.location_data[pos] = (t, value)
        self.grid.add_property_layer(self.type_property_layer)

    def _create_households(self, agents_list):
        """
        Create family groups (households) where members have higher
        interaction risk (static social network links).
        """
        agents_copy = agents_list.copy()
        random.shuffle(agents_copy)
        
        households = self.get_cells(CellType.HOUSEHOLD)
        household_sizes = np.ones(len(households), dtype=int)
        
        if len(households) < len(agents_copy):
            for _ in range(len(agents_copy) - len(households)):
                household_idx = random.randrange(0, len(households))
                household_sizes[household_idx] += 1
        
        for idx, household_size in enumerate(household_sizes):
            household = agents_copy[:household_size]
            agents_copy = agents_copy[household_size:]
            
            household_pos = households[idx]
            (_, household_id) = self.location_data[household_pos]

            for agent in household:
                agent.household_id = household_id
                agent.family_members = [a for a in household if a != agent]
                
    def _create_workplaces(self, workers_list):
        """
        Create family groups (households) where members have higher
        interaction risk (static social network links).
        """
        agents_copy = workers_list.copy()
        random.shuffle(agents_copy)
        
        workplaces = self.get_cells(CellType.WORKPLACE)
        workplace_sizes = np.ones(len(workplaces), dtype=int)
        
        if len(workplaces) < len(workers_list):
            for _ in range(len(agents_copy) - len(workplaces)):
                workplace_idx = random.randrange(0, len(workplaces))
                workplace_sizes[workplace_idx] += 1
        
        for idx, workplace_size in enumerate(workplace_sizes):
            workplace = agents_copy[:workplace_size]
            agents_copy = agents_copy[workplace_size:]
            
            workplace_pos = workplaces[idx]
            (_, workplace_id) = self.location_data[workplace_pos]

            for agent in workplace:
                agent.workplace_id = workplace_id
                agent.family_members = [a for a in workplace if a != agent]
                        
    def get_cells(self, cell_type):
        return [
            pos for pos, (t, _) in self.location_data.items()
            if t == cell_type
        ]
        
    def get_cell_position_by_id(self, cell_id, cell_type):
        matching_cell = [
            pos for pos, (type, id) in self.location_data.items() if id == cell_id and type == cell_type
        ]
        if len(matching_cell) == 0:
            print(f"[WARNING]: No cell found for type {cell_type} and ID {cell_id}")
            return None
        if len(matching_cell) > 1:
            print(f"[WARNING]: Multiple cells found for type {cell_type} and ID {cell_id}. Selecting {matching_cell[0]}")
        return matching_cell[0]
            
    def place_agent_in_city(self, agent):
        household_cells = self.get_cells(CellType.HOUSEHOLD)

        workplace_cells = self.get_cells(CellType.WORKPLACE)

        public_cells = self.get_cells(CellType.PUBLIC_SPACE)

        r = random.random()

        if r < 0.6:
            pos = random.choice(household_cells)
        elif r < 0.9:
            pos = random.choice(workplace_cells)
        else:
            pos = random.choice(public_cells)

        self.grid.place_agent(agent, pos)
    
    def update_time(self):
        self.time_of_day += self.timestep
        print(f"[SYSTEM]: TIME IS: {self.time_of_day}")
        if self.time_of_day >= 24:
            self.time_of_day -= 24
        
    def update_agents_based_on_time(self):
        for agent in self.agents:
            if not agent.work_time:
                continue
            if agent.work_time.end <= self.time_of_day:
                agent.target_destination = CellType.HOUSEHOLD
            elif agent.work_time.end >= self.time_of_day >= agent.work_time.start:
                agent.target_destination = CellType.WORKPLACE

    def step(self):
        self.agents.shuffle_do("step")
        self.update_time()
        self.update_agents_based_on_time()
