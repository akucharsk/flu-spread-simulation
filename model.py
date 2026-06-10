from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import PropertyLayer
import numpy as np

from agents import PersonAgent
from agent_types import AgentType, AGENT_CONFIG
import random

from city_utils import build_city_grid, load_city_map
from map_presets import get_map_label, resolve_city_map_path
from states import CellType, HealthState


def _infectious_of_type_reporter(agent_type: AgentType):
    """Build a DataCollector reporter for infectious agents of a given type."""
    return lambda m: sum(
        1 for a in m.agents
        if a.agent_type == agent_type and a.health_state == HealthState.INFECTIOUS
    )


class EpidemicModel(Model):

    def __init__(
        self,
        city_map_path="tests/test_city.txt",
        city_map_preset=None,
        population=2000,
        avg_household_size=3,
        time_of_day=10, # 10:00
        timestep=0.5, # hour
        verbose=False,
    ):
        super().__init__()
        self.location_data = {}
        self.city_map_preset = city_map_preset
        self.city_map_path = resolve_city_map_path(city_map_path, city_map_preset)
        self.map_name = get_map_label(self.city_map_preset, self.city_map_path)

        self.grid = build_city_grid(self, load_city_map(self.city_map_path))
        self.width = self.grid.width
        self.height = self.grid.height
        self.avg_household_size = avg_household_size

        self.running = True
        self.current_step = 0
        self.time_of_day = time_of_day
        self.timestep = timestep
        self.verbose = verbose
        self.metrics_history = []
        self.new_exposures_step = 0
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: m.get_health_counts()["susceptible"],
                "Exposed": lambda m: m.get_health_counts()["exposed"],
                "Infectious": lambda m: m.get_health_counts()["infectious"],
                "Recovered": lambda m: m.get_health_counts()["recovered"],
                "New_Exposures": lambda m: m.new_exposures_step,
                "Cumulative_Infected": lambda m: len(m.agents) - m.get_health_counts()["susceptible"],
                **{
                    f"Infectious_{t.value}": _infectious_of_type_reporter(t)
                    for t in AgentType
                },
            }
        )

        # Create agents
        agents_list = []

        self._create_property_layers()
        for _ in range(population):
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
        self.record_metrics()
        self.datacollector.collect(self)

    def get_health_counts(self):
        counts = {
            "susceptible": 0,
            "exposed": 0,
            "infectious": 0,
            "recovered": 0,
        }
        for agent in self.agents:
            if agent.health_state == HealthState.SUSCEPTIBLE:
                counts["susceptible"] += 1
            elif agent.health_state == HealthState.EXPOSED:
                counts["exposed"] += 1
            elif agent.health_state == HealthState.INFECTIOUS:
                counts["infectious"] += 1
            elif agent.health_state == HealthState.RECOVERED:
                counts["recovered"] += 1
        return counts

    def get_metrics_snapshot(self):
        counts = self.get_health_counts()
        population = len(self.agents)
        cumulative_infected = population - counts["susceptible"]
        return {
            "step": self.current_step,
            "time_of_day": round(self.time_of_day, 4),
            "susceptible": counts["susceptible"],
            "exposed": counts["exposed"],
            "infectious": counts["infectious"],
            "recovered": counts["recovered"],
            "population": population,
            "infected_ratio": counts["infectious"] / population if population else 0.0,
            "cumulative_infected": cumulative_infected,
            "cumulative_ratio": cumulative_infected / population if population else 0.0,
            "new_exposures": self.new_exposures_step,
        }

    def record_metrics(self):
        self.metrics_history.append(self.get_metrics_snapshot())

    def get_health_counts_by_type(self):
        """Return per-agent-type breakdown of the four health states."""
        result = {
            t.value: {"susceptible": 0, "exposed": 0, "infectious": 0, "recovered": 0}
            for t in AgentType
        }
        for agent in self.agents:
            bucket = result[agent.agent_type.value]
            if agent.health_state == HealthState.SUSCEPTIBLE:
                bucket["susceptible"] += 1
            elif agent.health_state == HealthState.EXPOSED:
                bucket["exposed"] += 1
            elif agent.health_state == HealthState.INFECTIOUS:
                bucket["infectious"] += 1
            elif agent.health_state == HealthState.RECOVERED:
                bucket["recovered"] += 1
        return result

    def get_summary_metrics(self):
        peak = max(self.metrics_history, key=lambda item: item["infectious"])
        final = self.metrics_history[-1]
        return {
            "steps_executed": self.current_step,
            "peak_infectious": peak["infectious"],
            "peak_infectious_step": peak["step"],
            "final_susceptible": final["susceptible"],
            "final_exposed": final["exposed"],
            "final_infectious": final["infectious"],
            "final_recovered": final["recovered"],
            "total_ever_infected": len(self.agents) - final["susceptible"],
        }
        
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
        if self.verbose:
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
        prev_susceptible = self.get_health_counts()["susceptible"]
        self.agents.shuffle_do("step")
        self.update_time()
        self.update_agents_based_on_time()
        self.current_step += 1
        new_susceptible = self.get_health_counts()["susceptible"]
        # Susceptible can only decrease so the delta is the number of newly
        # exposed agents during this step. clamp to 0 just in case.
        self.new_exposures_step = max(0, prev_susceptible - new_susceptible)
        self.record_metrics()
        self.datacollector.collect(self)
