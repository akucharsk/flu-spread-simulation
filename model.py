from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import PropertyLayer
import numpy as np

from agents import PersonAgent
from agent_types import (
    AGENT_CONFIG,
    AgentType,
    DEFAULT_AVG_FRIEND_GROUP_SIZE,
    DEFAULT_AVG_FAMILY_SIZE,
    MAX_TRANSMISSION_DISTANCE as _DEFAULT_MAX_DISTANCE,
    build_agent_config,
)
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
        avg_family_size=DEFAULT_AVG_FAMILY_SIZE,
        avg_friend_group_size=DEFAULT_AVG_FRIEND_GROUP_SIZE,
        time_of_day=10, # 10:00
        timestep=0.5, # hour
        verbose=False,
        max_transmission_distance=None,
        agent_overrides=None,
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
        self.avg_family_size = max(1, int(avg_family_size))
        self.avg_friend_group_size = max(1, int(avg_friend_group_size))

        # Transmission radius (in cells). None -> use module default.
        self.max_transmission_distance = (
            int(max_transmission_distance)
            if max_transmission_distance is not None
            else _DEFAULT_MAX_DISTANCE
        )
        # Per-type mobility / infection_rate overrides applied on top of
        # AGENT_CONFIG. ``self.agent_config`` is the source of truth used
        # when spawning agents.
        self.agent_overrides = agent_overrides or {}
        self.agent_config = build_agent_config(self.agent_overrides)

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
            params = self.agent_config[agent_type]

            agent = PersonAgent(
                self,
                agent_type=agent_type,
                infection_rate=params.infection_rate,
                mobility=params.mobility,
                work_time=params.active_hours,
                day_destination=params.active_destination,
            )

            agents_list.append(agent)

            self.place_agent_in_city(agent)

        # Spatial assignments (where each agent lives / studies / works).
        # No transmission boost comes from these on their own.
        self._create_households(agents_list)
        for cell_type, id_attribute, agent_types in (
            (CellType.WORKPLACE, "workplace_id", (AgentType.WORKER, AgentType.HEALTHCARE)),
            (CellType.UNIVERSITY, "university_id", (AgentType.STUDENT,)),
            (CellType.SCHOOL,     "school_id",     (AgentType.CHILDREN,)),
        ):
            self._assign_to_cells(
                [a for a in agents_list if a.agent_type in agent_types],
                cell_type=cell_type,
                id_attribute=id_attribute,
            )

        # Independent logical groups for transmission multipliers:
        #   family (2x) - relatives, may live in different households
        #   friends (1.5x) - small social circles
        self._create_family_groups(agents_list)
        self._create_friend_groups(agents_list)

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
        per_type = self.get_health_counts_by_type()
        population = len(self.agents)
        cumulative_infected = population - counts["susceptible"]
        snapshot = {
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
        # Per-agent-type breakdown so post-hoc analyses can ask 'which group
        # carries the peak?' without re-running the simulation.
        for type_name, buckets in per_type.items():
            for state, n in buckets.items():
                snapshot[f"{state}_{type_name}"] = n
        return snapshot

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
        population = len(self.agents)
        total_ever_infected = population - final["susceptible"]

        # Aggregates that are useful for post-hoc comparisons but cheap to
        # compute here from metrics_history.
        max_new_exposures = max(
            (row.get("new_exposures", 0) for row in self.metrics_history),
            default=0,
        )
        infectious_steps = sum(
            1 for row in self.metrics_history if row.get("infectious", 0) > 0
        )

        attack_rate = total_ever_infected / population if population else 0.0
        # Time-to-half-attack: first step at which 50% of the *eventually*
        # infected population has been infected.
        half_target = total_ever_infected / 2
        time_to_half_attack = None
        running_total = 0
        for row in self.metrics_history:
            running_total = population - row["susceptible"]
            if running_total >= half_target and half_target > 0:
                time_to_half_attack = row["step"]
                break

        return {
            "steps_executed": self.current_step,
            "peak_infectious": peak["infectious"],
            "peak_infectious_step": peak["step"],
            "peak_infectious_time": peak["time_of_day"],
            "max_new_exposures_per_step": max_new_exposures,
            "infectious_steps": infectious_steps,
            "attack_rate": round(attack_rate, 6),
            "time_to_half_attack_step": time_to_half_attack,
            "final_susceptible": final["susceptible"],
            "final_exposed": final["exposed"],
            "final_infectious": final["infectious"],
            "final_recovered": final["recovered"],
            "total_ever_infected": total_ever_infected,
        }
        
    def _create_property_layers(self):
        self.type_property_layer = PropertyLayer(
            "cell_type",
            width=self.width,
            height=self.height,
            default_value=CellType.DEFAULT.value,
        )
        # Each typed-cell kind gets its own monotonically increasing id space.
        next_id = {
            CellType.HOUSEHOLD: 1,
            CellType.WORKPLACE: 1,
            CellType.PUBLIC_SPACE: 1,
            CellType.UNIVERSITY: 1,
            CellType.SCHOOL: 1,
        }
        for pos, (t, _) in self.location_data.items():
            self.type_property_layer.set_cell(pos, t.value)
            if t in next_id:
                value = next_id[t]
                next_id[t] += 1
            else:
                value = 0
            self.location_data[pos] = (t, value)
        self.grid.add_property_layer(self.type_property_layer)

    def _create_households(self, agents_list):
        """
        Assign every agent to a household cell (spatial home). household_id is
        the physical place where the agent goes back at night; it grants no
        transmission boost on its own.

        While we're partitioning the population into households we also seed a
        *social* link between co-residents:
          - households whose members are ALL students -> shared friends_id
            (roommate / dorm-style flat)
          - all other households -> shared family_id (regular family, even if
            the household happens to include a student living with parents)

        These seeded ids are continued (not overwritten) by
        _create_family_groups / _create_friend_groups later, so single-person
        or student-only households still receive an extended family / friend
        circle assigned at random.
        """
        agents_copy = agents_list.copy()
        random.shuffle(agents_copy)

        households = self.get_cells(CellType.HOUSEHOLD)
        household_sizes = np.ones(len(households), dtype=int)

        if len(households) < len(agents_copy):
            for _ in range(len(agents_copy) - len(households)):
                household_idx = random.randrange(0, len(households))
                household_sizes[household_idx] += 1

        next_family_id = 1
        next_friends_id = 1
        for idx, household_size in enumerate(household_sizes):
            household = agents_copy[:household_size]
            agents_copy = agents_copy[household_size:]

            household_pos = households[idx]
            (_, household_id) = self.location_data[household_pos]

            for agent in household:
                agent.household_id = household_id

            # Co-residents must share at least one logical link.
            all_students = all(a.agent_type == AgentType.STUDENT for a in household)
            if all_students:
                for agent in household:
                    agent.friends_id = next_friends_id
                next_friends_id += 1
            else:
                for agent in household:
                    agent.family_id = next_family_id
                next_family_id += 1

        # Hand the next-id counters off to the random-group helpers so the id
        # spaces don't overlap.
        self._next_family_id = next_family_id
        self._next_friends_id = next_friends_id

    def _assign_to_cells(self, agents_list, cell_type, id_attribute):
        """Distribute the given agents across all cells of ``cell_type``.

        Writes the cell's id into ``id_attribute`` on every assigned agent.
        Acts as a no-op if the loaded map has no cells of that type, leaving
        the attribute as ``None`` (graceful fallback during ``move``).
        """
        if not agents_list:
            return
        cells = self.get_cells(cell_type)
        if not cells:
            # Map doesn't contain this destination type - leave the id None
            # and let the agent wander instead of crashing.
            return

        agents_copy = agents_list.copy()
        random.shuffle(agents_copy)

        cell_sizes = np.ones(len(cells), dtype=int)
        if len(cells) < len(agents_copy):
            for _ in range(len(agents_copy) - len(cells)):
                cell_sizes[random.randrange(len(cells))] += 1

        for idx, group_size in enumerate(cell_sizes):
            group = agents_copy[:group_size]
            agents_copy = agents_copy[group_size:]

            cell_pos = cells[idx]
            (_, cell_id) = self.location_data[cell_pos]

            for agent in group:
                setattr(agent, id_attribute, cell_id)

    def _create_workplaces(self, workers_list):
        """Backwards-compatible alias kept for tests/other callers; prefer
        ``_assign_to_cells`` directly with the desired cell type."""
        self._assign_to_cells(workers_list, CellType.WORKPLACE, "workplace_id")

    def _create_family_groups(self, agents_list):
        """
        Give every agent who isn't already part of a household-seeded family
        a random extended-family group of about avg_family_size members.
        Existing family_id assignments (from non-student households) are left
        intact, so a student living with friends still gets a family group
        living elsewhere.
        """
        unassigned = [a for a in agents_list if a.family_id is None]
        self._next_family_id = self._assign_random_groups(
            unassigned,
            attribute="family_id",
            avg_size=self.avg_family_size,
            start_id=getattr(self, "_next_family_id", 1),
        )

    def _create_friend_groups(self, agents_list):
        """
        Give every agent who isn't already part of a household-seeded friend
        circle (i.e. who isn't in an all-student household) a random friend
        group of about avg_friend_group_size members.
        """
        unassigned = [a for a in agents_list if a.friends_id is None]
        self._next_friends_id = self._assign_random_groups(
            unassigned,
            attribute="friends_id",
            avg_size=self.avg_friend_group_size,
            start_id=getattr(self, "_next_friends_id", 1),
        )

    @staticmethod
    def _assign_random_groups(
        agents_list, attribute: str, avg_size: int, start_id: int = 1,
    ) -> int:
        """Shuffle agents and chop them into chunks of ~avg_size, writing the
        chunk index to ``attribute`` on every agent in the chunk. Returns the
        next free id (so callers can chain calls and keep id spaces disjoint).
        """
        if not agents_list:
            return start_id
        agents_copy = agents_list.copy()
        random.shuffle(agents_copy)
        group_size = max(1, int(avg_size))
        next_id = start_id
        for start in range(0, len(agents_copy), group_size):
            for agent in agents_copy[start:start + group_size]:
                setattr(agent, attribute, next_id)
            next_id += 1
        return next_id
                        
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
        # Spawn agents across all walkable cell types so the t=0 snapshot
        # is plausible: most people are at home or already at their day
        # destination, but a meaningful slice is "in transit" on DEFAULT
        # (street) cells - real cities always have agents on the move,
        # not just clustered inside buildings.
        # Weights are interpreted relatively (the code normalises them).
        weighted_choices = [
            (CellType.HOUSEHOLD,    0.42),  # most people at home
            (CellType.DEFAULT,      0.30),  # ~1/3 of agents start on streets
            (CellType.WORKPLACE,    0.105),
            (CellType.PUBLIC_SPACE, 0.07),
            (CellType.UNIVERSITY,   0.0525),
            (CellType.SCHOOL,       0.0525),
        ]
        # Filter to types that actually exist on the map.
        available = [
            (cells, weight) for ctype, weight in weighted_choices
            for cells in [self.get_cells(ctype)] if cells
        ]
        if not available:
            raise RuntimeError("Map has no placeable cells (no household/public/etc.)")

        total_weight = sum(w for _, w in available)
        r = random.random() * total_weight
        accumulator = 0.0
        chosen_cells = available[-1][0]
        for cells, weight in available:
            accumulator += weight
            if r <= accumulator:
                chosen_cells = cells
                break

        self.grid.place_agent(agent, random.choice(chosen_cells))
    
    def update_time(self):
        self.time_of_day += self.timestep
        if self.verbose:
            print(f"[SYSTEM]: TIME IS: {self.time_of_day}")
        if self.time_of_day >= 24:
            self.time_of_day -= 24
        
    def update_agents_based_on_time(self):
        """Send each agent home or to their day destination based on their
        configured ``work_time`` window. Agents without an active schedule
        (e.g. seniors) are left alone."""
        for agent in self.agents:
            if not agent.work_time or not agent.day_destination:
                continue
            if agent.work_time.end <= self.time_of_day:
                agent.target_destination = CellType.HOUSEHOLD
            elif agent.work_time.start <= self.time_of_day <= agent.work_time.end:
                agent.target_destination = agent.day_destination

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
