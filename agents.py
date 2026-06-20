from mesa import Agent
from states import CellType, HealthState
from agent_types import (
    FAMILY_INTERACTION_MULTIPLIER,
    FRIEND_INTERACTION_MULTIPLIER,
    MAX_TRANSMISSION_DISTANCE as _DEFAULT_MAX_DISTANCE,
    AgentType,
    AgentWorkTime
)
import random
import math


def _resolve_max_distance(model) -> int:
    """Read the per-model transmission radius, falling back to the constant.

    Going through the model lets the GUI / CLI tweak the radius per run
    without re-importing the module.
    """
    return int(getattr(model, "max_transmission_distance", _DEFAULT_MAX_DISTANCE))

class PersonAgent(Agent):

    def __init__(
        self,
        model,
        agent_type,
        infection_rate,
        mobility,
        work_time: AgentWorkTime | None = None,
        day_destination: CellType | None = None,
    ):
        super().__init__(model)

        self.agent_type = agent_type
        self.health_state = HealthState.SUSCEPTIBLE

        self.infection_rate = infection_rate
        self.mobility = mobility

        self.days_infected = 0

        # Spatial-destination IDs. None means "no such place assigned" - either
        # because the agent type doesn't go there (e.g. a senior has no
        # workplace_id) or because the loaded map has no cells of that type.
        self.household_id = None   # where the agent lives
        self.workplace_id = None   # WORKER / HEALTHCARE day destination
        self.university_id = None  # STUDENT day destination
        self.school_id = None      # CHILDREN day destination

        # Logical social-network IDs (no spatial meaning). Same id grants a
        # transmission multiplier:
        #   family_id  -> 2.0x   (relatives, possibly across multiple homes)
        #   friends_id -> 1.5x   (friend circle / student roommates)
        self.family_id = None
        self.friends_id = None

        self.__target_destination = CellType.DEFAULT
        # ``work_time`` and ``day_destination`` describe where the agent goes
        # during their active hours. Both come from AGENT_CONFIG via the model.
        self.work_time = work_time
        self.day_destination = day_destination
        
    @property
    def x(self):
        return self.pos[0]
      
    @property
    def y(self):
        return self.pos[1]
    
    # CellTypes the agent may explicitly head toward (everything else falls
    # back to DEFAULT = wander).
    _ALLOWED_TARGETS = frozenset({
        CellType.HOUSEHOLD,
        CellType.PUBLIC_SPACE,
        CellType.WORKPLACE,
        CellType.UNIVERSITY,
        CellType.SCHOOL,
    })

    # CellType -> attribute name holding the agent's id for that cell type.
    _DESTINATION_ID_ATTR = {
        CellType.HOUSEHOLD: "household_id",
        CellType.WORKPLACE: "workplace_id",
        CellType.UNIVERSITY: "university_id",
        CellType.SCHOOL: "school_id",
    }

    @property
    def target_destination(self):
        return self.__target_destination

    @target_destination.setter
    def target_destination(self, value):
        if value in self._ALLOWED_TARGETS:
            self.__target_destination = value
        else:
            self.__target_destination = CellType.DEFAULT

    def _destination_id(self, cell_type: CellType):
        """Return the agent's id for the given destination cell type, or None
        if this agent has no such destination (e.g. seniors and workplaces)."""
        attr = self._DESTINATION_ID_ATTR.get(cell_type)
        return None if attr is None else getattr(self, attr)

    def get_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.
        Accounts for torus topology (wrapped world).
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Account for torus wrapping - ! only if torus topology is enabled in mesa model MultiGrid
        # dx = min(dx, self.model.grid.width - dx)
        # dy = min(dy, self.model.grid.height - dy)
        
        return math.sqrt(dx**2 + dy**2)

    def calculate_transmission_probability(self, distance, is_family=False, is_friend=False):
        """
        Calculate transmission probability based on distance.
        Uses linear decay: probability decreases linearly with distance.
        Family relation overrides friend relation if both are true.
        """
        max_distance = _resolve_max_distance(self.model)
        if distance > max_distance:
            return 0.0

        # Linear decay: 1.0 at distance 0, 0.0 at max_distance.
        base_probability = max(0.0, 1.0 - (distance / max_distance))

        if is_family:
            base_probability = min(1.0, base_probability * FAMILY_INTERACTION_MULTIPLIER)
        elif is_friend:
            base_probability = min(1.0, base_probability * FRIEND_INTERACTION_MULTIPLIER)

        return base_probability * self.infection_rate

    def is_family_member(self, other) -> bool:
        """Logical family relation (not necessarily living together)."""
        return (
            self.family_id is not None
            and self.family_id == other.family_id
        )

    def is_household_member(self, other) -> bool:
        """Spatial co-residence (same physical home)."""
        return (
            self.household_id is not None
            and self.household_id == other.household_id
        )

    def is_friend_of(self, other) -> bool:
        return (
            self.friends_id is not None
            and self.friends_id == other.friends_id
        )
    
    # get all cells where agent can move to in next step
    def get_allowed_neighbors(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True,
        )

        def is_allowed_position(pos):
            cell_type, cell_id = self.model.location_data[pos]
            # Anyone can walk through public spaces and default/street cells.
            if cell_type in (CellType.PUBLIC_SPACE, CellType.DEFAULT):
                return True
            # For typed cells (household / workplace / university / school)
            # only the agents assigned to that specific instance can enter.
            own_id = self._destination_id(cell_type)
            return own_id is not None and own_id == cell_id

        data = [pos for pos in neighbors if is_allowed_position(pos)]
        return [self.pos] if len(data) == 0 else data

    # get the cell which is optimal in terms of reaching assigned target to agent
    def get_optimal_neighbor(self, neighbors, target_pos):
        return min(neighbors, key=lambda pos: self.get_distance(pos, target_pos))

    def move(self):
        neighbors = self.get_allowed_neighbors()
        # No explicit target -> just wander (or hang around public space).
        if self.target_destination in (CellType.DEFAULT, CellType.PUBLIC_SPACE):
            new_position = random.choice(neighbors)
        else:
            target_id = self._destination_id(self.target_destination)
            destination = (
                self.model.get_cell_position_by_id(target_id, self.target_destination)
                if target_id is not None
                else None
            )
            # Map doesn't have this destination type or we have no id assigned
            # (e.g. STUDENT on a city map without UNIVERSITY cells). Fall back
            # to wandering instead of crashing.
            if destination is None:
                new_position = random.choice(neighbors)
            else:
                new_position = self.get_optimal_neighbor(neighbors, destination)
        self.model.grid.move_agent(self, new_position)

    def interact(self):
        """
        Spread disease through spatial interactions within transmission range.

        A single sweep over all agents in the radius-``max_transmission_distance``
        Moore neighbourhood (centre cell included). The per-pair multiplier is
        decided from static-network IDs:
          - same family_id   -> family   (2.0x)
          - same friends_id  -> friend   (1.5x)   (used only if not family)
          - otherwise        -> 1.0x
        household_id and workplace_id are spatial only - co-residents who do
        not share family_id transmit purely through spatial proximity.
        """
        if self.health_state != HealthState.INFECTIOUS:
            return

        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=True,
            radius=_resolve_max_distance(self.model),
        )
        for cell_pos in neighborhood:
            for agent in self.model.grid.get_cell_list_contents([cell_pos]):
                if agent is self or agent.health_state != HealthState.SUSCEPTIBLE:
                    continue

                is_family = self.is_family_member(agent)
                is_friend = (not is_family) and self.is_friend_of(agent)
                distance = self.get_distance(self.pos, agent.pos)
                transmission_prob = self.calculate_transmission_probability(
                    distance,
                    is_family=is_family,
                    is_friend=is_friend,
                )
                if random.random() < transmission_prob:
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
