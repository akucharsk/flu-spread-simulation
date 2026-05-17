from mesa import Agent
from states import HealthState
from agent_types import (
    FAMILY_INTERACTION_MULTIPLIER,
    MAX_TRANSMISSION_DISTANCE
)
import random
import math


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
        
        # Social network connections
        self.family_members = []  # Static links - family/housemates
        self.household_id = None  # Identifies which household they belong to
        
    @property
    def x(self):
        return self.pos[0]
      
    @property
    def y(self):
        return self.pos[1]

    def get_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.
        Accounts for torus topology (wrapped world).
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Account for torus wrapping
        dx = min(dx, self.model.grid.width - dx)
        dy = min(dy, self.model.grid.height - dy)
        
        return math.sqrt(dx**2 + dy**2)

    def calculate_transmission_probability(self, distance, is_family=False):
        """
        Calculate transmission probability based on distance.
        Uses linear decay: probability decreases linearly with distance.
        Family members have higher transmission probability (multiplier).
        """
        if distance > MAX_TRANSMISSION_DISTANCE:
            return 0.0
        
        # Linear decay: 1.0 at distance 0, 0.0 at MAX_TRANSMISSION_DISTANCE
        base_probability = max(0.0, 1.0 - (distance / MAX_TRANSMISSION_DISTANCE))
        
        # Apply family multiplier if applicable
        if is_family:
            base_probability = min(1.0, base_probability * FAMILY_INTERACTION_MULTIPLIER)
        
        return base_probability * self.infection_rate

    def move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        new_position = random.choice(neighbors)
        self.model.grid.move_agent(self, new_position)

    def interact(self):
        """
        Spread disease through social interactions:
        1. Family interactions (static links) - higher transmission risk
        2. Environmental/spatial interactions - distance-based transmission risk
        """
        if self.health_state != HealthState.INFECTIOUS:
            return

        # 1. Interact with family members (static social network)
        for family_member in self.family_members:
            if family_member.health_state == HealthState.SUSCEPTIBLE:
                distance = self.get_distance(self.pos, family_member.pos)
                transmission_prob = self.calculate_transmission_probability(
                    distance, 
                    is_family=True
                )
                
                if random.random() < transmission_prob:
                    family_member.health_state = HealthState.EXPOSED

        # 2. Interact with nearby agents (dynamic/spatial encounters)
        # Get agents within transmission radius
        nearby_agents = self.model.grid.get_cell_list_contents([self.pos])
        
        for agent in nearby_agents:
            if agent == self or agent.health_state != HealthState.SUSCEPTIBLE:
                continue
            
            # Check if already processed as family member
            if agent in self.family_members:
                continue
            
            # Same cell = distance 0, higher transmission probability
            transmission_prob = self.calculate_transmission_probability(
                distance=0.0,
                is_family=False
            )
            
            if random.random() < transmission_prob:
                agent.health_state = HealthState.EXPOSED
        
        # 3. Check neighboring cells for extended spatial transmission
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        
        for neighbor_pos in neighbors:
            neighbor_agents = self.model.grid.get_cell_list_contents([neighbor_pos])
            
            for agent in neighbor_agents:
                if agent.health_state != HealthState.SUSCEPTIBLE:
                    continue
                
                # Skip family members (already handled)
                if agent in self.family_members:
                    continue
                
                # Calculate distance and transmission probability
                distance = self.get_distance(self.pos, agent.pos)
                transmission_prob = self.calculate_transmission_probability(
                    distance,
                    is_family=False
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