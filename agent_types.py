from enum import Enum
from dataclasses import dataclass


class AgentType(Enum):
    """Agent types in the epidemic model"""
    STUDENT = "student"
    WORKER = "worker"
    SENIOR = "senior"
    HEALTHCARE = "healthcare"
    CHILDREN = "children"


@dataclass
class AgentParameters:
    """Parameters for each agent type"""
    mobility: float  # Probability of moving per step
    infection_rate: float  # Base infection probability (same cell)
    

# Agent type parameters configuration
AGENT_CONFIG = {
    AgentType.STUDENT: AgentParameters(
        mobility=0.9,
        infection_rate=0.2
    ),
    AgentType.WORKER: AgentParameters(
        mobility=0.8,
        infection_rate=0.25
    ),
    AgentType.SENIOR: AgentParameters(
        mobility=0.3,
        infection_rate=0.35
    ),
    AgentType.HEALTHCARE: AgentParameters(
        mobility=1.0,
        infection_rate=0.15
    ),
    AgentType.CHILDREN: AgentParameters(
        mobility=0.9,  # Same as students
        infection_rate=0.22  # Slightly higher than students
    ),
}


# Social network parameters
FAMILY_INTERACTION_MULTIPLIER = 2.0  # Family members have 2x higher transmission risk
DISTANCE_DECAY_FUNCTION = "linear"  # Linear decay of transmission with distance
MAX_TRANSMISSION_DISTANCE = 5  # Maximum distance for transmission to occur
