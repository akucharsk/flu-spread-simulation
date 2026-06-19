from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from states import CellType


class AgentType(Enum):
    """Agent types in the epidemic model"""
    STUDENT = "student"
    WORKER = "worker"
    SENIOR = "senior"
    HEALTHCARE = "healthcare"
    CHILDREN = "children"


@dataclass
class AgentWorkTime:
    """Inclusive start/end (hours, 0-24) of an agent's daily active window."""
    start: float
    end: float


@dataclass
class AgentParameters:
    """Per-agent-type tuning.

    - ``active_hours``: when the agent is at their day destination.
      ``None`` means the agent stays around home (e.g. retirees).
    - ``active_destination``: which CellType the agent commutes to during
      ``active_hours``. ``None`` for agents without a day destination.
    """
    mobility: float          # probability of moving per step
    infection_rate: float    # base infection probability (same cell)
    active_hours: Optional[AgentWorkTime] = None
    active_destination: Optional[CellType] = None


# Agent type parameters configuration
AGENT_CONFIG = {
    AgentType.STUDENT: AgentParameters(
        mobility=0.9,
        infection_rate=0.2,
        active_hours=AgentWorkTime(8, 20),
        active_destination=CellType.UNIVERSITY,
    ),
    AgentType.WORKER: AgentParameters(
        mobility=0.8,
        infection_rate=0.25,
        active_hours=AgentWorkTime(9, 17),
        active_destination=CellType.WORKPLACE,
    ),
    AgentType.SENIOR: AgentParameters(
        mobility=0.3,
        infection_rate=0.35,
        active_hours=None,
        active_destination=None,
    ),
    AgentType.HEALTHCARE: AgentParameters(
        mobility=1.0,
        infection_rate=0.15,
        active_hours=AgentWorkTime(9, 17),
        active_destination=CellType.WORKPLACE,
    ),
    AgentType.CHILDREN: AgentParameters(
        mobility=0.9,
        infection_rate=0.22,
        active_hours=AgentWorkTime(8, 15),
        active_destination=CellType.SCHOOL,
    ),
}


# Social network parameters
FAMILY_INTERACTION_MULTIPLIER = 2.0  # Household members have 2x higher transmission risk
FRIEND_INTERACTION_MULTIPLIER = 1.5  # Friend-group members have 1.5x higher transmission risk
DISTANCE_DECAY_FUNCTION = "linear"  # Linear decay of transmission with distance
MAX_TRANSMISSION_DISTANCE = 5  # Maximum distance for transmission to occur
DEFAULT_AVG_FAMILY_SIZE = 3  # Default average size of randomly assigned family groups
DEFAULT_AVG_FRIEND_GROUP_SIZE = 4  # Default average size of randomly assigned friend groups
