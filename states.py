from enum import Enum

class HealthState(Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTIOUS = "I"
    RECOVERED = "R"
    