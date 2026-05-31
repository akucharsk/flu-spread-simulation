from enum import Enum

class HealthState(Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTIOUS = "I"
    RECOVERED = "R"
    
class CellType(Enum):
    DEFAULT = 0
    HOUSEHOLD = 1
    WORKPLACE = 2
    PUBLIC_SPACE = 3
