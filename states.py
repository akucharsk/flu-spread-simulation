from enum import Enum

class HealthState(Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTIOUS = "I"
    RECOVERED = "R"
    
class CellType(Enum):
    DEFAULT = 0       # placeholder / not-yet-implemented places (safety bucket)
    HOUSEHOLD = 1     # where agents live
    WORKPLACE = 2     # WORKER / HEALTHCARE day destination
    PUBLIC_SPACE = 3  # parks, streets, plazas - anyone can enter
    UNIVERSITY = 4    # STUDENT day destination (longer day, 8-20)
    SCHOOL = 5        # CHILDREN day destination (shorter day, 8-15)
