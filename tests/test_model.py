"""
Test suite for the epidemic model
"""
import pytest
from model import EpidemicModel
from states import HealthState
from agent_types import AgentType

TEST_CITY = "tests/test_city.txt"


def test_model_initialization():
    """Test that model initializes with correct parameters"""
    population = 30
    model = EpidemicModel(city_map_path=TEST_CITY, population=population)

    assert len(model.agents) == population
    assert model.running is True


def test_grid_creation():
    """Test that grid is created from city map"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=10)

    assert model.grid.width == 10
    assert model.grid.height == 10


def test_model_step():
    """Test that model step executes without errors"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=10)

    for _ in range(10):
        model.step()


def test_disease_spreads():
    """Test that disease spreads from infectious agents"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=50, avg_household_size=3)

    # Mark two agents as infectious to improve spread reliability
    agents_list = list(model.agents)
    for agent in agents_list[:2]:
        agent.health_state = HealthState.INFECTIOUS

    initial_infectious = sum(
        1 for agent in model.agents
        if agent.health_state == HealthState.INFECTIOUS
    )

    for _ in range(50):
        model.step()

    total_affected = sum(
        1 for agent in model.agents
        if agent.health_state in [HealthState.EXPOSED, HealthState.INFECTIOUS, HealthState.RECOVERED]
    )

    assert total_affected > initial_infectious


def test_population_conservation():
    """Test that agent population doesn't change"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=20)
    initial_population = len(model.agents)

    for _ in range(20):
        model.step()

    assert len(model.agents) == initial_population


def test_health_states_valid():
    """Test that all agents have valid health states"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=30)

    for _ in range(10):
        model.step()

    for agent in model.agents:
        assert agent.health_state in [
            HealthState.SUSCEPTIBLE,
            HealthState.EXPOSED,
            HealthState.INFECTIOUS,
            HealthState.RECOVERED
        ]


def test_default_time_parameters():
    """Test model default time_of_day and timestep parameters"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=10)

    assert model.time_of_day == 10
    assert model.timestep == 0.5


def test_custom_time_parameters():
    """Test model accepts custom startTime and timestep"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=10, time_of_day=8, timestep=0.1)

    assert model.time_of_day == 8
    assert model.timestep == 0.1


def test_time_advances_each_step():
    """Test that time_of_day advances by timestep on each model step"""
    timestep = 0.25
    model = EpidemicModel(city_map_path=TEST_CITY, population=5, time_of_day=9.0, timestep=timestep)

    model.step()
    assert abs(model.time_of_day - (9.0 + timestep)) < 1e-9


def test_small_model():
    """Test model with minimal population"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=3)

    assert len(model.agents) == 3

    for _ in range(5):
        model.step()


def test_large_model():
    """Test model with larger population"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=100)

    assert len(model.agents) == 100

    for _ in range(3):
        model.step()


def test_household_assignment():
    """Test that all agents are assigned to households"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=30)

    for agent in model.agents:
        assert agent.household_id is not None
        for family_member in agent.family_members:
            assert family_member.household_id == agent.household_id


def test_average_household_size_respected():
    """Test that average household size is roughly respected"""
    avg_size = 4
    model = EpidemicModel(city_map_path=TEST_CITY, population=40, avg_household_size=avg_size)

    household_sizes = {}
    for agent in model.agents:
        if agent.household_id not in household_sizes:
            household_sizes[agent.household_id] = 0
        household_sizes[agent.household_id] += 1

    avg_actual = sum(household_sizes.values()) / len(household_sizes)

    assert 1 <= avg_actual <= avg_size + 2


def test_all_agent_types_present():
    """Test that all agent types can be created"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=100)

    types_found = set(agent.agent_type for agent in model.agents)

    assert len(types_found) > 1


def test_agents_on_grid():
    """Test that all agents are placed on grid"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=20)

    for agent in model.agents:
        x, y = agent.pos
        assert 0 <= x < model.width
        assert 0 <= y < model.height

        grid_agents = model.grid.get_cell_list_contents([agent.pos])
        assert agent in grid_agents


def test_simulation_runs_steps():
    """Test that a full simulation runs for many steps"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=15)

    for _ in range(100):
        model.step()

        assert len(model.agents) == 15
        for agent in model.agents:
            assert agent.health_state in [
                HealthState.SUSCEPTIBLE,
                HealthState.EXPOSED,
                HealthState.INFECTIOUS,
                HealthState.RECOVERED
            ]
