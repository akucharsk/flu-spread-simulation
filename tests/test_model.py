"""
Test suite for the epidemic model
"""
import pytest
from model import EpidemicModel
from states import HealthState
from agent_types import AgentType


def test_model_initialization():
    """Test that model initializes with correct parameters"""
    width, height, population = 25, 25, 100
    model = EpidemicModel(width=width, height=height, population=population)
    
    assert model.width == width
    assert model.height == height
    assert len(model.agents) == population
    assert model.running is True


def test_grid_creation():
    """Test that grid is created with correct dimensions"""
    model = EpidemicModel(width=30, height=40, population=50)
    
    assert model.grid.width == 30
    assert model.grid.height == 40
    assert model.grid.torus is True  # Should be toroidal


def test_model_step():
    """Test that model step executes without errors"""
    model = EpidemicModel(width=10, height=10, population=10)
    
    # Run multiple steps
    for _ in range(10):
        model.step()


def test_disease_spreads():
    """Test that disease spreads from infectious agents"""
    # Small population to ensure interaction
    model = EpidemicModel(width=5, height=5, population=15, avg_household_size=2)
    
    # Count initial state
    initial_infectious = sum(
        1 for agent in model.agents 
        if agent.health_state == HealthState.INFECTIOUS
    )
    initial_exposed = sum(
        1 for agent in model.agents 
        if agent.health_state == HealthState.EXPOSED
    )
    
    assert initial_infectious >= 1
    
    # Run simulation
    for _ in range(20):
        model.step()
    
    # After spreading, should have more exposed/infectious
    total_affected = sum(
        1 for agent in model.agents 
        if agent.health_state in [HealthState.EXPOSED, HealthState.INFECTIOUS, HealthState.RECOVERED]
    )
    
    # At least some disease spread should occur
    assert total_affected > initial_infectious


def test_population_conservation():
    """Test that agent population doesn't change"""
    model = EpidemicModel(width=10, height=10, population=50)
    initial_population = len(model.agents)
    
    for _ in range(20):
        model.step()
    
    assert len(model.agents) == initial_population


def test_health_states_valid():
    """Test that all agents have valid health states"""
    model = EpidemicModel(width=10, height=10, population=100)
    
    for _ in range(10):
        model.step()
    
    for agent in model.agents:
        assert agent.health_state in [
            HealthState.SUSCEPTIBLE,
            HealthState.EXPOSED,
            HealthState.INFECTIOUS,
            HealthState.RECOVERED
        ]


def test_default_parameters():
    """Test model with default parameters"""
    model = EpidemicModel()
    
    assert model.width == 20
    assert model.height == 20
    assert len(model.agents) == 100


def test_small_model():
    """Test model with minimal population"""
    model = EpidemicModel(width=5, height=5, population=3)
    
    assert len(model.agents) == 3
    
    for _ in range(5):
        model.step()


def test_large_model():
    """Test model with larger population"""
    model = EpidemicModel(width=50, height=50, population=500)
    
    assert len(model.agents) == 500
    
    # Should handle step without issues
    for _ in range(3):
        model.step()


def test_household_assignment():
    """Test that all agents are assigned to households"""
    model = EpidemicModel(width=10, height=10, population=30)
    
    for agent in model.agents:
        assert agent.household_id is not None
        # Each household should have member links
        if len(agent.family_members) > 0:
            for family_member in agent.family_members:
                assert family_member.household_id == agent.household_id


def test_average_household_size_respected():
    """Test that average household size is roughly respected"""
    avg_size = 4
    model = EpidemicModel(width=10, height=10, population=100, avg_household_size=avg_size)
    
    # Count households and members
    household_sizes = {}
    for agent in model.agents:
        if agent.household_id not in household_sizes:
            household_sizes[agent.household_id] = 0
        household_sizes[agent.household_id] += 1
    
    # Calculate average
    avg_actual = sum(household_sizes.values()) / len(household_sizes)
    
    # Should be somewhat close to requested average
    assert 1 <= avg_actual <= avg_size + 2


def test_all_agent_types_present():
    """Test that all agent types can be created"""
    # Use large population to ensure all types appear
    model = EpidemicModel(width=20, height=20, population=500)
    
    types_found = set()
    for agent in model.agents:
        types_found.add(agent.agent_type)
    
    # Should have multiple different types
    assert len(types_found) > 1


def test_agents_on_grid():
    """Test that all agents are placed on grid"""
    model = EpidemicModel(width=15, height=15, population=40)
    
    for agent in model.agents:
        x, y = agent.pos
        # Check agent is on grid
        assert 0 <= x < model.width
        assert 0 <= y < model.height
        
        # Check agent is in grid's agent list
        grid_agents = model.grid.get_cell_list_contents([agent.pos])
        assert agent in grid_agents


def test_simulation_runs_steps():
    """Test that a full simulation runs for many steps"""
    model = EpidemicModel(width=10, height=10, population=25)
    
    # Run 100 steps without errors
    for step_num in range(100):
        model.step()
        
        # Check invariants
        assert len(model.agents) == 25
        for agent in model.agents:
            assert agent.health_state in [
                HealthState.SUSCEPTIBLE,
                HealthState.EXPOSED,
                HealthState.INFECTIOUS,
                HealthState.RECOVERED
            ]
