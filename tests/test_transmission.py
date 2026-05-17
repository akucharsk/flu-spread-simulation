"""
Test suite for agent transmission logic
"""
import pytest
from model import EpidemicModel
from agents import PersonAgent
from states import HealthState
from agent_types import AgentType, FAMILY_INTERACTION_MULTIPLIER, MAX_TRANSMISSION_DISTANCE


def test_agent_creation():
    """Test that agents are created with correct attributes"""
    model = EpidemicModel(width=10, height=10, population=5)
    
    assert len(model.agents) == 5
    for agent in model.agents:
        assert isinstance(agent, PersonAgent)
        assert agent.health_state in [HealthState.SUSCEPTIBLE, HealthState.INFECTIOUS]
        assert agent.agent_type in list(AgentType)
        assert 0 <= agent.infection_rate <= 1
        assert 0 <= agent.mobility <= 1


def test_transmission_distance_calculation():
    """Test distance calculation between agents"""
    model = EpidemicModel(width=20, height=20, population=3)
    agents_list = list(model.agents)
    agent1 = agents_list[0]
    agent2 = agents_list[1]
    
    # Place agents at known positions
    model.grid.move_agent(agent1, (0, 0))
    model.grid.move_agent(agent2, (3, 4))
    
    distance = agent1.get_distance((0, 0), (3, 4))
    expected = 5.0  # sqrt(3^2 + 4^2) = 5
    assert abs(distance - expected) < 0.01


def test_transmission_probability_basic():
    """Test transmission probability calculation at same cell"""
    model = EpidemicModel(width=10, height=10, population=2)
    agent = list(model.agents)[0]
    
    # At distance 0 (same cell), non-family
    prob = agent.calculate_transmission_probability(distance=0.0, is_family=False)
    assert prob > 0
    assert prob <= 1
    
    # At MAX distance, should be near 0
    prob_max = agent.calculate_transmission_probability(
        distance=MAX_TRANSMISSION_DISTANCE, 
        is_family=False
    )
    assert prob_max == 0.0


def test_transmission_probability_family_multiplier():
    """Test that family members have higher transmission risk"""
    model = EpidemicModel(width=10, height=10, population=2)
    agent = list(model.agents)[0]
    
    prob_non_family = agent.calculate_transmission_probability(
        distance=2.0, 
        is_family=False
    )
    prob_family = agent.calculate_transmission_probability(
        distance=2.0, 
        is_family=True
    )
    
    # Family should have higher probability (capped at 1.0)
    assert prob_family >= prob_non_family


def test_transmission_distance_limit():
    """Test that transmission stops beyond MAX_TRANSMISSION_DISTANCE"""
    model = EpidemicModel(width=10, height=10, population=2)
    agent = list(model.agents)[0]
    
    prob_within = agent.calculate_transmission_probability(
        distance=MAX_TRANSMISSION_DISTANCE - 1, 
        is_family=False
    )
    prob_beyond = agent.calculate_transmission_probability(
        distance=MAX_TRANSMISSION_DISTANCE + 1, 
        is_family=False
    )
    
    assert prob_within > 0
    assert prob_beyond == 0.0


def test_linear_decay():
    """Test that transmission probability decays linearly with distance"""
    model = EpidemicModel(width=10, height=10, population=2)
    agent = list(model.agents)[0]
    
    prob_0 = agent.calculate_transmission_probability(distance=0.0, is_family=False)
    prob_2 = agent.calculate_transmission_probability(distance=2.0, is_family=False)
    prob_4 = agent.calculate_transmission_probability(distance=4.0, is_family=False)
    
    # Should decrease linearly
    diff_1 = prob_0 - prob_2
    diff_2 = prob_2 - prob_4
    
    # Differences should be approximately equal for linear decay
    assert abs(diff_1 - diff_2) < 0.01 * agent.infection_rate


def test_health_state_exposed_to_infectious():
    """Test that exposed agents can become infectious"""
    model = EpidemicModel(width=10, height=10, population=1)
    agent = list(model.agents)[0]
    
    agent.health_state = HealthState.EXPOSED
    
    # Run many steps to allow transition
    for _ in range(100):
        agent.update_health()
        if agent.health_state == HealthState.INFECTIOUS:
            break
    
    assert agent.health_state in [HealthState.EXPOSED, HealthState.INFECTIOUS]


def test_health_state_infectious_to_recovered():
    """Test that infectious agents eventually recover"""
    model = EpidemicModel(width=10, height=10, population=1)
    agent = list(model.agents)[0]
    
    agent.health_state = HealthState.INFECTIOUS
    agent.days_infected = 0
    
    # Run enough steps for recovery
    for _ in range(15):
        agent.update_health()
    
    assert agent.health_state == HealthState.RECOVERED


def test_household_creation():
    """Test that households are properly created"""
    model = EpidemicModel(width=10, height=10, population=30, avg_household_size=3)
    
    # Check that agents have household IDs
    household_ids = set()
    for agent in model.agents:
        assert agent.household_id is not None
        household_ids.add(agent.household_id)
    
    # Should have multiple households
    assert len(household_ids) > 1
    
    # Check family members links
    for agent in model.agents:
        assert len(agent.family_members) >= 0
        for family_member in agent.family_members:
            assert family_member.household_id == agent.household_id


def test_torus_wrapping():
    """Test that torus topology correctly wraps distances"""
    model = EpidemicModel(width=10, height=10, population=2)
    agent = list(model.agents)[0]
    
    # On torus, distance from (0,0) to (9,0) should be 1, not 9
    distance = agent.get_distance((0, 0), (9, 0))
    assert distance == 1.0
    
    # Same for vertical
    distance = agent.get_distance((0, 0), (0, 9))
    assert distance == 1.0
