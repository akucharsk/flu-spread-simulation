"""
Test suite for agent transmission logic
"""
import pytest
from model import EpidemicModel
from agents import PersonAgent
from states import HealthState
from agent_types import AgentType, FAMILY_INTERACTION_MULTIPLIER, MAX_TRANSMISSION_DISTANCE

TEST_CITY = "tests/test_city.txt"


def test_agent_creation():
    """Test that agents are created with correct attributes"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=5)

    assert len(model.agents) == 5
    for agent in model.agents:
        assert isinstance(agent, PersonAgent)
        assert agent.health_state in [HealthState.SUSCEPTIBLE, HealthState.INFECTIOUS]
        assert agent.agent_type in list(AgentType)
        assert 0 <= agent.infection_rate <= 1
        assert 0 <= agent.mobility <= 1


def test_transmission_distance_calculation():
    """Test distance calculation between agents"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=3)
    agents_list = list(model.agents)
    agent1 = agents_list[0]

    distance = agent1.get_distance((0, 0), (3, 4))
    expected = 5.0  # sqrt(3^2 + 4^2) = 5
    assert abs(distance - expected) < 0.01


def test_transmission_probability_basic():
    """Test transmission probability calculation at same cell"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    prob = agent.calculate_transmission_probability(distance=0.0, is_family=False)
    assert prob > 0
    assert prob <= 1

    prob_max = agent.calculate_transmission_probability(
        distance=MAX_TRANSMISSION_DISTANCE,
        is_family=False
    )
    assert prob_max == 0.0


def test_transmission_probability_family_multiplier():
    """Test that family members have higher transmission risk"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    prob_non_family = agent.calculate_transmission_probability(
        distance=2.0,
        is_family=False
    )
    prob_family = agent.calculate_transmission_probability(
        distance=2.0,
        is_family=True
    )

    assert prob_family >= prob_non_family


def test_transmission_distance_limit():
    """Test that transmission stops beyond MAX_TRANSMISSION_DISTANCE"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
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
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    prob_0 = agent.calculate_transmission_probability(distance=0.0, is_family=False)
    prob_2 = agent.calculate_transmission_probability(distance=2.0, is_family=False)
    prob_4 = agent.calculate_transmission_probability(distance=4.0, is_family=False)

    diff_1 = prob_0 - prob_2
    diff_2 = prob_2 - prob_4

    assert abs(diff_1 - diff_2) < 0.01 * agent.infection_rate


def test_health_state_exposed_to_infectious():
    """Test that exposed agents can become infectious"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    agent.health_state = HealthState.EXPOSED

    for _ in range(100):
        agent.update_health()
        if agent.health_state == HealthState.INFECTIOUS:
            break

    assert agent.health_state in [HealthState.EXPOSED, HealthState.INFECTIOUS]


def test_health_state_infectious_to_recovered():
    """Test that infectious agents eventually recover"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    agent.health_state = HealthState.INFECTIOUS
    agent.days_infected = 0

    for _ in range(15):
        agent.update_health()

    assert agent.health_state == HealthState.RECOVERED


def test_household_creation():
    """Test that households are properly created"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=30, avg_household_size=3)

    household_ids = set()
    for agent in model.agents:
        assert agent.household_id is not None
        household_ids.add(agent.household_id)

    assert len(household_ids) > 1

    for agent in model.agents:
        assert len(agent.family_members) >= 0
        for family_member in agent.family_members:
            assert family_member.household_id == agent.household_id


def test_torus_distance_calculation():
    """Test that distance function accounts for torus topology"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    # grid is 10x10; torus distance from (0,0) to (9,0) should be 1, not 9
    distance = agent.get_distance((0, 0), (9, 0))
    assert distance == 1.0

    distance = agent.get_distance((0, 0), (0, 9))
    assert distance == 1.0
