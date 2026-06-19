"""
Test suite for agent transmission logic
"""
import pytest
from model import EpidemicModel
from agents import PersonAgent
from states import HealthState
from agent_types import (
    AgentType,
    FAMILY_INTERACTION_MULTIPLIER,
    FRIEND_INTERACTION_MULTIPLIER,
    MAX_TRANSMISSION_DISTANCE,
)

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

    # is_household_member should be reflexive on the same id and false across
    # different households.
    agents_list = list(model.agents)
    for agent in agents_list:
        for other in agents_list:
            if agent is other:
                continue
            if agent.household_id == other.household_id:
                assert agent.is_household_member(other)
            else:
                assert not agent.is_household_member(other)


def test_friend_relation_is_symmetric():
    """is_friend_of should be symmetric and match friends_id equality."""
    model = EpidemicModel(
        city_map_path=TEST_CITY,
        population=20,
        avg_friend_group_size=3,
    )
    agents_list = list(model.agents)
    for agent in agents_list:
        assert agent.friends_id is not None
        for other in agents_list:
            if agent is other:
                continue
            assert agent.is_friend_of(other) == other.is_friend_of(agent)
            assert agent.is_friend_of(other) == (
                agent.friends_id == other.friends_id
            )


def test_family_relation_uses_family_id_not_household():
    """is_family_member must follow family_id, not household_id."""
    model = EpidemicModel(city_map_path=TEST_CITY, population=4)
    a, b = list(model.agents)[:2]

    # Manually align household but split family.
    a.household_id = b.household_id = 99
    a.family_id = 1
    b.family_id = 2
    assert not a.is_family_member(b)
    assert a.is_household_member(b)

    # And vice versa: same family but different homes.
    a.household_id = 10
    b.household_id = 20
    a.family_id = b.family_id = 7
    assert a.is_family_member(b)
    assert not a.is_household_member(b)


def test_transmission_probability_friend_multiplier():
    """Friend members should have a higher transmission probability."""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    prob_neutral = agent.calculate_transmission_probability(
        distance=2.0, is_family=False, is_friend=False
    )
    prob_friend = agent.calculate_transmission_probability(
        distance=2.0, is_family=False, is_friend=True
    )
    prob_family = agent.calculate_transmission_probability(
        distance=2.0, is_family=True, is_friend=False
    )

    assert prob_friend > prob_neutral
    assert prob_family > prob_friend
    # Family multiplier should win if both are set.
    prob_both = agent.calculate_transmission_probability(
        distance=2.0, is_family=True, is_friend=True
    )
    assert prob_both == prob_family


def test_non_torus_distance_calculation():
    """The city grid is non-torus, so distance is plain Euclidean."""
    model = EpidemicModel(city_map_path=TEST_CITY, population=2)
    agent = list(model.agents)[0]

    # grid is 10x10 but MultiGrid is created with torus=False - opposite
    # corners on the same row should be 9 cells apart, not 1.
    assert agent.get_distance((0, 0), (9, 0)) == 9.0
    assert agent.get_distance((0, 0), (0, 9)) == 9.0
    assert agent.get_distance((1, 2), (4, 6)) == 5.0
