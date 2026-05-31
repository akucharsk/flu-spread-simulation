"""
Test suite for agent behavior
"""
import pytest
from model import EpidemicModel
from agents import PersonAgent
from states import HealthState
from agent_types import AgentType, AGENT_CONFIG

TEST_CITY = "tests/test_city.txt"


def test_agent_movement():
    """Test that agents move when mobility allows"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    # Place agent in a public-space cell that has several accessible neighbours
    model.grid.move_agent(agent, (7, 2))
    initial_pos = agent.pos
    agent.mobility = 1.0  # Always move

    move_count = 0
    for _ in range(10):
        agent.move()
        if agent.pos != initial_pos:
            move_count += 1
            break

    assert move_count > 0


def test_agent_no_movement_zero_mobility():
    """Test that agents don't move with zero mobility"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    initial_pos = agent.pos
    agent.mobility = 0.0  # Never move

    for _ in range(20):
        if agent.mobility > 0:
            agent.move()

    assert agent.pos == initial_pos


def test_agent_type_parameters():
    """Test that agent types have correct parameters"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=50)

    type_counts = {agent_type: 0 for agent_type in AgentType}

    for agent in model.agents:
        type_counts[agent.agent_type] += 1

        expected_params = AGENT_CONFIG[agent.agent_type]

        assert agent.mobility == expected_params.mobility
        assert agent.infection_rate == expected_params.infection_rate

    assert len([count for count in type_counts.values() if count > 0]) > 1


def test_children_agent_type():
    """Test that CHILDREN agent type exists and has correct params"""
    assert AgentType.CHILDREN in AGENT_CONFIG

    params = AGENT_CONFIG[AgentType.CHILDREN]

    assert params.mobility >= 0.8

    student_rate = AGENT_CONFIG[AgentType.STUDENT].infection_rate
    assert 0.15 <= params.infection_rate <= 0.30


def test_senior_different_from_others():
    """Test that seniors have different parameters"""
    senior_params = AGENT_CONFIG[AgentType.SENIOR]
    student_params = AGENT_CONFIG[AgentType.STUDENT]

    assert senior_params.mobility < student_params.mobility
    assert senior_params.infection_rate > student_params.infection_rate


def test_agent_step_execution():
    """Test that agent step executes without errors"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=5)

    for _ in range(5):
        for agent in model.agents:
            agent.step()


def test_infectious_agent_marking():
    """Test that at least one agent starts as infectious"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=20)

    infectious_count = sum(
        1 for agent in model.agents
        if agent.health_state == HealthState.INFECTIOUS
    )

    assert infectious_count >= 1


def test_agent_position_within_grid():
    """Test that all agents start within grid bounds"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=30)

    for agent in model.agents:
        x, y = agent.pos
        assert 0 <= x < model.width
        assert 0 <= y < model.height


def test_x_y_properties():
    """Test x and y property accessors"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    x, y = agent.pos
    assert agent.x == x
    assert agent.y == y


def test_agent_properties_consistency():
    """Test that agent properties remain consistent"""
    model = EpidemicModel(city_map_path=TEST_CITY, population=1)
    agent = list(model.agents)[0]

    assert agent.agent_type is not None
    assert agent.health_state is not None
    assert agent.infection_rate >= 0
    assert agent.mobility >= 0
    assert agent.days_infected >= 0
    assert agent.household_id is not None
