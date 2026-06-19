"""
Test suite for the epidemic model
"""
import pytest
from model import EpidemicModel
from states import CellType, HealthState
from agent_types import AGENT_CONFIG, AgentType

TEST_CITY = "tests/test_city.txt"
TEST_CITY_FULL = "tests/test_city_full.txt"  # includes UNIVERSITY (4) and SCHOOL (5)


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

    # Group agents by household_id and assert agents in the same household
    # consistently share the same id (which is what `is_household_member`
    # uses to detect family relationships).
    by_household = {}
    for agent in model.agents:
        assert agent.household_id is not None
        by_household.setdefault(agent.household_id, []).append(agent)

    for household_id, members in by_household.items():
        for agent in members:
            for other in members:
                if agent is other:
                    continue
                assert agent.is_household_member(other)


def test_friend_group_assignment():
    """Every agent should have a friends_id and consistent group membership."""
    model = EpidemicModel(
        city_map_path=TEST_CITY,
        population=30,
        avg_friend_group_size=4,
    )

    by_friends = {}
    for agent in model.agents:
        assert agent.friends_id is not None
        by_friends.setdefault(agent.friends_id, []).append(agent)

    assert len(by_friends) >= 1
    for members in by_friends.values():
        for agent in members:
            for other in members:
                if agent is other:
                    continue
                assert agent.is_friend_of(other)


def test_family_group_assignment():
    """family_id is a logical relation independent of where you live."""
    model = EpidemicModel(
        city_map_path=TEST_CITY,
        population=30,
        avg_family_size=3,
    )

    by_family = {}
    for agent in model.agents:
        assert agent.family_id is not None
        by_family.setdefault(agent.family_id, []).append(agent)

    assert len(by_family) >= 1
    for members in by_family.values():
        for agent in members:
            for other in members:
                if agent is other:
                    continue
                assert agent.is_family_member(other)


def test_co_residents_share_family_or_friends_id():
    """Every household must seed exactly one shared family_id or friends_id."""
    from agent_types import AgentType

    model = EpidemicModel(
        city_map_path=TEST_CITY,
        population=80,
        avg_household_size=3,
    )

    by_household = {}
    for agent in model.agents:
        by_household.setdefault(agent.household_id, []).append(agent)

    for members in by_household.values():
        all_students = all(a.agent_type == AgentType.STUDENT for a in members)
        family_ids = {a.family_id for a in members}
        friends_ids = {a.friends_id for a in members}
        if all_students:
            # roommate household -> shared friends_id
            assert len(friends_ids) == 1
        else:
            # regular family household -> shared family_id
            assert len(family_ids) == 1


def test_student_only_household_shares_friends_id():
    """A student-only household must seed a shared friends_id (not family)."""
    from agent_types import AgentType

    # 25 agents on a 20-household map guarantees several multi-member homes.
    model = EpidemicModel(city_map_path=TEST_CITY, population=25)
    # Force the entire population to be students and re-run the household
    # seeding so it sees the new type composition.
    for agent in model.agents:
        agent.agent_type = AgentType.STUDENT
        agent.family_id = None
        agent.friends_id = None
    model._next_family_id = 1
    model._next_friends_id = 1
    model._create_households(list(model.agents))

    by_household = {}
    for agent in model.agents:
        by_household.setdefault(agent.household_id, []).append(agent)

    multi_member = [m for m in by_household.values() if len(m) > 1]
    assert multi_member, "test setup failed - no multi-member households"
    for members in multi_member:
        friends_ids = {a.friends_id for a in members}
        assert len(friends_ids) == 1, (
            f"all-student household should share one friends_id, got {friends_ids}"
        )
        # family_id stays unset until _create_family_groups is called
        # afterwards in normal model init.
        assert all(a.family_id is None for a in members)


def test_random_groups_extend_to_unbonded_agents():
    """Agents living alone (or only sharing one type of bond via household)
    should still receive an extended family / friend circle from the random
    group helpers."""
    model = EpidemicModel(
        city_map_path=TEST_CITY,
        population=30,
        avg_family_size=3,
        avg_friend_group_size=4,
    )

    # After full init, every agent has both ids set.
    for agent in model.agents:
        assert agent.family_id is not None
        assert agent.friends_id is not None


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


# ---------------------------------------------------------------------------
# Tests for the UNIVERSITY / SCHOOL day-destination logic
# ---------------------------------------------------------------------------

def test_students_assigned_university_id_on_university_map():
    """Students should receive a university_id when the map has UNIVERSITY cells."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)
    students = [a for a in model.agents if a.agent_type == AgentType.STUDENT]
    if not students:
        pytest.skip("No students were sampled in this run")
    assert all(s.university_id is not None for s in students), (
        "Every student must get a university_id on a map with UNIVERSITY cells"
    )


def test_children_assigned_school_id_on_school_map():
    """Children should receive a school_id when the map has SCHOOL cells."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)
    kids = [a for a in model.agents if a.agent_type == AgentType.CHILDREN]
    if not kids:
        pytest.skip("No children were sampled in this run")
    assert all(c.school_id is not None for c in kids), (
        "Every child must get a school_id on a map with SCHOOL cells"
    )


def test_non_students_have_no_university_id():
    """Workers / seniors / etc. must not be assigned a university_id."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)
    for agent in model.agents:
        if agent.agent_type != AgentType.STUDENT:
            assert agent.university_id is None
        if agent.agent_type != AgentType.CHILDREN:
            assert agent.school_id is None


def test_destination_id_namespaces_are_independent():
    """workplace_id / university_id / school_id are independent id spaces, so
    a workplace_id of 1 and a university_id of 1 do NOT mean the same cell."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)

    workplace_positions = set(model.get_cells(CellType.WORKPLACE))
    university_positions = set(model.get_cells(CellType.UNIVERSITY))
    school_positions = set(model.get_cells(CellType.SCHOOL))

    # All three pools exist and don't overlap.
    assert workplace_positions
    assert university_positions
    assert school_positions
    assert workplace_positions.isdisjoint(university_positions)
    assert workplace_positions.isdisjoint(school_positions)
    assert university_positions.isdisjoint(school_positions)


def test_students_target_university_during_active_hours():
    """At a time inside the student window (8-20), update_agents_based_on_time
    should send students to UNIVERSITY and children to SCHOOL."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)
    model.time_of_day = 10  # well inside both windows

    model.update_agents_based_on_time()
    for agent in model.agents:
        if agent.agent_type == AgentType.STUDENT:
            assert agent.target_destination == CellType.UNIVERSITY
        elif agent.agent_type == AgentType.CHILDREN:
            assert agent.target_destination == CellType.SCHOOL
        elif agent.agent_type in (AgentType.WORKER, AgentType.HEALTHCARE):
            assert agent.target_destination == CellType.WORKPLACE


def test_children_go_home_earlier_than_students():
    """At t=17 (after children's 15:00 end-of-day but before students' 20:00)
    children should be on their way home while students still head to uni."""
    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=80)
    model.time_of_day = 17

    model.update_agents_based_on_time()
    for agent in model.agents:
        if agent.agent_type == AgentType.CHILDREN:
            assert agent.target_destination == CellType.HOUSEHOLD
        elif agent.agent_type == AgentType.STUDENT:
            assert agent.target_destination == CellType.UNIVERSITY


def test_seniors_have_no_schedule():
    """Seniors don't commute - they should have neither work_time nor a
    day destination, and update_agents_based_on_time must leave them alone."""
    cfg = AGENT_CONFIG[AgentType.SENIOR]
    assert cfg.active_hours is None
    assert cfg.active_destination is None

    model = EpidemicModel(city_map_path=TEST_CITY_FULL, population=50)
    seniors = [a for a in model.agents if a.agent_type == AgentType.SENIOR]
    for s in seniors:
        assert s.work_time is None
        assert s.day_destination is None


def test_students_wander_gracefully_on_map_without_university():
    """Maps without UNIVERSITY cells (e.g. the legacy test_city.txt) must not
    crash when students would otherwise commute - they fall back to wandering.
    """
    model = EpidemicModel(city_map_path=TEST_CITY, population=40)
    model.time_of_day = 10
    model.update_agents_based_on_time()
    # Run a few steps to exercise the move() fallback path.
    for _ in range(5):
        model.step()
    # All students stayed alive on the grid.
    students = [a for a in model.agents if a.agent_type == AgentType.STUDENT]
    for s in students:
        assert 0 <= s.pos[0] < model.width
        assert 0 <= s.pos[1] < model.height
        # No university_id was assigned because the map has none.
        assert s.university_id is None


def test_agent_config_active_hours_match_spec():
    """Hard-coded sanity check: students 8-20, children 8-15, workers 9-17."""
    assert AGENT_CONFIG[AgentType.STUDENT].active_hours.start == 8
    assert AGENT_CONFIG[AgentType.STUDENT].active_hours.end == 20
    assert AGENT_CONFIG[AgentType.STUDENT].active_destination == CellType.UNIVERSITY

    assert AGENT_CONFIG[AgentType.CHILDREN].active_hours.start == 8
    assert AGENT_CONFIG[AgentType.CHILDREN].active_hours.end == 15
    assert AGENT_CONFIG[AgentType.CHILDREN].active_destination == CellType.SCHOOL

    assert AGENT_CONFIG[AgentType.WORKER].active_hours.start == 9
    assert AGENT_CONFIG[AgentType.WORKER].active_hours.end == 17
    assert AGENT_CONFIG[AgentType.WORKER].active_destination == CellType.WORKPLACE
