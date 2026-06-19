# Flu Spread Simulation

Agent-based epidemic simulation modeling how influenza spreads across society through spatial proximity, family networks, and dynamic social interactions.

Now includes:
- live interactive S/E/I/R charts during visualization,
- automatic export of analytics data (CSV/JSON),
- static plots generated from single or multiple simulation runs.

## 📋 Project Overview

This project uses **Mesa** (Multi-Agent Simulator with Python) to simulate disease transmission in a population. Agents have different characteristics and interact based on proximity and social networks, creating realistic epidemic dynamics.

## 🧬 Agent Types

| Type | Mobility | Infection Rate | Day Destination | Active Hours | Use Case |
|------|----------|-----------------|------------------|--------------|----------|
| **STUDENT** | 0.9 (high) | 0.20 | `UNIVERSITY` | 08:00 – 20:00 | University students - long study day |
| **WORKER** | 0.8 (medium) | 0.25 | `WORKPLACE` | 09:00 – 17:00 | Office/factory workers - regular commute |
| **CHILDREN** | 0.9 (high) | 0.22 | `SCHOOL` | 08:00 – 15:00 | School-age children - shorter school day |
| **SENIOR** | 0.3 (low) | 0.35 | – | – | Elderly population - no commute, stay around home |
| **HEALTHCARE** | 1.0 (very high) | 0.15 | `WORKPLACE` | 09:00 – 17:00 | Healthcare workers - constant movement, lower infection rate |

- **Mobility**: Probability of moving to adjacent cell per simulation step
- **Infection Rate**: Base transmission probability when in same cell as infectious agent
- **Day Destination / Active Hours**: each agent type has its own schedule; outside the active window they head back to their `HOUSEHOLD`. Maps that lack a given destination type (e.g. no `UNIVERSITY` cells) make the affected agents wander instead — no crash.

## 🦠 Disease Transmission Model

### Transmission Mechanisms

The model implements **three-layer transmission**:

Each agent carries four static IDs assigned at model start:

| ID | Meaning | Source | Transmission boost |
|----|---------|--------|--------------------|
| `household_id` | Where the agent **lives** (physical home cell) | `_create_households` | none on its own |
| `workplace_id` | Where the agent **works** | `_create_workplaces` (only WORKER/HEALTHCARE) | none on its own |
| `family_id` | Logical **family / relatives** | `_create_family_groups` (random, ~`avg_family_size`) | **2.0x** |
| `friends_id` | **Friend circle** | `_create_friend_groups` (random, ~`avg_friend_group_size`) | **1.5x** |

Important: `family_id` is **independent** of `household_id` — relatives can live in different households (e.g. parents, siblings who've moved out).

### How co-residents get socially linked

Every household always seeds **one** logical link between its residents:

- **All-student household** (everyone is `AgentType.STUDENT`) → all residents share a `friends_id` (roommate / dorm scenario, 1.5x). Their `family_id` is then filled in later by the random family-group helper — their family lives elsewhere.
- **Any other household** → all residents share a `family_id` (regular family unit, 2x). Even a household with one student and their parents is treated this way.

Once the household-seeded ids are in place, `_create_family_groups` and `_create_friend_groups` only fill in the *missing* ids (single-person households, students that need an extended family group elsewhere, etc.), keeping id spaces disjoint via a continued counter.

#### 1. **Family Network** (Logical, 2x)
- `agent.is_family_member(other)` ⇔ same `family_id`.
- Independent of physical residence.

#### 2. **Friend Network** (Logical, 1.5x)
- `agent.is_friend_of(other)` ⇔ same `friends_id`.
- If two agents are both family and friends, the family multiplier wins.

#### 3. **Spatial Transmission** (Same / Neighbouring Cells)
- Each step, every infectious agent sweeps the radius-5 Moore neighbourhood (centre included) and infects susceptibles with a linear-decay probability.
- Formula: `probability = max(0, 1 - distance/5) × infection_rate × multiplier`
- `multiplier = 2.0` if family, else `1.5` if friend, else `1.0`.
- **Maximum transmission distance**: 5 cells.

### Health States

```
SUSCEPTIBLE → EXPOSED → INFECTIOUS → RECOVERED
                ↓(20% chance per step)
                INFECTIOUS (lasts ~10 days)
```

## 🗺️ Grid Configuration

- **Grid Type**: Toroidal 2D grid (wraps around edges)
- **Default Size**: 20×20 cells
- **Default Population**: 100 agents
- **Average Household Size**: 3 members
- **Initial Condition**: 1 infected agent ("patient zero")

### Cell types

| Code | `CellType` | Who can enter | Notes |
|------|------------|---------------|-------|
| `0` | `DEFAULT` | anyone | **Background tile** — visually represents streets, sidewalks and any open public space agents can roam through. Also acts as the safety placeholder for not-yet-implemented place types. |
| `1` | `HOUSEHOLD` | only residents (`household_id`) | Where each agent goes when off the clock. |
| `2` | `WORKPLACE` | only the agents assigned there (`workplace_id`) | Day destination for `WORKER` / `HEALTHCARE`. |
| `3` | `PUBLIC_SPACE` | anyone | Parks, plazas, sidewalks - any agent may wander through. |
| `4` | `UNIVERSITY` | only the students assigned there (`university_id`) | Day destination for `STUDENT` (08-20). |
| `5` | `SCHOOL` | only the children assigned there (`school_id`) | Day destination for `CHILDREN` (08-15). |

`workplace_id`, `university_id` and `school_id` are independent id spaces — workplace #1 and university #1 are unrelated buildings.

### Bundled map presets (`maps/`)

All maps are square. Buildings are blob-grown procedurally, so no two
buildings look like clean rectangles; streets between buildings vary in
width. Re-run `python maps/_generate_maps.py` to regenerate them (each
preset uses a fixed seed for reproducibility).

| Preset key | Size | Theme |
|------------|------|-------|
| `campus_30` | 30×30 | Small academic campus: two large lecture halls, a school, dormitories. |
| `suburban_town_60` | 60×60 | Quiet town: scattered houses, two schools, one university, a few offices. |
| `mixed_district_100` | 100×100 | Balanced urban district: residential blocks, offices, schools and parks intermixed. |
| `industrial_corridor_150` | 150×150 | Work-heavy belt: large industrial sites, worker housing, two schools. |
| `megacity_200` | 200×200 | Full metropolitan area with every infrastructure type and wide arterial roads. |

## 📊 Simulation Parameters (from `agent_types.py`)

```python
FAMILY_INTERACTION_MULTIPLIER = 2.0       # Same-household multiplier
FRIEND_INTERACTION_MULTIPLIER = 1.5       # Same friend-group multiplier
MAX_TRANSMISSION_DISTANCE = 5             # Cells for spatial transmission
DISTANCE_DECAY_FUNCTION = "linear"        # Linear probability decay
DEFAULT_AVG_FRIEND_GROUP_SIZE = 4         # Average random friend-group size
```

## ⚙️ Configuration (`config.json`)

| Key | Type | Description |
|-----|------|-------------|
| `cityMapPath` | string | Path to the city map text file (cell codes: `0`=default/street (safety placeholder), `1`=household, `2`=workplace, `3`=public space, `4`=university, `5`=school) |
| `population` | int | Number of agents to simulate |
| `steps` | int \| null | Steps to run headless; `null` launches the interactive visualizer |
| `runs` | int | Number of independent runs for batch analytics (headless mode) |
| `outputDir` | string | Folder where CSV/JSON and static plots are saved |
| `figsize` | [w, h] | Visualizer window size in inches |
| `agentSize` | int | Rendered agent dot size (Pygame uses it as the dot radius hint) |
| `windowSize` | [w, h] | Optional Pygame window size in pixels (default `[1600, 900]`) |
| `startTime` | float | Starting time of day in 24 h format (e.g. `10` = 10:00) |
| `timestep` | float | Hours advanced per simulation step (e.g. `0.1`) |
| `verbose` | bool | Print per-step internal time updates to stdout |

Example:
```json
{
  "cityMapPath": "city1.txt",
  "population": 10000,
   "steps": 300,
   "runs": 5,
   "outputDir": "output",
  "figsize": [20, 20],
  "agentSize": 10,
  "startTime": 10,
   "timestep": 0.1,
   "verbose": false
}
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run headless simulation (single run, exports CSV/JSON/plots)
python main_simulation.py --steps 100 --runs 1 --output-dir output

# Run batch experiment for analysis (e.g. for grade 5.0 requirements)
python main_simulation.py --steps 300 --runs 20 --output-dir output/experiment_01

# Run with interactive Pygame visualization (local window)
python main_visualization.py
```

## 📈 Analytics Outputs

After `main_simulation.py` finishes, these files are generated in `outputDir`:

- `simulation_timeseries.csv` - full per-step data for every run (`run_id`, `step`, `time_of_day`, `S/E/I/R`, ratios)
- `simulation_runs_summary.csv` - one row per run (peak infections, final states, total infected)
- `simulation_aggregated_by_step.csv` - per-step mean/std across runs
- `simulation_metadata.json` - experiment metadata and aggregate metrics
- `health_states_mean_std.png` - mean +/- std chart for S/E/I/R over time
- `infectious_per_run.png` - infectious curve for each run
- `peak_infectious_histogram.png` - distribution of infection peaks across runs

## 🖥️ Live Interactive Stats (Pygame)

`main_visualization.py` opens a local Pygame window containing:
- a coloured agent map (cells coloured by type, agents coloured by health state),
- a live statistics panel (step, time, peak infectious, S/E/I/R bars, infectious per agent type),
- four live matplotlib plots (S/E/I/R curves, new exposures, cumulative infected, infectious by agent type),
- a sidebar with simulation controls (Play / Step / Reset), model-parameter sliders, map-preset dropdown and an "Generate visualization" export button.

Keyboard shortcuts: `Space` = play/pause, `→` or `N` = single step, `R` = reset, `E` = export.

## 📁 Project Structure

```
├── agent_types.py          # Agent types and configuration (STUDENT, WORKER, CHILDREN, SENIOR, HEALTHCARE)
├── agents.py               # PersonAgent implementation with disease logic
├── model.py                # EpidemicModel - grid, households, initialization
├── states.py               # HealthState enum (SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED)
├── main_simulation.py      # Headless entry point  (python main_simulation.py --steps N)
├── main_visualization.py   # Interactive entry point (python main_visualization.py)
├── pygame_visualizer.py    # Pygame GUI: map + stats + plots + controls
├── city_utils.py           # City map loading and grid construction
├── config.json             # Default simulation configuration
├── requirements.txt        # Python dependencies
└── tests/                  # Unit and functional tests
    ├── test_transmission.py
    ├── test_agents.py
    ├── test_model.py
    └── test_city.txt       # Minimal city map fixture used by tests
```

## 🔄 Agent Behavior (Per Step)

Each agent performs three actions in `PersonAgent.step()`:

1. **Movement** (based on mobility parameter)
   - 90% chance: STUDENT moves to adjacent cell
   - 80% chance: WORKER moves
   - 30% chance: SENIOR moves
   - Etc.

2. **Interaction** (disease transmission)
   - Interact with family members (static network, high risk)
   - Interact with agents in same cell (same location)
   - Interact with agents in neighboring cells (distance-based)

3. **Health Update**
   - EXPOSED → INFECTIOUS (20% per step)
   - INFECTIOUS → RECOVERED (after >10 days)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_transmission.py -v
```

## 📈 Future Enhancements

- [ ] Visualization with real-time plotting
- [ ] Vaccination mechanism
- [ ] Age groups with differential mortality
- [ ] Geographic regions with travel
- [ ] Weather/seasonal effects on transmission
- [ ] Public health interventions (quarantine, masks)

## 📚 Dependencies

- `mesa` - Agent-based modeling framework (model, grid, scheduling, data collection)
- `pygame` - Interactive visualization window
- `matplotlib` - Static plots and the live plots blitted into Pygame
- `pandas` - Required transitively by mesa data collection

## 📝 License

MIT
