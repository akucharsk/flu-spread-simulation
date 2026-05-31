# Flu Spread Simulation

Agent-based epidemic simulation modeling how influenza spreads across society through spatial proximity, family networks, and dynamic social interactions.

## 📋 Project Overview

This project uses **Mesa** (Multi-Agent Simulator with Python) to simulate disease transmission in a population. Agents have different characteristics and interact based on proximity and social networks, creating realistic epidemic dynamics.

## 🧬 Agent Types

| Type | Mobility | Infection Rate | Use Case |
|------|----------|-----------------|----------|
| **STUDENT** | 0.9 (high) | 0.20 | University/school students - frequent movement |
| **WORKER** | 0.8 (medium) | 0.25 | Office/factory workers - regular commute |
| **CHILDREN** | 0.9 (high) | 0.22 | School-age children - high activity |
| **SENIOR** | 0.3 (low) | 0.35 | Elderly population - limited mobility, higher susceptibility |
| **HEALTHCARE** | 1.0 (very high) | 0.15 | Healthcare workers - constant movement, lower infection rate |

- **Mobility**: Probability of moving to adjacent cell per simulation step
- **Infection Rate**: Base transmission probability when in same cell as infectious agent

## 🦠 Disease Transmission Model

### Transmission Mechanisms

The model implements **three-layer transmission**:

#### 1. **Family/Household Interactions** (Static Networks)
- Household members have **2x higher transmission risk** (multiplier)
- Static links created during initialization based on household groups
- Agents remain in same family throughout simulation

#### 2. **Environmental/Spatial Transmission** (Same Cell)
- When infectious and susceptible agents occupy **same cell**: distance = 0
- Highest transmission probability
- Represents close contact (work, public transport, stores)

#### 3. **Distance-Based Transmission** (Neighboring Cells)
- Agents in neighboring cells interact with reduced probability
- Linear decay function: transmission probability decreases linearly with distance
- **Maximum transmission distance**: 5 cells
- Formula: `probability = max(0, 1 - distance/5) × infection_rate × [2.0 if family else 1.0]`

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

## 📊 Simulation Parameters (from `agent_types.py`)

```python
FAMILY_INTERACTION_MULTIPLIER = 2.0      # Family risk multiplier
MAX_TRANSMISSION_DISTANCE = 5             # Cells for spatial transmission
DISTANCE_DECAY_FUNCTION = "linear"        # Linear probability decay
```

## ⚙️ Configuration (`config.json`)

| Key | Type | Description |
|-----|------|-------------|
| `cityMapPath` | string | Path to the city map text file (cell codes: `0`=default, `1`=household, `2`=workplace, `3`=public space) |
| `population` | int | Number of agents to simulate |
| `steps` | int \| null | Steps to run headless; `null` launches the interactive visualizer |
| `figsize` | [w, h] | Visualizer window size in inches |
| `agentSize` | int | Rendered agent dot size |
| `startTime` | float | Starting time of day in 24 h format (e.g. `10` = 10:00) |
| `timestep` | float | Hours advanced per simulation step (e.g. `0.1`) |

Example:
```json
{
  "cityMapPath": "city1.txt",
  "population": 10000,
  "steps": null,
  "figsize": [20, 20],
  "agentSize": 10,
  "startTime": 10,
  "timestep": 0.1
}
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python main.py
```

## 📁 Project Structure

```
├── agent_types.py      # Agent types and configuration (STUDENT, WORKER, CHILDREN, SENIOR, HEALTHCARE)
├── agents.py           # PersonAgent implementation with disease logic
├── model.py            # EpidemicModel - grid, households, initialization
├── states.py           # HealthState enum (SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED)
├── main.py             # Entry point for simulation
├── drawer.py           # Visualization rendering
├── visualization.py    # Visualization setup
├── requirements.txt    # Python dependencies
└── tests/              # Unit and functional tests
    ├── test_transmission.py
    ├── test_agents.py
    └── test_model.py
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

- `mesa` - Agent-based modeling framework
- `pygame` - Visualization (optional)

## 📝 License

MIT
