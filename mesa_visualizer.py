from mesa.visualization import SolaraViz, SpaceRenderer, make_space_component
from mesa.visualization.components import PropertyLayerStyle
from agents import PersonAgent
from model import EpidemicModel
from states import HealthState
import solara

def agent_portrayal(agent: PersonAgent):
    color = {
        HealthState.SUSCEPTIBLE: "green",
        HealthState.EXPOSED: "yellow",
        HealthState.INFECTIOUS: "red",
        HealthState.RECOVERED: "gray",
    }[agent.health_state]
    return {
        "color": color,
        "size": 5,
    }
    
def propertylayer_portrayal(layer):
    print(layer.__dict__)
    return PropertyLayerStyle(color="lightblue", colorbar=False)
    
def post_process(ax):
    # Set width and height in inches (e.g., 12x12)
    ax.figure.set_size_inches(12, 9)
    
    # Optional: You can also use this hook to add a title or hide axes
    ax.set_title("Epidemic Model Simulation")

model = EpidemicModel()
renderer = SpaceRenderer(
  model,
  backend="matplotlib",
).render(
  agent_portrayal=agent_portrayal,
  post_process=post_process,
  propertylayer_portrayal=propertylayer_portrayal
)

page = SolaraViz(
    model,
    renderer=renderer,
    components=[],
)

# 3. Custom CSS to force the Solara UI to use the full screen width
css_style = solara.Style(".v-container { max-width: 100% !important; }")

# 4. Wrap the page and the CSS style in a Solara Column to display them together
solara.Column(children=[css_style, page])
