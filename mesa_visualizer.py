import math
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import PropertyLayerStyle
from agents import PersonAgent
from model import EpidemicModel
from states import HealthState
from matplotlib.colors import ListedColormap
import solara

custom_cmap = ListedColormap(["white", "lightgreen", "peachpuff", "lightblue"])


@solara.component
def live_stats_panel(model: EpidemicModel):
    stats = model.get_metrics_snapshot()
    summary = model.get_summary_metrics()
    markdown = (
        f"### Live stats\n"
        f"- Step: {stats['step']}\n"
        f"- Time of day: {stats['time_of_day']:.2f}\n"
        f"- Susceptible: {stats['susceptible']}\n"
        f"- Exposed: {stats['exposed']}\n"
        f"- Infectious: {stats['infectious']}\n"
        f"- Recovered: {stats['recovered']}\n"
        f"- Current infectious ratio: {stats['infected_ratio']:.2%}\n"
        f"- Peak infectious so far: {summary['peak_infectious']} "
        f"(step {summary['peak_infectious_step']})"
    )
    solara.Markdown(markdown)


class MesaVisualizer:
    def __init__(self, model: EpidemicModel, figsize: tuple[int, int] = (12, 9), agent_size: int = 20):
        self.model = model
        self.agent_size = agent_size
        self.figsize = figsize
        self.renderer = SpaceRenderer(
            model,
            backend="matplotlib",
        ).render(
            agent_portrayal=self.agent_portrayal,
            post_process=self.post_process,
            propertylayer_portrayal=self.propertylayer_portrayal
        )
        
    def agent_portrayal(self, agent: PersonAgent):
        color = {
            HealthState.SUSCEPTIBLE: "green",
            HealthState.EXPOSED: "yellow",
            HealthState.INFECTIOUS: "red",
            HealthState.RECOVERED: "gray",
        }[agent.health_state]
        return {
            "color": color,
            "size": self.agent_size,
        }
        
    def post_process(self, ax):
        ax.figure.set_size_inches(*self.figsize)
        
        ax.set_title(f"Epidemic Model Simulation, time: {self.model.time_of_day}")
        
    def propertylayer_portrayal(self, layer):
        return PropertyLayerStyle(colormap=custom_cmap)

    def run(self):
        health_plot = make_plot_component(
            ["Susceptible", "Exposed", "Infectious", "Recovered"],
            backend="matplotlib",
        )

        page = SolaraViz(
            self.model,
            renderer=self.renderer,
            components=[health_plot, lambda: live_stats_panel(self.model)],
        )
        # 3. Custom CSS to force the Solara UI to use the full screen width
        css_style = solara.Style(".v-container { max-width: 100% !important; }")

        # 4. Wrap the page and the CSS style in a Solara Column to display them together
        solara.Column(children=[css_style, page])
        return page

