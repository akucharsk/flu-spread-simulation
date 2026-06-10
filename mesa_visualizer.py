from datetime import datetime
from pathlib import Path

import solara
from matplotlib.colors import ListedColormap
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.utils import update_counter

from agents import PersonAgent
from agent_types import AgentType
from analytics import export_live_snapshot
from model import EpidemicModel
from states import HealthState

custom_cmap = ListedColormap(["white", "lightgreen", "peachpuff", "lightblue"])


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


@solara.component
def LiveStatsCard(model: EpidemicModel):
    """Card with live counts and aggregate metrics that refreshes every step."""
    update_counter.get()  # subscribe to model step updates so this re-renders

    stats = model.get_metrics_snapshot()
    summary = model.get_summary_metrics()
    by_type = model.get_health_counts_by_type()

    population = max(stats["population"], 1)
    susceptible_pct = stats["susceptible"] / population
    exposed_pct = stats["exposed"] / population
    infectious_pct = stats["infectious"] / population
    recovered_pct = stats["recovered"] / population

    with solara.Card("Live statistics", style={"height": "100%"}):
        with solara.Columns([1, 1]):
            with solara.Column():
                solara.Markdown(
                    f"**Step**: {stats['step']}  \n"
                    f"**Time of day**: {stats['time_of_day']:.2f} h  \n"
                    f"**Population**: {_format_int(stats['population'])}  \n"
                    f"**New exposures (last step)**: {_format_int(stats['new_exposures'])}"
                )
            with solara.Column():
                solara.Markdown(
                    f"**Peak infectious**: {_format_int(summary['peak_infectious'])} "
                    f"(step {summary['peak_infectious_step']})  \n"
                    f"**Cumulative infected**: {_format_int(stats['cumulative_infected'])} "
                    f"({stats['cumulative_ratio']:.1%})  \n"
                    f"**Currently infectious**: {_format_int(stats['infectious'])} "
                    f"({infectious_pct:.1%})"
                )

        solara.Markdown("---")

        with solara.Columns([1, 1, 1, 1]):
            with solara.Column():
                solara.Markdown(
                    f"**Susceptible**  \n"
                    f"{_format_int(stats['susceptible'])}  \n"
                    f"{susceptible_pct:.1%}"
                )
            with solara.Column():
                solara.Markdown(
                    f"**Exposed**  \n"
                    f"{_format_int(stats['exposed'])}  \n"
                    f"{exposed_pct:.1%}"
                )
            with solara.Column():
                solara.Markdown(
                    f"**Infectious**  \n"
                    f"{_format_int(stats['infectious'])}  \n"
                    f"{infectious_pct:.1%}"
                )
            with solara.Column():
                solara.Markdown(
                    f"**Recovered**  \n"
                    f"{_format_int(stats['recovered'])}  \n"
                    f"{recovered_pct:.1%}"
                )

        solara.Markdown("---")
        solara.Markdown("**Infectious by agent type**")

        type_lines = []
        for agent_type in AgentType:
            counts = by_type[agent_type.value]
            total = sum(counts.values())
            inf = counts["infectious"]
            ratio = inf / total if total else 0.0
            type_lines.append(
                f"- {agent_type.value.capitalize()}: "
                f"{_format_int(inf)} infectious / {_format_int(total)} ({ratio:.1%})"
            )
        solara.Markdown("\n".join(type_lines))


@solara.component
def ExportPanel(model: EpidemicModel):
    """Interactive panel that lets the user export analytics on demand."""
    update_counter.get()  # so the displayed step in the hint stays current

    base_dir = solara.use_reactive("output/live_export")
    last_message = solara.use_reactive("")
    is_error = solara.use_reactive(False)

    def do_export():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path(base_dir.value) / f"step_{model.current_step:04d}_{timestamp}"
            artifacts = export_live_snapshot(model, str(target))
            is_error.value = False
            last_message.value = f"Exported {len(artifacts['plots'])} plots and 3 CSVs to: {artifacts['output_dir']}"
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            is_error.value = True
            last_message.value = f"Export failed: {exc}"

    with solara.Card("Generate visualization & data export", style={"height": "100%"}):
        solara.Markdown(
            "Click the button below at any time during the simulation to dump the "
            "current run's metrics history (CSV) and matplotlib plots used for "
            "analysis. Each click creates a new timestamped subfolder."
        )
        solara.InputText(
            label="Output directory (created if missing)",
            value=base_dir,
        )
        solara.Markdown(
            f"Current step: **{model.current_step}** &middot; "
            f"recorded snapshots: **{len(model.metrics_history)}**"
        )
        solara.Button(
            label="Generate visualization",
            on_click=do_export,
            color="primary",
            icon_name="mdi-chart-line",
        )

        if last_message.value:
            if is_error.value:
                solara.Error(last_message.value)
            else:
                solara.Success(last_message.value)


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
            propertylayer_portrayal=self.propertylayer_portrayal,
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

    @staticmethod
    def _make_post_process(title: str, ylabel: str):
        def post_process(ax):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        return post_process

    def _build_plot_components(self):
        """Create the matplotlib plot components shown next to the simulation."""
        seir_plot = make_plot_component(
            {
                "Susceptible": "tab:green",
                "Exposed": "tab:orange",
                "Infectious": "tab:red",
                "Recovered": "tab:gray",
            },
            backend="matplotlib",
            post_process=self._make_post_process("Health states (S/E/I/R)", "Agents"),
        )

        new_exposures_plot = make_plot_component(
            {"New_Exposures": "tab:purple"},
            backend="matplotlib",
            post_process=self._make_post_process("New exposures per step (S->E)", "Agents"),
        )

        cumulative_plot = make_plot_component(
            {"Cumulative_Infected": "tab:red"},
            backend="matplotlib",
            post_process=self._make_post_process(
                "Cumulative ever-infected (population - susceptible)", "Agents"
            ),
        )

        type_color_map = {
            AgentType.STUDENT: "tab:blue",
            AgentType.WORKER: "tab:orange",
            AgentType.SENIOR: "tab:red",
            AgentType.HEALTHCARE: "tab:green",
            AgentType.CHILDREN: "tab:purple",
        }
        infectious_by_type_plot = make_plot_component(
            {f"Infectious_{t.value}": color for t, color in type_color_map.items()},
            backend="matplotlib",
            post_process=self._make_post_process(
                "Currently infectious by agent type", "Infectious agents"
            ),
        )

        return seir_plot, new_exposures_plot, cumulative_plot, infectious_by_type_plot

    def run(self):
        seir_plot, new_exposures_plot, cumulative_plot, infectious_by_type_plot = (
            self._build_plot_components()
        )

        # make_plot_component already returns a (callable, page) tuple, so it
        # can be added to the components list as-is. Custom Solara components
        # are wrapped in (component, page) tuples manually.
        components = [
            (LiveStatsCard, 0),
            seir_plot,
            new_exposures_plot,
            cumulative_plot,
            infectious_by_type_plot,
            (ExportPanel, 0),
        ]

        page = SolaraViz(
            self.model,
            renderer=self.renderer,
            components=components,
            name="Flu Spread Simulation",
        )

        # Force the Solara container to use full screen width so the draggable
        # grid of charts has enough room.
        css_style = solara.Style(".v-container { max-width: 100% !important; }")
        solara.Column(children=[css_style, page])
        return page
