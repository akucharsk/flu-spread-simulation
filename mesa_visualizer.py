from datetime import datetime
from pathlib import Path

import solara
from matplotlib.colors import ListedColormap
from mesa.visualization import SpaceRenderer, make_plot_component
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.solara_viz import ModelController, ModelCreator, SpaceRendererComponent
from mesa.visualization.utils import update_counter

from agents import PersonAgent
from agent_types import AgentType
from analytics import export_live_snapshot
from map_presets import PREDEFINED_CITY_MAPS
from model import EpidemicModel
from states import HealthState

custom_cmap = ListedColormap(["white", "lightgreen", "peachpuff", "lightblue"])
INFECTIOUS_COLOR = "tab:red"


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


@solara.component
def live_stats_content(model: EpidemicModel):
    """Live counts and aggregate metrics that refresh every step."""
    update_counter.get()  # subscribe to model step updates so this re-renders

    stats = model.get_metrics_snapshot()
    summary = model.get_summary_metrics()
    by_type = model.get_health_counts_by_type()

    population = max(stats["population"], 1)
    susceptible_pct = stats["susceptible"] / population
    exposed_pct = stats["exposed"] / population
    infectious_pct = stats["infectious"] / population
    recovered_pct = stats["recovered"] / population

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
def right_panel(model: EpidemicModel, model_parameters: solara.Reactive[dict]):
    """Right-side panel: map presets and export actions."""
    update_counter.get()  # so the displayed step in the hint stays current

    base_dir = solara.use_reactive("output/live_export")
    last_message = solara.use_reactive("")
    is_error = solara.use_reactive(False)

    label_to_key = {data["label"]: key for key, data in PREDEFINED_CITY_MAPS.items()}
    labels = list(label_to_key.keys())
    initial_label = PREDEFINED_CITY_MAPS.get(
        model.city_map_preset,
        {"label": model.map_name},
    )["label"]
    selected_map_label = solara.use_reactive(initial_label)

    def on_map_change(label: str):
        selected_map_label.value = label
        selected_key = label_to_key[label]
        model_parameters.value = {
            **model_parameters.value,
            "city_map_preset": selected_key,
        }

    def do_export():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path(base_dir.value) / f"step_{model.current_step:04d}_{timestamp}"
            artifacts = export_live_snapshot(
                model,
                str(target),
                config={
                    "cityMapPath": model.city_map_path,
                    "cityMapPreset": model.city_map_preset,
                    "mapName": model.map_name,
                },
            )
            is_error.value = False
            last_message.value = f"Exported {len(artifacts['plots'])} plots and 3 CSVs to: {artifacts['output_dir']}"
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            is_error.value = True
            last_message.value = f"Export failed: {exc}"

    with solara.Card("Map presets & export", classes=["sidebar-card-compact"], style={"height": "100%"}):
        solara.Select(
            "",
            value=selected_map_label,
            values=labels,
            on_value=on_map_change,
        )

        solara.InputText(
            label="",
            value=base_dir,
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
                "Infectious": INFECTIOUS_COLOR,
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
            {"Cumulative_Infected": INFECTIOUS_COLOR},
            backend="matplotlib",
            post_process=self._make_post_process(
                "Cumulative ever-infected (population - susceptible)", "Agents"
            ),
        )

        type_color_map = {
            AgentType.STUDENT: "tab:blue",
            AgentType.WORKER: "tab:orange",
            AgentType.SENIOR: INFECTIOUS_COLOR,
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

    @staticmethod
    def _plot_callable(plot_component):
        if isinstance(plot_component, tuple):
            return plot_component[0]
        return plot_component

    def _build_model_params(self):
        return {
            "city_map_path": self.model.city_map_path,
            "city_map_preset": self.model.city_map_preset,
            "population": {
                "type": "SliderInt",
                "label": "Population",
                "value": len(self.model.agents),
                "min": 200,
                "max": 20000,
                "step": 200,
            },
            "avg_household_size": {
                "type": "SliderInt",
                "label": "Average household size",
                "value": int(self.model.avg_household_size),
                "min": 1,
                "max": 8,
                "step": 1,
            },
            "time_of_day": {
                "type": "SliderFloat",
                "label": "Start time (h)",
                "value": float(self.model.time_of_day),
                "min": 0.0,
                "max": 23.5,
                "step": 0.5,
            },
            "timestep": {
                "type": "SliderFloat",
                "label": "Time step (h)",
                "value": float(self.model.timestep),
                "min": 0.05,
                "max": 2.0,
                "step": 0.05,
            },
            "verbose": {
                "type": "Checkbox",
                "label": "Verbose logs",
                "value": bool(self.model.verbose),
            },
        }

    def run(self):
        seir_plot, new_exposures_plot, cumulative_plot, infectious_by_type_plot = (
            self._build_plot_components()
        )
        model_params = self._build_model_params()

        @solara.component
        def page():
            model = solara.use_reactive(self.model)
            renderer = solara.use_reactive(self.renderer)

            reactive_model_parameters = solara.use_reactive({})
            reactive_play_interval = solara.use_reactive(100)
            reactive_render_interval = solara.use_reactive(1)
            reactive_use_threads = solara.use_reactive(False)

            with solara.AppBar():
                solara.AppBarTitle("Flu Spread Simulation")
                solara.lab.ThemeToggle()

            with solara.Sidebar(), solara.Column():
                with solara.Card("Controls", classes=["sidebar-card-compact"]):
                    solara.SliderInt(
                        label="Play Interval (ms)",
                        value=reactive_play_interval,
                        on_value=lambda v: reactive_play_interval.set(v),
                        min=1,
                        max=500,
                        step=10,
                    )
                    solara.SliderInt(
                        label="Render Interval (steps)",
                        value=reactive_render_interval,
                        on_value=lambda v: reactive_render_interval.set(v),
                        min=1,
                        max=100,
                        step=2,
                    )
                    solara.Checkbox(
                        label="Use Threads",
                        value=reactive_use_threads,
                        on_value=lambda v: reactive_use_threads.set(v),
                    )
                    ModelController(
                        model,
                        renderer=renderer,
                        model_parameters=reactive_model_parameters,
                        play_interval=reactive_play_interval,
                        render_interval=reactive_render_interval,
                        use_threads=reactive_use_threads,
                    )

                with solara.Card("Model Parameters", classes=["sidebar-card-compact", "sidebar-model-compact"]):
                    ModelCreator(
                        model,
                        model_params,
                        model_parameters=reactive_model_parameters,
                    )

                right_panel(model.value, reactive_model_parameters)

            solara.Style(
                """
                .v-container { max-width: 100% !important; }
                .simulation-main { padding: 16px; }
                .visualization-card { margin-top: 22px; }

                .sidebar-card-compact .v-card__title {
                    padding: 12px 14px 0 !important;
                }
                .sidebar-card-compact .v-card__text {
                    padding: 8px 14px 12px !important;
                }
                .sidebar-card-compact .v-input {
                    margin-top: 4px !important;
                    padding-top: 0 !important;
                }
                .sidebar-card-compact .v-messages {
                    min-height: 0 !important;
                }
                .sidebar-card-compact .v-btn {
                    min-height: 32px !important;
                    height: 32px !important;
                    min-width: 72px !important;
                    padding: 0 12px !important;
                    font-size: 0.8rem !important;
                }

                .sidebar-model-compact .v-input {
                    margin-top: 0 !important;
                    padding-top: 0 !important;
                }
                .sidebar-model-compact .v-input__details {
                    min-height: 0 !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }
                .sidebar-model-compact .v-messages {
                    display: none !important;
                }
                """
            )

            with solara.Column(classes=["simulation-main"]):
                with solara.Columns([3, 2]):
                    with solara.Column():
                        with solara.Card("Agent map"):
                            SpaceRendererComponent(model.value, renderer.value)

                    with solara.Column():
                        with solara.Card("Live statistics", style={"height": "100%"}):
                            live_stats_content(model.value)

                with solara.Card("Visualization", classes=["visualization-card"]):
                    with solara.Columns([1, 1]):
                        with solara.Column():
                            self._plot_callable(seir_plot)(model.value)
                        with solara.Column():
                            self._plot_callable(new_exposures_plot)(model.value)

                    with solara.Columns([1, 1]):
                        with solara.Column():
                            self._plot_callable(cumulative_plot)(model.value)
                        with solara.Column():
                            self._plot_callable(infectious_by_type_plot)(model.value)

        return page
