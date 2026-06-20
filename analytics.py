from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

from agent_types import AgentType
from model import EpidemicModel
from time_utils import format_time_ampm

UNKNOWN_MAP_NAME = "Unknown map"

# Matplotlib colour per agent type. Mirrors AGENT_TYPE_COLORS in
# pygame_visualizer so the exported PNGs look identical to the live plots.
_AGENT_TYPE_PLOT_COLORS = {
    AgentType.STUDENT: "tab:blue",
    AgentType.WORKER: "tab:orange",
    AgentType.SENIOR: "tab:red",
    AgentType.HEALTHCARE: "tab:green",
    AgentType.CHILDREN: "tab:purple",
}


def model_metrics_to_timeseries(model: EpidemicModel, run_id: int = 1) -> list[dict]:
    """Convert a model's recorded metrics history into normalized timeseries rows.

    Rows include per-agent-type S/E/I/R counts plus useful derived per-step
    measures (``new_exposures``, ``cumulative_*``) so post-hoc analyses can
    work straight from CSV without re-running anything.
    """
    map_name = getattr(model, "map_name", UNKNOWN_MAP_NAME)
    timeseries = []
    for row in model.metrics_history:
        record = {
            "run_id": run_id,
            "step": row["step"],
            "time_of_day": row["time_of_day"],
            "susceptible": row["susceptible"],
            "exposed": row["exposed"],
            "infectious": row["infectious"],
            "recovered": row["recovered"],
            "new_exposures": row.get("new_exposures", 0),
            "cumulative_infected": row.get("cumulative_infected"),
            "cumulative_ratio": round(row.get("cumulative_ratio", 0.0), 6),
            "population": row["population"],
            "infected_ratio": round(row["infected_ratio"], 6),
            "map_name": map_name,
        }
        # Pass through every per-agent-type bucket
        # (susceptible_student, exposed_worker, ...).
        for key, value in row.items():
            if "_" in key and key.split("_", 1)[0] in (
                "susceptible", "exposed", "infectious", "recovered",
            ):
                record[key] = value
        timeseries.append(record)
    return timeseries


def run_simulation_collect(config: dict, steps: int, run_id: int) -> tuple[list[dict], dict]:
    # All non-trivial parameters are pulled from config so the same call
    # path is reused by the parameter-sweep experiments.
    model_kwargs = dict(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        city_map_preset=config.get("cityMapPreset"),
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
        verbose=config.get("verbose", False),
    )
    if "avg_household_size" in config:
        model_kwargs["avg_household_size"] = int(config["avg_household_size"])
    if "avg_family_size" in config:
        model_kwargs["avg_family_size"] = int(config["avg_family_size"])
    if "avg_friend_group_size" in config:
        model_kwargs["avg_friend_group_size"] = int(config["avg_friend_group_size"])
    if config.get("max_transmission_distance") is not None:
        model_kwargs["max_transmission_distance"] = int(config["max_transmission_distance"])
    if config.get("agent_overrides"):
        model_kwargs["agent_overrides"] = config["agent_overrides"]

    model = EpidemicModel(**model_kwargs)

    for _ in range(steps):
        model.step()

    timeseries = model_metrics_to_timeseries(model, run_id=run_id)
    summary = model.get_summary_metrics()
    summary["run_id"] = run_id
    summary["population"] = len(model.agents)
    summary["steps_requested"] = steps
    summary["map_name"] = getattr(model, "map_name", UNKNOWN_MAP_NAME)
    # Record the parameters that produced this run so the CSV is self-contained
    # for post-hoc statistical analysis ("which knob did what").
    summary["param_avg_household_size"] = getattr(model, "avg_household_size", None)
    summary["param_avg_family_size"] = getattr(model, "avg_family_size", None)
    summary["param_avg_friend_group_size"] = getattr(model, "avg_friend_group_size", None)
    summary["param_timestep"] = getattr(model, "timestep", None)
    summary["param_start_time"] = config.get("startTime")
    summary["param_city_map_preset"] = getattr(model, "city_map_preset", None)
    summary["param_max_transmission_distance"] = getattr(
        model, "max_transmission_distance", None
    )
    # Flatten agent overrides so the per-run summary is self-describing.
    overrides = config.get("agent_overrides") or {}
    for agent_key, fields in overrides.items():
        if not isinstance(fields, dict):
            continue
        for field_name, value in fields.items():
            summary[f"param_{agent_key}_{field_name}"] = value
    return timeseries, summary


def _live_export_payload(
    model: EpidemicModel,
    config: dict | None,
) -> tuple[list[dict], list[dict], dict]:
    """Build the (timeseries, summaries, effective_config) tuple shared
    between :func:`export_live_data` and :func:`export_live_plots`. Raises
    ValueError if the model hasn't been stepped yet."""
    if not model.metrics_history:
        raise ValueError(
            "Model has no recorded metrics yet - run at least one step "
            "before exporting."
        )
    timeseries = model_metrics_to_timeseries(model, run_id=1)
    summary = model.get_summary_metrics()
    summary["run_id"] = 1
    summary["population"] = len(model.agents)
    summary["steps_requested"] = model.current_step
    summary["map_name"] = getattr(model, "map_name", UNKNOWN_MAP_NAME)
    effective_config = {
        "cityMapPath": getattr(model, "city_map_path", None),
        "cityMapPreset": getattr(model, "city_map_preset", None),
        "mapName": getattr(model, "map_name", UNKNOWN_MAP_NAME),
        **(config or {}),
    }
    return timeseries, [summary], effective_config


def export_live_data(
    model: EpidemicModel,
    output_dir: str,
    config: dict | None = None,
) -> dict:
    """Export only the data artifacts (CSV/JSON) for the current model state.

    No plots are written - use :func:`export_live_plots` for that. The data
    files are the same shape as in the batch ``export_artifacts`` output.
    """
    timeseries, summaries, effective_config = _live_export_payload(model, config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return _write_data_files(
        output_path=output_path,
        config=effective_config,
        steps=model.current_step,
        all_timeseries=timeseries,
        run_summaries=summaries,
    )


def export_live_plots(
    model: EpidemicModel,
    output_dir: str,
    config: dict | None = None,
) -> dict:
    """Export the four live charts shown in the GUI as PNG files.

    Saves exactly the same four panels that the Pygame visualizer renders
    in real time (SEIR, new exposures, cumulative infected, infectious by
    agent type) - no batch-mean / per-run / histogram plots.
    """
    _, _, effective_config = _live_export_payload(model, config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    map_name = effective_config.get("mapName") or UNKNOWN_MAP_NAME
    data = model.datacollector.model_vars
    steps = list(range(len(next(iter(data.values()), []))))
    if not steps:
        raise ValueError("Model has no collected steps yet.")
    # Per-step clock used to annotate x-axis ticks ("3\n10:30am").
    step_times = [row["time_of_day"] for row in model.metrics_history]

    plot_paths: list[str] = []
    for cfg in _live_plot_configs():
        target = output_path / cfg["filename"]
        _plot_live_chart(
            steps=steps,
            step_times=step_times,
            data=data,
            series=cfg["series"],
            title=cfg["title"],
            ylabel=cfg["ylabel"],
            map_name=map_name,
            output_path=target,
        )
        plot_paths.append(str(target.resolve()))

    return {
        "output_dir": str(output_path.resolve()),
        "plots": plot_paths,
    }


def _live_plot_configs() -> list[dict]:
    """The four live-plot specs, resolved lazily so AgentType lookups happen
    at call time (not import time)."""
    return [
        {
            "filename": "live_health_states.png",
            "title": "Health states (S/E/I/R)",
            "ylabel": "Agents",
            "series": [
                ("Susceptible", "#4caf50"),
                ("Exposed", "#ffb300"),
                ("Infectious", "#e53935"),
                ("Recovered", "#9e9e9e"),
            ],
        },
        {
            "filename": "live_new_exposures.png",
            "title": "New exposures per step (S->E)",
            "ylabel": "Agents",
            "series": [("New_Exposures", "#7e57c2")],
        },
        {
            "filename": "live_cumulative_infected.png",
            "title": "Cumulative infected",
            "ylabel": "Agents",
            "series": [("Cumulative_Infected", "#e53935")],
        },
        {
            "filename": "live_infectious_by_type.png",
            "title": "Infectious by agent type",
            "ylabel": "Infectious agents",
            "series": [
                (f"Infectious_{t.value}", _AGENT_TYPE_PLOT_COLORS[t])
                for t in AgentType
            ],
        },
    ]


def _plot_live_chart(
    *,
    steps: list[int],
    step_times: list[float],
    data: dict,
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    map_name: str,
    output_path: Path,
) -> None:
    """Render a single live-style chart to ``output_path``. Visually matches
    the in-window plots (dark theme), at print resolution. X-axis ticks are
    annotated with the corresponding time-of-day (e.g. ``"3\\n10:30am"``)."""
    fig = plt.figure(figsize=(10, 6), dpi=160, facecolor="#2a2a35")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1e1e26")
    for key, color in series:
        if key not in data:
            continue
        label = key.replace("Infectious_", "")
        ax.plot(steps, data[key], label=label, color=color, linewidth=1.8)
    ax.set_title(f"{title} - {map_name}", color="#e6e6eb", fontsize=12)
    ax.set_xlabel("Step (time of day)", color="#9999a3", fontsize=10)
    ax.set_ylabel(ylabel, color="#9999a3", fontsize=10)
    ax.tick_params(colors="#9999a3", labelsize=9)

    def _fmt(tick: float, _pos: int) -> str:
        s = int(round(tick))
        if 0 <= s < len(step_times):
            return f"{s}\n{format_time_ampm(step_times[s])}"
        return f"{s}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    for spine in ax.spines.values():
        spine.set_color("#55555f")
    ax.grid(alpha=0.2, color="#888892")
    if len(series) > 1:
        legend = ax.legend(
            loc="best", fontsize=9, framealpha=0.7,
            facecolor="#2a2a35", edgecolor="#55555f", labelcolor="#e6e6eb",
        )
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#e6e6eb")
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _collect_fieldnames(
    rows: list[dict], required_first: list[str],
) -> list[str]:
    """Build a CSV header that puts ``required_first`` columns up front and
    appends any extra keys discovered in the rows (insertion-ordered)."""
    seen: dict[str, None] = {}
    for col in required_first:
        seen.setdefault(col, None)
    for row in rows:
        for key in row.keys():
            seen.setdefault(key, None)
    return list(seen.keys())


def _aggregate_by_step(all_timeseries: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in all_timeseries:
        grouped[row["step"]].append(row)

    aggregated = []
    for step in sorted(grouped):
        rows = grouped[step]
        aggregated.append(
            {
                "step": step,
                "susceptible_mean": mean([r["susceptible"] for r in rows]),
                "susceptible_std": pstdev([r["susceptible"] for r in rows]) if len(rows) > 1 else 0.0,
                "exposed_mean": mean([r["exposed"] for r in rows]),
                "exposed_std": pstdev([r["exposed"] for r in rows]) if len(rows) > 1 else 0.0,
                "infectious_mean": mean([r["infectious"] for r in rows]),
                "infectious_std": pstdev([r["infectious"] for r in rows]) if len(rows) > 1 else 0.0,
                "recovered_mean": mean([r["recovered"] for r in rows]),
                "recovered_std": pstdev([r["recovered"] for r in rows]) if len(rows) > 1 else 0.0,
            }
        )
    return aggregated


def _plot_state_means(aggregated: list[dict], output_dir: Path, map_name: str) -> None:
    steps = [row["step"] for row in aggregated]

    plt.figure(figsize=(12, 7))
    for state, color in [
        ("susceptible", "green"),
        ("exposed", "orange"),
        ("infectious", "red"),
        ("recovered", "gray"),
    ]:
        means = [row[f"{state}_mean"] for row in aggregated]
        stds = [row[f"{state}_std"] for row in aggregated]
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]

        plt.plot(steps, means, label=f"{state.title()} mean", color=color, linewidth=2)
        plt.fill_between(steps, lower, upper, color=color, alpha=0.15, linewidth=0)

    plt.title(f"Health states over time (mean +/- std) - {map_name}")
    plt.xlabel("Step")
    plt.ylabel("Agents")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "health_states_mean_std.png", dpi=160)
    plt.close()


def _plot_infectious_per_run(all_timeseries: list[dict], output_dir: Path, map_name: str) -> None:
    runs: dict[int, list[dict]] = defaultdict(list)
    for row in all_timeseries:
        runs[row["run_id"]].append(row)

    plt.figure(figsize=(12, 7))
    for run_id in sorted(runs):
        rows = sorted(runs[run_id], key=lambda item: item["step"])
        plt.plot(
            [row["step"] for row in rows],
            [row["infectious"] for row in rows],
            linewidth=1.5,
            alpha=0.9,
            label=f"Run {run_id}",
        )

    plt.title(f"Infectious curve per run - {map_name}")
    plt.xlabel("Step")
    plt.ylabel("Infectious agents")
    plt.grid(alpha=0.3)
    if len(runs) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "infectious_per_run.png", dpi=160)
    plt.close()


def _plot_peak_histogram(run_summaries: list[dict], output_dir: Path, map_name: str) -> None:
    peaks = [row["peak_infectious"] for row in run_summaries]

    plt.figure(figsize=(10, 6))
    bins = min(20, max(5, len(peaks)))
    plt.hist(peaks, bins=bins, color="tomato", alpha=0.85, edgecolor="black")
    plt.title(f"Distribution of peak infectious counts - {map_name}")
    plt.xlabel("Peak infectious")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "peak_infectious_histogram.png", dpi=160)
    plt.close()


def _write_data_files(
    *,
    output_path: Path,
    config: dict,
    steps: int,
    all_timeseries: list[dict],
    run_summaries: list[dict],
) -> dict:
    """Write the four data artifacts (3 CSVs + 1 JSON) and return their
    resolved paths. Shared by batch ``export_artifacts`` and the GUI's
    ``export_live_data``."""
    aggregated = _aggregate_by_step(all_timeseries)
    map_name = config.get("mapName") if isinstance(config, dict) else None
    if not map_name and run_summaries:
        map_name = run_summaries[0].get("map_name")
    map_name = map_name or UNKNOWN_MAP_NAME

    timeseries_fieldnames = _collect_fieldnames(
        all_timeseries,
        required_first=[
            "run_id", "step", "time_of_day",
            "susceptible", "exposed", "infectious", "recovered",
            "new_exposures", "cumulative_infected", "cumulative_ratio",
            "population", "infected_ratio", "map_name",
        ],
    )
    _write_csv(
        output_path / "simulation_timeseries.csv",
        timeseries_fieldnames,
        all_timeseries,
    )

    summary_fieldnames = _collect_fieldnames(
        run_summaries,
        required_first=[
            "run_id", "map_name", "steps_requested", "steps_executed",
            "population", "peak_infectious", "peak_infectious_step",
            "peak_infectious_time", "max_new_exposures_per_step",
            "infectious_steps", "attack_rate", "time_to_half_attack_step",
            "final_susceptible", "final_exposed", "final_infectious",
            "final_recovered", "total_ever_infected",
        ],
    )
    _write_csv(
        output_path / "simulation_runs_summary.csv",
        summary_fieldnames,
        run_summaries,
    )

    _write_csv(
        output_path / "simulation_aggregated_by_step.csv",
        [
            "step",
            "susceptible_mean",
            "susceptible_std",
            "exposed_mean",
            "exposed_std",
            "infectious_mean",
            "infectious_std",
            "recovered_mean",
            "recovered_std",
        ],
        aggregated,
    )

    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "runs": len(run_summaries),
        "config": config,
        "aggregate_summary": {
            "peak_infectious_mean": mean([row["peak_infectious"] for row in run_summaries]),
            "peak_infectious_std": pstdev([row["peak_infectious"] for row in run_summaries]) if len(run_summaries) > 1 else 0.0,
            "total_ever_infected_mean": mean([row["total_ever_infected"] for row in run_summaries]),
            "final_recovered_mean": mean([row["final_recovered"] for row in run_summaries]),
        },
        "map_name": map_name,
    }
    with (output_path / "simulation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return {
        "output_dir": str(output_path.resolve()),
        "timeseries_csv": str((output_path / "simulation_timeseries.csv").resolve()),
        "summary_csv": str((output_path / "simulation_runs_summary.csv").resolve()),
        "aggregate_csv": str((output_path / "simulation_aggregated_by_step.csv").resolve()),
        "metadata_json": str((output_path / "simulation_metadata.json").resolve()),
    }


def export_artifacts(
    output_dir: str,
    config: dict,
    steps: int,
    all_timeseries: list[dict],
    run_summaries: list[dict],
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregated = _aggregate_by_step(all_timeseries)
    map_name = config.get("mapName") if isinstance(config, dict) else None
    if not map_name and run_summaries:
        map_name = run_summaries[0].get("map_name")
    map_name = map_name or UNKNOWN_MAP_NAME

    data_paths = _write_data_files(
        output_path=output_path,
        config=config,
        steps=steps,
        all_timeseries=all_timeseries,
        run_summaries=run_summaries,
    )

    # Batch-style summary plots (mean +/- std across runs, per-run curves,
    # peak histogram). The live GUI uses different charts via
    # ``export_live_plots``.
    _plot_state_means(aggregated, output_path, map_name)
    _plot_infectious_per_run(all_timeseries, output_path, map_name)
    _plot_peak_histogram(run_summaries, output_path, map_name)

    return {
        **data_paths,
        "plots": [
            str((output_path / "health_states_mean_std.png").resolve()),
            str((output_path / "infectious_per_run.png").resolve()),
            str((output_path / "peak_infectious_histogram.png").resolve()),
        ],
    }
