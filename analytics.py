from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

from model import EpidemicModel


def run_simulation_collect(config: dict, steps: int, run_id: int) -> tuple[list[dict], dict]:
    model = EpidemicModel(
        population=config["population"],
        city_map_path=config["cityMapPath"],
        time_of_day=config.get("startTime", 10),
        timestep=config.get("timestep", 0.5),
        verbose=config.get("verbose", False),
    )

    for _ in range(steps):
        model.step()

    timeseries = []
    for row in model.metrics_history:
        timeseries.append(
            {
                "run_id": run_id,
                "step": row["step"],
                "time_of_day": row["time_of_day"],
                "susceptible": row["susceptible"],
                "exposed": row["exposed"],
                "infectious": row["infectious"],
                "recovered": row["recovered"],
                "population": row["population"],
                "infected_ratio": round(row["infected_ratio"], 6),
            }
        )

    summary = model.get_summary_metrics()
    summary["run_id"] = run_id
    summary["population"] = len(model.agents)
    summary["steps_requested"] = steps
    return timeseries, summary


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _plot_state_means(aggregated: list[dict], output_dir: Path) -> None:
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

    plt.title("Health states over time (mean +/- std)")
    plt.xlabel("Step")
    plt.ylabel("Agents")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "health_states_mean_std.png", dpi=160)
    plt.close()


def _plot_infectious_per_run(all_timeseries: list[dict], output_dir: Path) -> None:
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

    plt.title("Infectious curve per run")
    plt.xlabel("Step")
    plt.ylabel("Infectious agents")
    plt.grid(alpha=0.3)
    if len(runs) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "infectious_per_run.png", dpi=160)
    plt.close()


def _plot_peak_histogram(run_summaries: list[dict], output_dir: Path) -> None:
    peaks = [row["peak_infectious"] for row in run_summaries]

    plt.figure(figsize=(10, 6))
    bins = min(20, max(5, len(peaks)))
    plt.hist(peaks, bins=bins, color="tomato", alpha=0.85, edgecolor="black")
    plt.title("Distribution of peak infectious counts")
    plt.xlabel("Peak infectious")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "peak_infectious_histogram.png", dpi=160)
    plt.close()


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

    _write_csv(
        output_path / "simulation_timeseries.csv",
        [
            "run_id",
            "step",
            "time_of_day",
            "susceptible",
            "exposed",
            "infectious",
            "recovered",
            "population",
            "infected_ratio",
        ],
        all_timeseries,
    )

    _write_csv(
        output_path / "simulation_runs_summary.csv",
        [
            "run_id",
            "steps_requested",
            "steps_executed",
            "population",
            "peak_infectious",
            "peak_infectious_step",
            "final_susceptible",
            "final_exposed",
            "final_infectious",
            "final_recovered",
            "total_ever_infected",
        ],
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

    _plot_state_means(aggregated, output_path)
    _plot_infectious_per_run(all_timeseries, output_path)
    _plot_peak_histogram(run_summaries, output_path)

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
    }

    with (output_path / "simulation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return {
        "output_dir": str(output_path.resolve()),
        "timeseries_csv": str((output_path / "simulation_timeseries.csv").resolve()),
        "summary_csv": str((output_path / "simulation_runs_summary.csv").resolve()),
        "aggregate_csv": str((output_path / "simulation_aggregated_by_step.csv").resolve()),
        "metadata_json": str((output_path / "simulation_metadata.json").resolve()),
        "plots": [
            str((output_path / "health_states_mean_std.png").resolve()),
            str((output_path / "infectious_per_run.png").resolve()),
            str((output_path / "peak_infectious_histogram.png").resolve()),
        ],
    }
