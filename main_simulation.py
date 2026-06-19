import argparse
import json

from analytics import export_artifacts, run_simulation_collect


def main():
    parser = argparse.ArgumentParser(description="Run flu spread simulation headless")
    parser.add_argument("--steps", type=int, help="Number of steps to simulate")
    parser.add_argument("--runs", type=int, help="How many independent runs to execute")
    parser.add_argument("--output-dir", help="Directory where CSV/JSON/plots will be saved")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    steps = args.steps if args.steps is not None else config.get("steps")
    runs = args.runs if args.runs is not None else config.get("runs", 1)
    output_dir = args.output_dir if args.output_dir is not None else config.get("outputDir", "output")
    if steps is None:
        raise ValueError("Number of steps must be provided via --steps or config['steps']")
    if steps <= 0:
        raise ValueError("Steps must be > 0")
    if runs <= 0:
        raise ValueError("Runs must be > 0")

    print(f"Running {runs} simulation run(s) for {steps} steps each...")
    all_timeseries = []
    run_summaries = []
    for run_id in range(1, runs + 1):
        timeseries, summary = run_simulation_collect(config=config, steps=steps, run_id=run_id)
        all_timeseries.extend(timeseries)
        run_summaries.append(summary)

        print(
            f"  Run {run_id}/{runs}: "
            f"peak_I={summary['peak_infectious']} "
            f"at step {summary['peak_infectious_step']}, "
            f"final_R={summary['final_recovered']}"
        )

    artifacts = export_artifacts(
        output_dir=output_dir,
        config=config,
        steps=steps,
        all_timeseries=all_timeseries,
        run_summaries=run_summaries,
    )

    print("Simulation completed successfully")
    print(f"Artifacts saved to: {artifacts['output_dir']}")
    print(f"- Timeseries CSV: {artifacts['timeseries_csv']}")
    print(f"- Runs summary CSV: {artifacts['summary_csv']}")
    print(f"- Aggregate CSV: {artifacts['aggregate_csv']}")
    print(f"- Metadata JSON: {artifacts['metadata_json']}")
    print("- Plots:")
    for plot_path in artifacts["plots"]:
        print(f"  - {plot_path}")


if __name__ == "__main__":
    main()
