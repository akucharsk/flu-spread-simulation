import argparse
import json

from analytics import export_artifacts, run_simulation_collect


def main():
    parser = argparse.ArgumentParser(description="Run flu spread simulation headless")
    parser.add_argument("--steps", type=int, help="Number of steps to simulate")
    parser.add_argument("--runs", type=int, help="How many independent runs to execute")
    parser.add_argument("--output-dir", help="Directory where CSV/JSON/plots will be saved")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    # Per-experiment overrides - lets the parameter-sweep script reuse one
    # config.json and override only the knob it's varying.
    parser.add_argument("--population", type=int, help="Override agent population")
    parser.add_argument("--city-map-path", help="Override path to the city .txt map")
    parser.add_argument("--city-map-preset", help="Override built-in map preset key")
    parser.add_argument("--avg-household-size", type=int,
                        help="Override average household size")
    parser.add_argument("--avg-family-size", type=int,
                        help="Override average family group size")
    parser.add_argument("--avg-friend-group-size", type=int,
                        help="Override average friend-group size")
    parser.add_argument("--start-time", type=float, help="Override start time of day (h)")
    parser.add_argument("--timestep", type=float, help="Override step length (h)")
    parser.add_argument("--max-transmission-distance", type=int,
                        help="Override transmission radius (in cells).")
    parser.add_argument(
        "--mobility", action="append", default=[],
        metavar="AGENT_TYPE=VALUE",
        help="Override mobility for an agent type, e.g. --mobility worker=0.4."
             " Repeat for several types.",
    )
    parser.add_argument(
        "--infection-rate", action="append", default=[],
        metavar="AGENT_TYPE=VALUE",
        help="Override infection_rate for an agent type, e.g. "
             "--infection-rate senior=0.5. Repeat for several types.",
    )
    parser.add_argument("--label", help="Free-form experiment label stored in metadata")
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    # CLI overrides win over the JSON config.
    if args.population is not None:
        config["population"] = args.population
    if args.city_map_path is not None:
        config["cityMapPath"] = args.city_map_path
    if args.city_map_preset is not None:
        config["cityMapPreset"] = args.city_map_preset
    if args.avg_household_size is not None:
        config["avg_household_size"] = args.avg_household_size
    if args.avg_family_size is not None:
        config["avg_family_size"] = args.avg_family_size
    if args.avg_friend_group_size is not None:
        config["avg_friend_group_size"] = args.avg_friend_group_size
    if args.start_time is not None:
        config["startTime"] = args.start_time
    if args.timestep is not None:
        config["timestep"] = args.timestep
    if args.max_transmission_distance is not None:
        config["max_transmission_distance"] = args.max_transmission_distance

    # Collect per-type overrides (mobility / infection_rate) from CLI into
    # ``config['agent_overrides']`` so analytics.py picks them up.
    agent_overrides = dict(config.get("agent_overrides") or {})

    def _merge(field_name: str, raw_pairs: list[str]) -> None:
        for raw in raw_pairs:
            if "=" not in raw:
                raise ValueError(f"Expected AGENT_TYPE=VALUE, got: {raw!r}")
            agent_key, value_str = raw.split("=", 1)
            agent_key = agent_key.strip().lower()
            slot = dict(agent_overrides.get(agent_key, {}))
            slot[field_name] = float(value_str)
            agent_overrides[agent_key] = slot

    _merge("mobility", args.mobility)
    _merge("infection_rate", args.infection_rate)
    if agent_overrides:
        config["agent_overrides"] = agent_overrides
    if args.label is not None:
        config["experimentLabel"] = args.label

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
