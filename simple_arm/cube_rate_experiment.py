"""Paired command-rate-limit trials with a fixed learned model and score."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .contact_learning import new_plot
from .cube_mpc import DT, ModelAdapter
from .cube_mpc_comparison import summarize, workspace_metrics
from .cube_mpc_experiment import (CASES, SEEDS, file_hash, load_cases, metrics,
                                  run_episode, source_hashes, write_json)
from .cube_workspace import Workspace


def motion_metrics(episode):
    commands, initial = episode["commands"], episode["states"][0, :2]
    changes = np.diff(np.vstack([initial, commands]), axis=0)
    rate = changes / DT
    acceleration = np.diff(np.vstack([np.zeros(2), rate]), axis=0) / DT
    return {"maximum_command_rate_rad_s": float(np.abs(rate).max()),
            "rms_command_rate_rad_s": float(np.sqrt(np.mean(rate**2))),
            "command_total_variation_rad": np.abs(changes).sum(axis=0).tolist(),
            "maximum_command_acceleration_rad_s2": float(np.abs(acceleration).max()),
            "maximum_actual_joint_speed_rad_s": float(np.abs(episode["fine_states"][:, 2:4]).max())}


def plots(output, records):
    figure, _ = new_plot()
    figure.set_size_inches(12, 8)
    figure.clear()
    for panel, case in zip(figure.subplots(2, 2).flat, CASES):
        for r in records:
            if r["case"] != case:
                continue
            actual, target = r["episode"]["states"], r["target"]
            panel.plot(np.arange(len(actual)) * DT, np.linalg.norm(actual[:, 4:6] - target, axis=1),
                       color="tab:blue" if r["mode"] == "limited" else "tab:orange",
                       linestyle={42: "-", 43: "--", 44: ":"}[r["seed"]], label=f"{r['mode']} {r['seed']}")
        panel.axhline(.02, color="gray", linestyle=":")
        panel.set(title=f"Episode {case}", xlabel="Simulation time (s)", ylabel="Actual target distance (m)")
        panel.legend(fontsize=7)
    figure.savefig(output / "distance_comparison.png")
    figure, _ = new_plot()
    figure.set_size_inches(12, 8)
    figure.clear()
    panels = figure.subplots(2, 2)
    for r in records:
        if r["case"] != 23 or r["seed"] != 42:
            continue
        e = r["episode"]
        commands = np.vstack([e["states"][0, :2], e["commands"]])
        for joint in range(2):
            panels[0, joint].plot(np.arange(len(commands)) * DT, commands[:, joint], label=r["mode"])
            panels[0, joint].set(xlabel="Simulation time (s)", ylabel="Joint target (rad)", title=["Shoulder", "Elbow"][joint])
            panels[1, joint].plot(np.arange(len(e["fine_states"])) * .002, e["fine_states"][:, 9 + joint], label=r["mode"])
            panels[1, joint].set(xlabel="Simulation time (s)", ylabel=f"Actual cube {'XY'[joint]} velocity (m/s)")
    for panel in panels.flat:
        panel.legend()
    figure.suptitle("Episode 23, seed 42: commands and physical cube velocity")
    figure.savefig(output / "commands_cube_velocity.png")


def compare(checkpoint, dataset, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    rate_limit = config.command_rate_limit or .5
    before = source_hashes(checkpoint, dataset)
    predict, workspace = ModelAdapter(checkpoint), Workspace()
    provenance = predict.checkpoint["metadata"]
    if any(p["seed"] == 292 for p in [provenance, *provenance.get("parents", [])]):
        raise ValueError("Benchmark dataset overlaps training")
    cases = load_cases(dataset)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"config": asdict(config), "checkpoint": str(checkpoint), "dataset": str(dataset),
               "rate_limit_rad_s": rate_limit, "maximum_command_change_per_step_rad": rate_limit * DT,
               "cases": list(CASES), "seeds": list(SEEDS), "duration_s": 5, "source_hashes_before": before,
               "intervention": "Sequentially project each candidate before neural rollout; same model, score and CEM settings in both modes"})
    records = []
    for case, (initial, target) in cases.items():
        for seed in SEEDS:
            for mode, rate in (("unlimited", 0.), ("limited", rate_limit)):
                key = f"{case}_{seed}_{mode}"
                e = run_episode(initial, target, predict, replace(config, seed=seed, command_rate_limit=rate))
                report = dict(metrics(e, target), **workspace_metrics(e, workspace), **motion_metrics(e))
                report.update(case=case, seed=seed, mode=mode, command_rate_limit_rad_s=rate)
                if rate and report["maximum_command_rate_rad_s"] > rate + 1e-5:
                    raise RuntimeError("Executed commands exceeded configured rate")
                np.savez_compressed(output / f"{key}.npz", **{k: e[k] for k in (
                    "states", "fine_states", "commands", "predictions", "timings", "joint_limits")}, target=target)
                write_json(output / f"{key}.json", dict(report, costs=e["costs"], diagnostics=e["diagnostics"]))
                records.append({"case": case, "seed": seed, "mode": mode, "target": target, "episode": e, "report": report})
                print(f"{key}: success={report['success']} distance={report['final_distance_m']:.4f}m "
                      f"escaped={report['escaped']} cube_vmax={report['maximum_cube_speed_m_s']:.3f}m/s "
                      f"command_rate={report['maximum_command_rate_rad_s']:.3f}rad/s", flush=True)
    summary = {"runs": {f"{r['case']}_{r['seed']}_{r['mode']}": r["report"] for r in records}, "aggregate": {}}
    for mode in ("unlimited", "limited"):
        subset = [r for r in records if r["mode"] == mode]
        summary["aggregate"][mode] = dict(summarize(subset),
            maximum_command_rate_rad_s=max(r["report"]["maximum_command_rate_rad_s"] for r in subset),
            mean_rms_command_rate_rad_s=float(np.mean([r["report"]["rms_command_rate_rad_s"] for r in subset])),
            maximum_actual_joint_speed_rad_s=max(r["report"]["maximum_actual_joint_speed_rad_s"] for r in subset))
    plots(output, records)
    after = {p: file_hash(p) for p in before}
    summary.update(source_hashes_after=after, sources_unchanged=before == after)
    write_json(output / "summary.json", summary)
    if before != after:
        raise RuntimeError("Existing checkpoint or dataset changed")
    return summary
