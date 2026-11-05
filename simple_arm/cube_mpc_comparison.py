"""Paired actual-physics comparison of original and workspace-aware scores."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .contact_learning import new_plot
from .cube_mpc import DT, ModelAdapter
from .cube_mpc_experiment import (CASES, SEEDS, file_hash, load_cases, metrics, run_episode,
                                  source_hashes, timing_metrics, write_json)
from .cube_workspace import Workspace


def workspace_metrics(episode, workspace):
    fine = episode["fine_states"]
    with torch.inference_mode():
        excess = workspace.excess(torch.tensor(fine[:, 4:6], dtype=torch.float64)).numpy()
    outside = excess > 1e-6  # numerical tolerance, not an extra workspace margin
    transitions = np.diff(np.r_[False, outside].astype(int))
    speed = np.linalg.norm(fine[:, 9:11], axis=1)
    return {"escaped": bool(outside.any()), "final_outside": bool(outside[-1]),
            "escape_events": int((transitions == 1).sum()),
            "outside_physics_steps": int(outside.sum()), "outside_fraction": float(outside.mean()),
            "maximum_workspace_excess_m": float(excess.max()),
            "maximum_cube_speed_m_s": float(speed.max()),
            "maximum_cube_3d_speed_m_s": float(np.linalg.norm(fine[:, 9:12], axis=1).max()),
            "beyond_any_contact_outer_bound": bool((np.linalg.norm(fine[:, 4:6] - workspace.origin.numpy(), axis=1) > workspace.outer_contact_bound + 1e-6).any())}


def summarize(records):
    reports = [r["report"] for r in records]
    return {"runs": len(reports), "success_rate": float(np.mean([r["success"] for r in reports])),
            "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in reports])),
            "escape_run_rate": float(np.mean([r["escaped"] for r in reports])),
            "escape_events": sum(r["escape_events"] for r in reports),
            "outside_fraction": float(np.mean([r["outside_fraction"] for r in reports])),
            "mean_run_maximum_cube_speed_m_s": float(np.mean([r["maximum_cube_speed_m_s"] for r in reports])),
            "maximum_cube_speed_m_s": max(r["maximum_cube_speed_m_s"] for r in reports),
            "timing": timing_metrics(np.concatenate([r["episode"]["timings"] for r in records]))}


def comparison_plots(output, records, workspace):
    figure, _ = new_plot()
    figure.clear()
    axes = figure.subplots(2, 2).flat
    reach = sum(workspace.lengths) + workspace.contact_radius
    x = np.linspace(-reach, reach, 180)
    y = np.linspace(-reach, reach, 180)
    xx, yy = np.meshgrid(x, y)
    with torch.inference_mode():
        excess = workspace.excess(torch.tensor(np.stack([xx, yy], -1), dtype=torch.float32)).numpy()
    for panel, case in zip(axes, CASES):
        panel.contour(xx, yy, excess, levels=[1e-6], colors="gray", linewidths=.8)
        for r in records:
            if r["case"] != case:
                continue
            actual = r["episode"]["states"]
            panel.plot(actual[:, 4], actual[:, 5], color="tab:orange" if r["scoring"] == "original" else "tab:blue",
                       linestyle={42: "-", 43: "--", 44: ":"}[r["seed"]], label=f"{r['scoring']} {r['seed']}")
            panel.scatter(*actual[-1, 4:6], marker="x", s=16)
            target = r["target"]
        panel.scatter(*target, color="green", marker="*", s=70)
        panel.set(title=f"Episode {case}: actual paths", xlabel="X (m)", ylabel="Y (m)", aspect="equal")
        panel.legend(fontsize=5)
    figure.savefig(output / "actual_workspace_paths.png")
    figure, _ = new_plot()
    figure.clear()
    panels = figure.subplots(2, 2).flat
    for panel, case in zip(panels, CASES):
        subset = [r for r in records if r["case"] == case]
        names = list(subset[-1]["episode"]["diagnostics"][0]["score_components"])
        bottom = np.zeros(len(subset))
        for name in names:
            values = np.array([np.mean([d["score_components"].get(name, 0.) for d in r["episode"]["diagnostics"]]) for r in subset])
            panel.bar(np.arange(len(subset)), values, bottom=bottom, label=name)
            bottom += values
        panel.set_xticks(np.arange(len(subset)), [f"{r['scoring']}\n{r['seed']}" for r in subset], fontsize=6)
        panel.set(title=f"Episode {case}: selected neural score", ylabel="Mean weighted contribution")
        panel.legend(fontsize=4)
    figure.savefig(output / "score_breakdown.png")


def compare(checkpoint, dataset, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    before = source_hashes(checkpoint, dataset)
    prior = {str(p): file_hash(p) for p in Path("runs/cube_cem_initial").glob("*") if p.is_file()}
    predict, workspace = ModelAdapter(checkpoint), Workspace()
    cases = load_cases(dataset)
    provenance = predict.checkpoint["metadata"]
    if any(p["seed"] == 292 for p in [provenance, *provenance.get("parents", [])]):
        raise ValueError("Benchmark dataset is not independent of checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"config": asdict(config), "checkpoint": str(checkpoint), "dataset": str(dataset),
               "cases": list(CASES), "seeds": list(SEEDS), "duration_s": 5., "dt_s": DT,
               "workspace": {"lengths_m": workspace.lengths, "joint_limits_rad": workspace.limits.tolist(),
                             "contact_dilation_m": workspace.contact_radius, "any_contact_outer_bound_m": workspace.outer_contact_bound,
                             "definition": "Joint-limited tip workspace dilated by pusher radius plus cube inscribed radius; kinematic proxy, no directional contact guarantee"},
               "source_hashes_before": before, "prior_experiment_hashes_before": prior})
    records = []
    for case, (initial, target) in cases.items():
        for seed in SEEDS:
            for scoring in ("original", "workspace"):
                key = f"{case}_{seed}_{scoring}"
                e = run_episode(initial, target, predict, replace(config, seed=seed, scoring=scoring))
                report = dict(metrics(e, target), **workspace_metrics(e, workspace))
                report.update(case=case, seed=seed, scoring=scoring)
                historical = Path(f"runs/cube_cem_initial/{case}_cem_{seed}.npz")
                if scoring == "original" and historical.exists():
                    with np.load(historical) as saved:
                        report["matches_historical_commands"] = bool(np.array_equal(e["commands"], saved["commands"]))
                np.savez_compressed(output / f"{key}.npz", **{k: e[k] for k in (
                    "states", "fine_states", "commands", "predictions", "timings", "joint_limits")}, target=target)
                write_json(output / f"{key}.json", dict(report, costs=e["costs"], diagnostics=e["diagnostics"]))
                records.append({"case": case, "seed": seed, "scoring": scoring, "target": target, "episode": e, "report": report})
                print(f"{key}: success={report['success']} final={report['final_distance_m']:.4f}m "
                      f"escaped={report['escaped']} vmax={report['maximum_cube_speed_m_s']:.3f}m/s", flush=True)
    summary = {"runs": {f"{r['case']}_{r['seed']}_{r['scoring']}": r["report"] for r in records},
               "aggregate": {name: summarize([r for r in records if r["scoring"] == name]) for name in ("original", "workspace")},
               "per_case": {str(case): {name: summarize([r for r in records if r["case"] == case and r["scoring"] == name])
                                        for name in ("original", "workspace")} for case in CASES}}
    comparison_plots(output, records, workspace)
    after = {p: file_hash(p) for p in before}
    prior_after = {p: file_hash(p) for p in prior}
    summary.update(source_hashes_after=after, sources_unchanged=before == after,
                   prior_experiment_unchanged=prior == prior_after)
    write_json(output / "summary.json", summary)
    if before != after or prior != prior_after:
        raise RuntimeError("Existing source or experiment artifacts changed")
    return summary
