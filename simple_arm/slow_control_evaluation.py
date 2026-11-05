"""Compare old and slow-trained world models with the same 0.2 rad/s MPC."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contact_learning import new_plot
from .cube_mpc import Config, DATASET, DT, ModelAdapter
from .cube_mpc_comparison import summarize, workspace_metrics
from .cube_mpc_experiment import CASES, SEEDS, file_hash, load_cases, metrics, run_episode, source_hashes, write_json
from .cube_rate_experiment import motion_metrics
from .cube_workspace import Workspace
from .slow_training import SOURCE


def compare(candidate, output, baseline=SOURCE, rate=.2):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    config = Config(command_rate_limit=rate)
    if rate <= 0:
        raise ValueError("Use a positive command-rate limit")
    before = source_hashes(candidate, DATASET)
    before[str(baseline)] = file_hash(baseline)
    models = {"baseline": ModelAdapter(baseline), "slow_refined": ModelAdapter(candidate)}
    for model in models.values():
        metadata = model.checkpoint["metadata"]
        if any(parent["seed"] == 292 for parent in [metadata, *metadata.get("parents", [])]):
            raise ValueError("Benchmark collection seed overlaps model training")
    workspace, cases = Workspace(), load_cases(DATASET)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"candidate": str(candidate), "baseline": str(baseline), "config": asdict(config),
               "seeds": list(SEEDS), "cases": list(CASES), "duration_s": 5., "source_hashes_before": before,
               "intervention": "Model only; both use original score and identical 0.2 rad/s command constraint"})
    records = []
    for case, (initial, target) in cases.items():
        for seed in SEEDS:
            for name, model in models.items():
                c = Config(command_rate_limit=rate, seed=seed)
                e = run_episode(initial, target, model, c)
                r = dict(metrics(e, target), **workspace_metrics(e, workspace), **motion_metrics(e))
                r.update(case=case, seed=seed, model=name)
                key = f"{case}_{seed}_{name}"
                write_json(output / f"{key}.json", dict(r, costs=e["costs"], diagnostics=e["diagnostics"]))
                np.savez_compressed(output / f"{key}.npz", **{k: e[k] for k in (
                    "states", "fine_states", "commands", "predictions", "timings", "joint_limits")}, target=target)
                records.append(dict(case=case, seed=seed, model=name, target=target, episode=e, report=r))
                print(f"{key}: success={r['success']} final={r['final_distance_m']:.4f}m "
                      f"cube_vmax={r['maximum_cube_speed_m_s']:.3f}m/s escaped={r['escaped']}", flush=True)
    summary = {"runs": {f"{r['case']}_{r['seed']}_{r['model']}": r["report"] for r in records},
               "aggregate": {name: summarize([r for r in records if r["model"] == name]) for name in models},
               "source_hashes_after": {p: file_hash(p) for p in before}}
    summary["sources_unchanged"] = summary["source_hashes_after"] == before
    write_json(output / "summary.json", summary)
    figure, _ = new_plot()
    figure.set_size_inches(12, 8)
    figure.clear()
    for panel, case in zip(figure.subplots(2, 2).flat, CASES):
        for r in records:
            if r["case"] != case:
                continue
            e = r["episode"]
            panel.plot(np.arange(len(e["states"])) * DT, np.linalg.norm(e["states"][:, 4:6] - r["target"], axis=1),
                       color="tab:blue" if r["model"] == "slow_refined" else "tab:orange",
                       linestyle={42: "-", 43: "--", 44: ":"}[r["seed"]], label=f"{r['model']} {r['seed']}")
        panel.axhline(.02, color="gray", linestyle=":")
        panel.set(title=f"Episode {case}", xlabel="Simulation time (s)", ylabel="Actual target distance (m)")
        panel.legend(fontsize=7)
    figure.savefig(output / "actual_distance.png")
    if not summary["sources_unchanged"]:
        raise RuntimeError("Existing sources changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("runs/arm_cube_slow_02_expanded/refined/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/arm_cube_slow_02_control"))
    parser.add_argument("--baseline", type=Path, default=SOURCE)
    parser.add_argument("--rate", type=float, default=.2)
    args = parser.parse_args()
    compare(args.candidate, args.output, args.baseline, args.rate)


if __name__ == "__main__":
    main()
