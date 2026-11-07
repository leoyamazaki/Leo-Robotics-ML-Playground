"""Paired ten-second frozen residual-model approach experiment."""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from .contact_boundary import load_boundary
from .cube_mpc import ModelAdapter, tip_xy
from .cube_workspace import Workspace
from .cube_mpc_experiment import (CASES, SEEDS, file_hash, load_cases, metrics,
                                  run_episode, timing_metrics, write_json)


def compare(checkpoint, dataset, output, config, device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    predict = ModelAdapter(checkpoint, device=device)
    cases, geometry = load_cases(dataset, expected_seed=None), Workspace()
    dataset_seed = load_boundary(dataset)["metadata"]["seed"]
    before = {str(p): file_hash(p) for p in (checkpoint, dataset)}
    output.mkdir(parents=True)
    write_json(output / "config.json", dict(config=asdict(config), checkpoint=str(checkpoint),
        dataset=str(dataset), hashes=before, device=str(predict.device),
        dataset_seed=dataset_seed,
        input_coordinate_frame=predict.checkpoint.get("input_coordinate_frame", "tip_relative"), duration_s=10., cases=list(CASES), seeds=list(SEEDS),
        success_definition="Within 2 cm throughout final 0.5 s at 2 ms physics samples",
        contact_definition="MuJoCo active penetrating cube contact with pusher or arm links",
        distance_definition="XY pusher tip center to cube centroid, including initial state",
        initial_states={str(k): s.tolist() for k, (s, _) in cases.items()},
        targets={str(k): t.tolist() for k, (_, t) in cases.items()}))
    reports, times = {}, {s: [] for s in ("original", "approach")}
    for case, (initial, target) in cases.items():
        for seed in SEEDS:
            for scoring in ("original", "approach"):
                key = f"{case}_{seed}_{scoring}"
                e = run_episode(initial, target, predict, replace(config, scoring=scoring, seed=seed), duration=10.)
                r = metrics(e, target)
                contacts = e["actual_contacts"]
                fine = np.vstack([initial, e["fine_states"]])
                with torch.inference_mode():
                    tip = tip_xy(torch.tensor(fine), geometry).numpy()
                r.update(actual_contact_duration_s=float(contacts.sum() * .002),
                    time_to_first_contact_s=float((np.flatnonzero(contacts)[0] + 1) * .002) if contacts.any() else None,
                    minimum_actual_tip_to_cube_distance_m=float(np.linalg.norm(tip - fine[:, 4:6], axis=1).min()),
                    case=case, seed=seed, scoring=scoring)
                np.savez_compressed(output / f"{key}.npz", **{k: v for k, v in e.items()
                    if isinstance(v, np.ndarray)}, target=target)
                write_json(output / f"{key}.json", dict(r, costs=e["costs"], diagnostics=e["diagnostics"]))
                reports[key] = r
                times[scoring].extend(e["timings"])
                print(f"{key}: success={r['success']} final={r['final_distance_m']:.5f}m contact={r['actual_contact_duration_s']:.3f}s", flush=True)
    aggregate = {}
    for scoring in times:
        subset = [r for r in reports.values() if r["scoring"] == scoring]
        aggregate[scoring] = dict(successes=sum(r["success"] for r in subset), runs=len(subset),
            **{k: float(np.mean([r[k] for r in subset])) for k in
               ("final_distance_m", "actual_contact_duration_s", "minimum_actual_tip_to_cube_distance_m")},
            timing=timing_metrics(times[scoring]))
    pairs = []
    for case in CASES:
        for seed in SEEDS:
            a, b = (reports[f"{case}_{seed}_{s}"] for s in ("original", "approach"))
            pairs.append(dict(case=case, seed=seed, final_error_improvement_m=a["final_distance_m"]-b["final_distance_m"]))
    unchanged = before == {p: file_hash(p) for p in before}
    write_json(output / "summary.json", dict(aggregate=aggregate, pairs=pairs, runs=reports, sources_unchanged=unchanged))
    lines = ["# Approach scoring comparison", "", "Frozen residual checkpoint; paired saved states, targets and CEM seeds. 10 simulated seconds per run.",
             f"Inference device: {predict.device}. Saved dataset seed: {dataset_seed}; older benchmarks expected seed 292. Config JSON records exact hashes, initial states and targets.", "",
             "| Mode | Success | Mean final error | Mean contact | Mean / p95 planning |", "|---|---:|---:|---:|---:|"]
    for s, a in aggregate.items():
        lines.append(f"| {s} | {a['successes']}/12 | {a['final_distance_m']:.5f} m | {a['actual_contact_duration_s']:.3f} s | {a['timing']['mean_s']*1000:.1f} / {a['timing']['p95_s']*1000:.1f} ms |")
    lines += ["", "| Case | Seed | Original final error | Approach final error | Improvement |", "|---|---:|---:|---:|---:|"]
    for p in pairs:
        prefix = f"{p['case']}_{p['seed']}"
        lines.append(f"| {p['case']} | {p['seed']} | {reports[prefix+'_original']['final_distance_m']:.5f} | {reports[prefix+'_approach']['final_distance_m']:.5f} | {p['final_error_improvement_m']:+.5f} m |")
    lines += ["", "Contact is measured in actual physics; predicted displacement and individual selected costs are saved separately in run JSON files. All trajectories, including failures, are retained.",
              "", "Original remains the default. This radius does not choose a pushing side or guarantee contact. If the measured gains warrant another experiment, test 30 steps with the same five knots and 100 ms steps; defer learned terminal values.",
              f"", f"Checkpoint and dataset unchanged: {unchanged}."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("Checkpoint or dataset changed")
    return aggregate
