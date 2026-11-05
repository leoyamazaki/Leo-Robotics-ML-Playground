"""Matched stationary-loss/horizon study on saved data; never collects physics."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .capacity_screen import DATASET, prepare
from .contact_boundary import load_boundary
from .contact_learning import load_predictor, new_plot, rollout
from .cube_mpc import CHECKPOINT
from .cube_mpc_experiment import file_hash, source_hashes, write_json
from .stationary_training import TrainingConfig, train


OUTPUT = Path("runs/arm_cube_stationary_experiment")
INDEPENDENT = {"boundary": Path("data/arm_cube_contact_boundary/independent.npz"),
               "broad": Path("data/arm_cube_contact_boundary/broad_independent.npz")}


def vector_rmse(error):
    return float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))) if len(error) else None


def drift_metrics(trajectory, actual, mask):
    if not mask.any():
        return {"episodes": 0}
    movement = np.linalg.norm(trajectory[mask, :, 4:6] - actual[mask, :1, 4:6], axis=-1).max(1)
    return {"episodes": int(mask.sum()), "mean_max_drift_m": float(movement.mean()),
            "median_max_drift_m": float(np.median(movement)), "p95_max_drift_m": float(np.quantile(movement, .95)),
            "worst_max_drift_m": float(movement.max()), "fraction_over_2mm": float((movement > .002).mean())}


def case_masks(dataset):
    actual = dataset["states"]
    contact_duration = dataset["contacts"].sum((1, 2)) * .002
    excursion = np.linalg.norm(actual[..., 4:6] - actual[:, :1, 4:6], axis=-1).max(1)
    if dataset["metadata"]["kinds"] == ["overhead", "near_miss", "light_contact", "clear_push"]:
        return {"no_contact": np.isin(dataset["kinds"], [0, 1]),
                "light_contact": dataset["kinds"] == 2, "clear_push": dataset["kinds"] == 3}
    return {"no_contact": contact_duration == 0,
            "light_contact": (contact_duration > 0) & (contact_duration <= .2 + 1e-9) & (excursion <= .02),
            "clear_push": (contact_duration >= .1 - 1e-9) & (excursion >= .05)}


def prediction_metrics(dataset, teacher_forced, autoregressive, accumulated):
    actual = dataset["states"][:, ::5]
    interval_contacts = dataset["contacts"].reshape(len(actual), 40, 50).sum(-1)
    interval_motion = np.linalg.norm(actual[:, 1:, 4:6] - actual[:, :-1, 4:6], axis=-1)
    still = (np.linalg.norm(dataset["states"][..., 4:6] - actual[:, :1, 4:6], axis=-1).max(1) <= .002)
    no_contact = ~dataset["contacts"].any((1, 2))
    masks = case_masks(dataset)
    one_step = teacher_forced[..., 4:6] - actual[:, 1:, 4:6]
    push = (~no_contact) & (np.linalg.norm(actual[:, -1, 4:6] - actual[:, 0, 4:6], axis=-1) > .02)
    report = {"episodes": len(actual), "stationary_episodes": int(still.sum()),
              "one_step_overall_rmse_m": vector_rmse(one_step),
              "one_step_cases": {name: {"episodes": int(mask.sum()), "transitions": int(mask.sum()) * 40,
                                         "cube_displacement_rmse_m": vector_rmse(one_step[mask])} for name, mask in masks.items()},
              "one_step_intervals": {}, "stationary_drift": {}, "teacher_forced_vs_autoregressive": {},
              "push_4s": {"episodes": int(push.sum()),
                          "cube_endpoint_rmse_m": vector_rmse(autoregressive[push, -1, 4:6] - actual[push, -1, 4:6])},
              "clear_push_4s_rmse_m": vector_rmse(autoregressive[masks["clear_push"], -1, 4:6] - actual[masks["clear_push"], -1, 4:6])}
    for name, mask in {"no_contact": interval_contacts == 0,
                       "light_contact": (interval_contacts > 0) & (interval_contacts <= 10),
                       "active_push": (interval_contacts > 0) & (interval_motion > .002)}.items():
        report["one_step_intervals"][name] = {"transitions": int(mask.sum()), "cube_displacement_rmse_m": vector_rmse(one_step[mask])}
    stationary_masks = {"all_stationary": still, "stationary_no_contact": still & no_contact}
    if "overhead" in dataset["metadata"]["kinds"]:
        stationary_masks.update({name: still & (dataset["kinds"] == index) for index, name in enumerate(dataset["metadata"]["kinds"][:2])})
    for name, mask in stationary_masks.items():
        report["stationary_drift"][name] = {
            "autoregressive": drift_metrics(autoregressive, actual, mask),
            "teacher_forced_next": drift_metrics(teacher_forced, actual, mask),
            "teacher_forced_accumulated_diagnostic": drift_metrics(accumulated, actual, mask)}
    for name, mask in dict(all=np.ones(len(actual), dtype=bool), **masks).items():
        report["teacher_forced_vs_autoregressive"][name] = {
            "episodes": int(mask.sum()), "teacher_forced_one_step_rmse_m": vector_rmse(one_step[mask]),
            "autoregressive_prefix_position_rmse_m": vector_rmse(autoregressive[mask, 1:, 4:6] - actual[mask, 1:, 4:6]),
            "autoregressive_increment_rmse_m": vector_rmse(np.diff(autoregressive[mask, :, 4:6], axis=1) - np.diff(actual[mask, :, 4:6], axis=1)),
            "autoregressive_4s_endpoint_rmse_m": vector_rmse(autoregressive[mask, -1, 4:6] - actual[mask, -1, 4:6]),
            "teacher_forced_accumulated_4s_diagnostic_rmse_m": vector_rmse(accumulated[mask, -1, 4:6] - actual[mask, -1, 4:6])}
    return report


def evaluate(dataset, checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if any(p["seed"] == dataset["metadata"]["seed"] for p in [saved["metadata"], *saved["metadata"].get("parents", [])]):
        raise ValueError("Independent evaluation overlaps training seed")
    actual, actions = dataset["states"][:, ::5], dataset["actions"][:, ::5]
    predict = load_predictor(checkpoint, 5)
    teacher = predict(actual[:, :-1], actions)
    recursive = rollout(predict, actual[:, 0], actions)
    # Sum local predictions from truth only as an error-accumulation diagnostic;
    # this is explicitly not an autoregressive physical rollout.
    accumulated = actual.copy()
    accumulated[:, 1:, 4:6] = actual[:, :1, 4:6] + np.cumsum(teacher[..., 4:6] - actual[:, :-1, 4:6], axis=1)
    report = prediction_metrics(dataset, teacher, recursive, accumulated)
    report.update(checkpoint=str(checkpoint), dataset_metadata=dataset["metadata"],
                  definitions={"stationary_episode": "All saved 20 ms XY samples within 2 mm of initial position",
                               "case_categories": "Boundary uses saved collector kinds; broad uses no contact, light <=0.2s contact with <=2cm excursion, clear >=0.1s contact with >=5cm excursion",
                               "interval_categories": "No contact: zero saved contact substeps; light: 1..10 of 50 active; active push: saved contact plus >2mm XY displacement; light/active may overlap",
                               "teacher_forced_accumulated": "Diagnostic sum of predicted local displacements from truth, not a physical rollout"})
    write_json(output / "metrics.json", report)
    # `learned` aliases the AR prediction so the existing replay viewer works.
    np.savez_compressed(output / "test_rollouts.npz", actual=actual, actions=actions, learned=recursive,
                        teacher_forced_next=teacher, teacher_forced_accumulated=accumulated)
    write_json(output / "evaluation.json", {"dt": .1, "checkpoint": str(checkpoint)})
    return report


def plots(summary, output):
    figure, axes = new_plot()
    for name, model in summary["models"].items():
        if "training" not in model:
            continue
        history = model["training"]["history"]
        axes.plot([r["epoch"] for r in history], [r["validation_40_step"]["state_scaled_mse"] for r in history], label=name)
    axes.set(xlabel="Additional epoch", ylabel="Common 40-step validation state MSE")
    axes.legend(fontsize=7)
    figure.savefig(output / "validation_losses.png")
    for dataset in INDEPENDENT:
        figure, axes = new_plot()
        for name, model in summary["models"].items():
            with np.load(output / "evaluations" / dataset / name / "test_rollouts.npz") as saved:
                actual, ar, teacher = saved["actual"], saved["learned"], saved["teacher_forced_next"]
                rmse = np.sqrt(np.mean(np.sum((ar[:, 1:, 4:6] - actual[:, 1:, 4:6])**2, axis=-1), axis=0))
                tf = np.sqrt(np.mean(np.sum((teacher[..., 4:6] - actual[:, 1:, 4:6])**2, axis=-1), axis=0))
            line, = axes.plot(np.arange(1, 41) * .1, rmse * 1000, label=f"{name} AR")
            axes.plot(np.arange(1, 41) * .1, tf * 1000, linestyle=":", color=line.get_color(), label=f"{name} teacher-forced")
        axes.set(xlabel="Time (s)", ylabel="Cube XY vector RMSE (mm)", title=dataset)
        axes.legend(fontsize=6)
        figure.savefig(output / f"{dataset}_teacher_forced_vs_ar.png")
        figure, axes = new_plot()
        for name, model in summary["models"].items():
            report = model["evaluation"][dataset]
            drift = report["stationary_drift"]["stationary_no_contact"]["autoregressive"]
            if drift["episodes"]:
                axes.scatter(drift["mean_max_drift_m"] * 1000, report["push_4s"]["cube_endpoint_rmse_m"] * 100, label=name)
        axes.set(xlabel="Mean max stationary/no-contact drift (mm)", ylabel="Actual-push 4 s endpoint RMSE (cm)", title=f"{dataset}: lower/left is better")
        axes.legend(fontsize=7)
        figure.savefig(output / f"{dataset}_drift_push_tradeoff.png")


def run(output=OUTPUT, source=CHECKPOINT, config=TrainingConfig()):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    torch.set_num_threads(2)
    before = source_hashes(source, DATASET)
    dataset = load_boundary(DATASET)
    checkpoint = torch.load(source, weights_only=True, map_location="cpu")
    if len(dataset["states"]) * 40 != 99200 or np.bincount(dataset["split"]).tolist() != [1736, 372, 372]:
        raise ValueError("Expected original 99,200 transitions and unchanged splits")
    if checkpoint.get("hidden_width") != 256 or checkpoint.get("training_method") != "one_step_plus_recursive":
        raise ValueError("Expected existing recursively refined Medium checkpoint")
    prepare(dataset, checkpoint)  # verify original normalization and provenance
    independent = {name: load_boundary(path) for name, path in INDEPENDENT.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"config": asdict(config), "source": str(source), "dataset": str(DATASET),
               "source_hashes_before": before, "independent_datasets": {k: str(v) for k, v in INDEPENDENT.items()},
               "window_starts": list(range(21)), "batch_order": "Same dedicated seeded generator per run",
               "budget": "Equal epochs, windows, batch size and optimizer updates; 20 steps costs twice the forward/backward model transitions",
               "selector": "Lowest unweighted 40-step state-scaled validation MSE, including epoch zero",
               "stationary_anchor": "Actual beginning of each 100 ms interval; all five saved 20 ms samples must stay within threshold",
               "no_collection": True, "state_scales": [1, 1, 3, 3, .1, .1, .01, .5, .5, .5, .5, .1, 2]})
    models = {"baseline": {"checkpoint": str(source), "training": None}}
    for name, horizon, weight in (("control_10", 10, 0.), ("stationary_10", 10, config.stationary_weight),
                                  ("stationary_20", 20, config.stationary_weight)):
        stats = train(dataset, source, output / name, horizon, weight, config)
        models[name] = {"checkpoint": str(output / name / "model.pt"), "training": stats}
    # No independent metrics are inspected until all validation selections finish.
    for name, info in models.items():
        saved = torch.load(info["checkpoint"], weights_only=True, map_location="cpu")
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            if not torch.equal(saved[key], checkpoint[key]):
                raise RuntimeError("Normalization changed")
        info["evaluation"] = {data_name: evaluate(data, info["checkpoint"], output / "evaluations" / data_name / name)
                              for data_name, data in independent.items()}
        for data_name, report in info["evaluation"].items():
            print(f"{name}/{data_name}: drift={report['stationary_drift']['stationary_no_contact']['autoregressive']} "
                  f"push={report['push_4s']}", flush=True)
    summary = {"models": models, "source_hashes_after": {p: file_hash(p) for p in before}}
    summary["sources_unchanged"] = summary["source_hashes_after"] == before
    write_json(output / "summary.json", summary)
    # Baseline has no new training history.
    plot_summary = dict(summary, models={k: dict(v) for k, v in models.items()})
    plot_summary["models"]["baseline"].pop("training")
    plots(plot_summary, output)
    if not summary["sources_unchanged"]:
        raise RuntimeError("Existing checkpoints or datasets changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=CHECKPOINT)
    for name, value in asdict(TrainingConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    run(args.output, args.source, TrainingConfig(**{name: getattr(args, name) for name in asdict(TrainingConfig())}))


if __name__ == "__main__":
    main()
