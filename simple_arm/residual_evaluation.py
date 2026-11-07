"""Paired held-out evaluation and CPU latency for residual-training checkpoints."""

from pathlib import Path
import json
import time

import numpy as np
import torch

from .contact_learning import (differentiable_next, labels, load_predictor, new_plot,
                               physical_errors, rollout)
from .contact_models import model_from_checkpoint
from .one_step_stationary import displacement_metrics
from .residual_training import file_hash, load_data, write_json
from .stationary_experiment import INDEPENDENT, prediction_metrics


def subset(dataset, mask):
    count = len(dataset["states"])
    return {key: value[mask] if isinstance(value, np.ndarray) and value.ndim and len(value) == count else value
            for key, value in dataset.items()}


def batched_predictor(checkpoint, batch_size=256):
    predict = load_predictor(checkpoint, 5)

    def batched(states, actions):
        shape = states.shape
        flat, commands = states.reshape(-1, 13), actions.reshape(-1, 2)
        return np.concatenate([predict(flat[start:start + batch_size], commands[start:start + batch_size])
                               for start in range(0, len(flat), batch_size)], axis=0).reshape(shape)
    return batched


def evaluate_checkpoint(dataset, checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actual, actions = dataset["states"][:, ::5], dataset["actions"][:, ::5]
    predict = batched_predictor(checkpoint)
    try:
        teacher = predict(actual[:, :-1], actions)
        ar = rollout(predict, actual[:, 0], actions)
    except RuntimeError as error:
        raise RuntimeError(f"Evaluation failed for {checkpoint}: {error}") from error
    accumulated = actual.copy()
    accumulated[:, 1:, 4:6] = actual[:, :1, 4:6] + np.cumsum(teacher[..., 4:6] - actual[:, :-1, 4:6], axis=1)
    active = dataset["contacts"].reshape(len(actual), 40, 50).any(-1)
    masks = dict(labels(dataset, 5), contact=active, without_contact=~active)
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
              "episodes": len(actual), "one_step": physical_errors(teacher, actual[:, 1:], actual[:, :-1]),
              "one_step_conditioned": {key: physical_errors(teacher[mask], actual[:, 1:][mask], actual[:, :-1][mask])
                                       if mask.any() else {"samples": 0} for key, mask in masks.items()},
              "displacement": displacement_metrics(dataset, teacher),
              "diagnostics": prediction_metrics(dataset, teacher, ar, accumulated),
              "rollout": {}, "endpoint": {}}
    for seconds in (1, 2, 4):
        stop = seconds * 10
        report["rollout"][str(seconds)] = physical_errors(ar[:, 1:stop + 1], actual[:, 1:stop + 1],
                                                        np.broadcast_to(actual[:, :1], actual[:, 1:stop + 1].shape))
        report["endpoint"][str(seconds)] = physical_errors(ar[:, stop], actual[:, stop], actual[:, 0])
    write_json(output / "metrics.json", report)
    np.savez_compressed(output / "test_rollouts.npz", actual=actual, actions=actions, learned=ar,
                        teacher_forced_next=teacher, teacher_forced_accumulated=accumulated)
    # Existing contact viewer understands these artifacts.
    write_json(output / "evaluation.json", {"dt": .1, "checkpoint": str(checkpoint), "evaluation_metadata": dataset["metadata"]})
    figure, axes = new_plot()
    axes.plot(np.arange(1, 41) * .1, np.sqrt(np.mean(np.sum((ar[:, 1:, 4:6] - actual[:, 1:, 4:6])**2, -1), 0)) * 100)
    axes.set(xlabel="Rollout horizon (s)", ylabel="Cube XY vector RMSE (cm)")
    figure.savefig(output / "rollout_error.png")
    return report


def cpu_latency(checkpoint, states, actions, warmup=10, repeats=50):
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = model_from_checkpoint(saved).eval().requires_grad_(False)
    model.load_state_dict(saved["weights"])
    states = torch.tensor(states.reshape(-1, 13), dtype=torch.float32)
    actions = torch.tensor(actions.reshape(-1, 2), dtype=torch.float32)
    report = {"threads": 2, "warmup": warmup, "repeats": repeats,
              "measurement": "CPU float32 complete tensor transition, including features, normalization and apply-delta; no disk or NumPy conversion",
              "torch_version": str(torch.__version__), "batch_sizes": {}}
    with torch.inference_mode():
        for size in (1, 256):
            indices = torch.arange(size) % len(states)
            x, u = states[indices].contiguous(), actions[indices].contiguous()
            for _ in range(warmup):
                differentiable_next(model, x, u, saved)
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                differentiable_next(model, x, u, saved)
                timings.append((time.perf_counter() - started) * 1000)
            report["batch_sizes"][str(size)] = {"median_ms": float(np.median(timings)),
                                               "mean_ms": float(np.mean(timings)),
                                               "p95_ms": float(np.quantile(timings, .95)), "samples_ms": timings}
    return report


def training_seeds(metadata):
    return {metadata.get("seed"), *(seed for parent in metadata.get("parents", []) for seed in training_seeds(parent))}


def evaluate_run(dataset_path, run, output, historical=None):
    torch.set_num_threads(2)
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Require a new evaluation directory: {output}")
    dataset = load_data(dataset_path)
    checkpoints = {stage: run / stage / "model.pt" for stage in ("one_step", "recursive")}
    if historical:
        checkpoints["historical_reference"] = Path(historical)
    saved = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {path}; copy original runs/ artifacts or complete training first")
        saved[name] = torch.load(path, map_location="cpu", weights_only=True)
        if name != "historical_reference" and saved[name]["provenance"]["dataset_sha256"] != file_hash(dataset_path):
            raise ValueError(f"Evaluation dataset differs from {name} training archive")
    datasets = {"mixed_test": subset(dataset, dataset["split"] == 2)}
    omitted = {}
    for name, path in INDEPENDENT.items():
        if path.is_file():
            independent = load_data(path)
            for stage in ("one_step", "recursive"):
                if training_seeds(independent["metadata"]) & training_seeds(saved[stage]["metadata"]):
                    raise ValueError(f"Independent {name} collection seed overlaps training")
            datasets[name] = independent
        else:
            omitted[name] = f"Unavailable: {path}"
    output.mkdir(parents=True)
    actual, actions = datasets["mixed_test"]["states"][:, ::5], datasets["mixed_test"]["actions"][:, ::5]
    summary = {"datasets": {"mixed_test": {"path": str(dataset_path), "sha256": file_hash(dataset_path), "split": 2},
                            **{name: {"path": str(INDEPENDENT[name]), "sha256": file_hash(INDEPENDENT[name]), "split": "all episodes"}
                               for name in datasets if name != "mixed_test"}},
               "omitted_independent": omitted, "models": {},
               "selection": "Validation only. Test and independent results are evaluation; controller defaults unchanged.",
               "definitions": {"physical_errors": "Existing contact_learning.physical_errors",
                               "stationary_and_motion_rates": "Existing one_step_stationary.displacement_metrics: stationary speed <=5mm/s and excursion <=0.5mm across all five saved samples; false/missed thresholds 0.5mm and 2mm",
                               "drift": "Existing stationary_experiment.prediction_metrics: all saved episode XY within 2mm of initial position"}}
    for name, path in checkpoints.items():
        summary["models"][name] = {"checkpoint": str(path), "parameter_count": sum(p.numel() for p in model_from_checkpoint(saved[name]).parameters()),
                                    "selected_epoch": saved[name].get("refinement_epoch", saved[name].get("epoch")),
                                    "label": "Historical reference; training history and inputs differ; not an isolated architecture comparison" if name == "historical_reference" else name,
                                    "latency": cpu_latency(path, actual[:, :-1], actions), "evaluation": {}}
        history = run / name / "history.json"
        if name != "historical_reference" and history.is_file():
            rows = json.loads(history.read_text(encoding="utf-8"))
            summary["models"][name]["training"] = {"epochs_completed": rows[-1]["epoch"],
                                                      "recorded_seconds": sum(row["seconds"] for row in rows)}
        for dataset_name, cases in datasets.items():
            summary["models"][name]["evaluation"][dataset_name] = evaluate_checkpoint(cases, path, output / dataset_name / name)
    write_json(output / "summary.json", summary)
    write_report(summary, output / "REPORT.md")
    return summary


def write_report(summary, path):
    lines = ["# Base-relative residual experiment", "", summary["selection"], "",
             "Architecture: 15 → 768 SiLU, four two-linear 768-wide residual blocks with SiLU after addition, 768 → 12 linear. 4,746,252 parameters; base-relative XY inputs. Fresh training-only normalization, preserved during refinement. Float32; no stationary penalty.", "",
             "|Checkpoint|Selected epoch|Completed epochs|Recorded stage seconds|CPU batch 1 ms|CPU batch 256 ms|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, info in summary["models"].items():
        training = info.get("training", {})
        latency = info["latency"]["batch_sizes"]
        lines.append(f"|{name}|{info['selected_epoch']}|{training.get('epochs_completed', 'unavailable')}|{training.get('recorded_seconds', 'unavailable')}|{latency['1']['median_ms']:.3f}|{latency['256']['median_ms']:.3f}|")
    if summary["models"]["recursive"]["selected_epoch"] == 0:
        lines += ["", "Recursive epoch zero won validation: none of the refinement epochs improved the initial selected one-step checkpoint."]
    if "historical_reference" in summary["models"]:
        lines += ["", summary["models"]["historical_reference"]["label"] + "."]
    for dataset_name in summary["datasets"]:
        lines += ["", f"## {dataset_name}", "",
                  "|Checkpoint|One-step XY mm|Contact XY mm|Stationary XY mm|False 2mm %|Missed 2mm %|1s endpoint cm|2s endpoint cm|4s endpoint cm|Stationary max drift mm|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(value, scale=1):
            return "unavailable" if value is None else f"{value * scale:.3f}"
        for name, info in summary["models"].items():
            r = info["evaluation"][dataset_name]
            rate = r["displacement"]["rates"]["0.002"]
            drift = r["diagnostics"]["stationary_drift"]["all_stationary"]["autoregressive"].get("mean_max_drift_m")
            values = [fmt(r["one_step"]["displacement_vector_rmse_m"], 1000),
                      fmt(r["one_step_conditioned"]["contact"].get("displacement_vector_rmse_m"), 1000),
                      fmt(r["displacement"]["stationary"]["rmse_m"], 1000),
                      fmt(rate["false_motion_rate"], 100), fmt(rate["missed_motion_rate"], 100),
                      *[fmt(r["endpoint"][str(sec)]["displacement_vector_rmse_m"], 100) for sec in (1, 2, 4)], fmt(drift, 1000)]
            lines.append("|" + "|".join([name, *values]) + "|")
        a, b = [summary["models"][stage]["evaluation"][dataset_name] for stage in ("one_step", "recursive")]
        lines += ["", "Prefix rollout cube XY RMSE (cm), averaging every predicted step through each horizon:", "",
                  "|Checkpoint|Through 1s|Through 2s|Through 4s|", "|---|---:|---:|---:|"]
        for name, info in summary["models"].items():
            r = info["evaluation"][dataset_name]
            lines.append("|" + "|".join([name, *[fmt(r["rollout"][str(sec)]["displacement_vector_rmse_m"], 100) for sec in (1, 2, 4)]]) + "|")
        comparisons = [("one-step XY RMSE", a["one_step"]["displacement_vector_rmse_m"], b["one_step"]["displacement_vector_rmse_m"]),
                       ("contact one-step XY RMSE", a["one_step_conditioned"]["contact"].get("displacement_vector_rmse_m"), b["one_step_conditioned"]["contact"].get("displacement_vector_rmse_m")),
                       ("stationary XY RMSE", a["displacement"]["stationary"]["rmse_m"], b["displacement"]["stationary"]["rmse_m"]),
                       ("false 2mm motion rate", a["displacement"]["rates"]["0.002"]["false_motion_rate"], b["displacement"]["rates"]["0.002"]["false_motion_rate"]),
                       ("missed 2mm motion rate", a["displacement"]["rates"]["0.002"]["missed_motion_rate"], b["displacement"]["rates"]["0.002"]["missed_motion_rate"]),
                       *[(f"{sec}s endpoint RMSE", a["endpoint"][str(sec)]["displacement_vector_rmse_m"], b["endpoint"][str(sec)]["displacement_vector_rmse_m"]) for sec in (1, 2, 4)],
                       *[(f"{sec}s prefix rollout RMSE", a["rollout"][str(sec)]["displacement_vector_rmse_m"], b["rollout"][str(sec)]["displacement_vector_rmse_m"]) for sec in (1, 2, 4)],
                       ("stationary maximum drift", a["diagnostics"]["stationary_drift"]["all_stationary"]["autoregressive"].get("mean_max_drift_m"), b["diagnostics"]["stationary_drift"]["all_stationary"]["autoregressive"].get("mean_max_drift_m"))]
        regressions = [name for name, before, after in comparisons if before is not None and after is not None and after > before]
        lines += ["", "Recursive regressions versus selected one-step: " + (", ".join(regressions) if regressions else "none among the listed comparison metrics") + "."]
    if summary["omitted_independent"]:
        lines += ["", "Independent datasets omitted: " + "; ".join(summary["omitted_independent"].values()) + "."]
    lines += ["", "Full physical-state errors, 1/2/4-second prefix and endpoint errors, contact cohorts, drift, false/missed motion, and latency samples are in summary.json and per-model metrics.json. Latency uses two CPU threads, 10 warm-up calls and 50 measurements under identical conditions; it describes this desktop CPU, not laptop throughput.", "",
              "## Future comparison", "", "Train a plain MLP with three hidden layers of width 1,536 (4,764,684 parameters, versus 4,746,252 here), using matching base-relative features, normalization, data splits, optimization budgets and seeds. It has not been trained in this experiment."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
