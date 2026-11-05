"""Equal-budget recursive follow-up of the saved medium/large capacity screen."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .capacity_screen import DATASET, OUTPUT as SCREEN, file_hash, prepare, summarize_evaluation
from .contact_boundary import evaluate_boundary, load_boundary
from .contact_learning import new_plot, train_multistep


OUTPUT = Path("runs/arm_cube_capacity_refinement")


def plot_behavior_comparison(result, output):
    labels, metrics = [], []
    for name, pair in result["comparison"].items():
        prefix = "M" if name == "medium_256" else "L"
        for method, evaluation in pair.items():
            labels.append(prefix + (" one-step" if method == "one_step_only" else " refined"))
            metrics.append(evaluation)
    labels.append("128 historical")
    metrics.append(result["historical_128_refined"])
    for filename, ylabel, values in (
        ("drift_comparison.png", "Mean maximum no-contact drift (mm)", [1000 * m["no_contact"]["mean_max_drift_m"] for m in metrics]),
        ("clear_push_comparison.png", "Clear-push 4 s cube XY endpoint RMSE (cm)", [100 * m["boundary_clear_push"]["learned"]["horizon_endpoints"]["4"]["displacement_vector_rmse_m"] for m in metrics]),
    ):
        figure, axes = new_plot()
        axes.bar(range(len(labels)), values)
        axes.set_xticks(range(len(labels)), labels, rotation=15)
        axes.set(ylabel=ylabel)
        figure.savefig(Path(output) / filename)


def run(output=OUTPUT, epochs=10):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing refinement experiment: {output}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(2)
    dataset = load_boundary(DATASET)
    screen = json.loads((SCREEN / "summary.json").read_text(encoding="utf-8"))
    if file_hash(DATASET) != screen["config"]["dataset_sha256"]:
        raise ValueError("Screening dataset has changed")
    sources = {name: SCREEN / name / "model.pt" for name in ("medium_256", "large_512")}
    for name, path in sources.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        prepare(dataset, checkpoint)
        if checkpoint["hidden_width"] != (256 if name == "medium_256" else 512):
            raise ValueError("Unexpected source checkpoint width")
    output.mkdir(parents=True)
    config = {"dataset": str(DATASET), "dataset_sha256": file_hash(DATASET),
              "source_checkpoints": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in sources.items()},
              "recursive_epoch_cap": epochs, "batch_size": 256, "threads": 2, "learning_rate": .0002,
              "shuffle_seed": 93, "validation_every": 1, "training_horizon_steps": 10,
              "validation_horizon_steps": 40, "contact_weight": 3, "gradient_norm_limit": 5,
              "initialization": "Existing 12-epoch one-step screen; selected epoch 11, no additional one-step training",
              "no_collection": True}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    models = {}
    for name, source in sources.items():
        print(f"Refining {name}: {epochs} epochs, full autoregressive validation", flush=True)
        timing = train_multistep(dataset, source, output / name, epochs,
                                shuffle_seed=93, validation_every=1)
        checkpoint = torch.load(output / name / "model.pt", map_location="cpu", weights_only=True)
        models[name] = {"checkpoint": str(output / name / "model.pt"), "timing": timing,
                        "selected_refinement_epoch": checkpoint["refinement_epoch"]}
        # Held-out evaluation only starts after this model's validation selection.
        directory = output / "evaluations" / name
        report = evaluate_boundary(dataset, output / name / "model.pt", directory)
        models[name]["evaluation"] = summarize_evaluation(dataset, directory, report)
        with np.load(SCREEN / "evaluations" / name / "test_rollouts.npz") as original, np.load(directory / "test_rollouts.npz") as refined:
            np.testing.assert_array_equal(original["actual"], refined["actual"])
            np.testing.assert_array_equal(original["actions"], refined["actions"])
        history = json.loads((output / name / "refinement_history.json").read_text())
        models[name]["history"] = history
        figure, axes = new_plot()
        axes.plot([row["epoch"] for row in history], [row["validation_recursive_scaled_mse"] for row in history], marker="o")
        axes.set(xlabel="Recursive epoch", ylabel="Scaled 4 s validation rollout MSE", title=name)
        figure.savefig(output / name / "validation_rollout.png")
    comparison = {name: {"one_step_only": screen["models"][name]["evaluation"], "refined": info["evaluation"]} for name, info in models.items()}
    result = {"config": config, "models": models, "comparison": comparison,
              "historical_128_refined": screen["models"]["baseline_recursive"]["evaluation"]}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    figure, axes = new_plot()
    for name, pair in comparison.items():
        for method, metrics in pair.items():
            axes.plot([1, 2, 4], [100 * metrics["horizon_endpoints"][str(sec)]["displacement_vector_rmse_m"] for sec in (1, 2, 4)], marker="o", linestyle="--" if method == "one_step_only" else "-", label=f"{name}: {method}")
    metrics = result["historical_128_refined"]
    axes.plot([1, 2, 4], [100 * metrics["horizon_endpoints"][str(sec)]["displacement_vector_rmse_m"] for sec in (1, 2, 4)], color="black", label="Historical 128 refined (longer training)")
    axes.set(xlabel="Autoregressive horizon (s)", ylabel="Cube XY endpoint RMSE (cm)")
    axes.legend(fontsize=7)
    figure.savefig(output / "endpoint_comparison.png")
    plot_behavior_comparison(result, output)
    for name, pair in comparison.items():
        print(name + " " + json.dumps({method: {"one_step_mm": 1000 * m["one_step"]["displacement_vector_rmse_m"],
              "endpoint_4s_cm": 100 * m["horizon_endpoints"]["4"]["displacement_vector_rmse_m"],
              "no_contact_drift_mm": 1000 * m["no_contact"]["mean_max_drift_m"],
              "clear_push_cm": 100 * m["boundary_clear_push"]["learned"]["horizon_endpoints"]["4"]["displacement_vector_rmse_m"]} for method, m in pair.items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, args.epochs)


if __name__ == "__main__":
    main()
