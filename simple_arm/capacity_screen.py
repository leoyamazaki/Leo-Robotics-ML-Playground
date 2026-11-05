"""Capped one-step width screening on saved data; no simulation collection."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .contact_boundary import evaluate_boundary, load_boundary
from .contact_learning import load_predictor, make_model, new_plot, physical_errors, training_arrays


DATASET = Path("data/arm_cube_contact_boundary/mixed.npz")
BASELINE = Path("runs/arm_cube_contact_boundary")
OUTPUT = Path("runs/arm_cube_capacity_screen")
NORMALIZATION_KEYS = ("input_mean", "input_scale", "target_mean", "target_scale")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset, reference):
    if dataset["metadata"] != reference["metadata"] or reference["stride"] != 5:
        raise ValueError("Capacity screening requires the baseline's exact 100 ms dataset")
    train_x, train_y, train_weights = training_arrays(dataset, 5, 0)
    val_x, val_y, val_weights = training_arrays(dataset, 5, 1)
    torch.testing.assert_close(reference["input_mean"], train_x.mean(0))
    torch.testing.assert_close(reference["target_mean"], train_y.mean(0))

    def normalized(x, y, weights):
        return ((x - reference["input_mean"]) / reference["input_scale"],
                (y - reference["target_mean"]) / reference["target_scale"], weights)

    return normalized(train_x, train_y, train_weights), normalized(val_x, val_y, val_weights)


def normalized_losses(model, arrays):
    """Post-epoch train/validation losses, with both weightings explicit."""
    model.eval()
    x, y, weights = arrays
    plain, weighted = 0., 0.
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            error = (model(x[start:start + 4096]) - y[start:start + 4096]).square().mean(dim=-1)
            plain += error.sum().item()
            weighted += (error * weights[start:start + 4096]).sum().item()
    return {"unweighted": plain / len(x), "contact_weighted": weighted / len(x)}


def train_width(width, reference, train_arrays, val_arrays, output, epochs=12):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(91)
    model = make_model(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Model initialization consumes different random draws at each width.
    # A separate generator gives every model exactly the same batch orders.
    shuffle = torch.Generator().manual_seed(91)
    x, y, weights = train_arrays
    history, best, updates = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        online_loss = 0.
        for indices in torch.randperm(len(x), generator=shuffle).split(256):
            optimizer.zero_grad()
            loss = ((model(x[indices]) - y[indices]).square().mean(dim=-1) * weights[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite capacity-screen training loss")
            loss.backward()
            optimizer.step()
            online_loss += loss.item() * len(indices)
            updates += 1
        train_loss, val_loss = normalized_losses(model, train_arrays), normalized_losses(model, val_arrays)
        if not np.isfinite(val_loss["unweighted"]):
            raise RuntimeError("Nonfinite validation loss")
        if val_loss["unweighted"] < best:
            best = val_loss["unweighted"]
            saved = {key: reference[key] for key in (*NORMALIZATION_KEYS, "metadata", "stride")}
            saved.update(weights=model.state_dict(), hidden_width=width, epoch=epoch,
                         training_method="one_step_only_capacity_screen", init_seed=91, shuffle_seed=91,
                         epoch_cap=epochs, batch_size=256, optimizer="Adam", learning_rate=.001)
            torch.save(saved, output / "model.pt")
        history.append({"epoch": epoch, "updates": updates, "online_weighted_train": online_loss / len(x),
                        "train": train_loss, "validation": val_loss,
                        "epoch_seconds": time.perf_counter() - epoch_started})
        if epoch % 3 == 0 or epoch == epochs:
            print(f"width={width} epoch={epoch}/{epochs} train={train_loss['unweighted']:.5f} validation={val_loss['unweighted']:.5f} seconds={history[-1]['epoch_seconds']:.2f}", flush=True)
    total = time.perf_counter() - started
    selected = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    stats = {"hidden_width": width, "parameters": sum(p.numel() for p in model.parameters()),
             "method": "one_step_only", "epoch_cap": epochs, "epochs_run": epochs, "optimizer_updates": updates,
             "selected_epoch": selected["epoch"], "training_wall_seconds": total,
             "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
             "median_epoch_seconds": float(np.median([h["epoch_seconds"] for h in history])),
             "history": history}
    (output / "training.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    figure, axes = new_plot()
    axes.semilogy(range(1, epochs + 1), [h["train"]["unweighted"] for h in history], label="Train, unweighted")
    axes.semilogy(range(1, epochs + 1), [h["validation"]["unweighted"] for h in history], label="Validation, unweighted")
    axes.semilogy(range(1, epochs + 1), [h["online_weighted_train"] for h in history], "--", label="Training objective, 5x contact")
    axes.set(xlabel="Epoch", ylabel="Normalized residual MSE", title=f"Width {width}: one-step screening")
    axes.legend()
    figure.savefig(output / "loss.png")
    return stats


def summarize_evaluation(dataset, directory, report):
    selected = dataset["split"] == 2
    with np.load(Path(directory) / "test_rollouts.npz") as archive:
        actual, predicted = archive["actual"], archive["learned"]
        motion = np.linalg.norm(actual[..., 4:6] - actual[:, :1, 4:6], axis=-1).max(axis=1)
        no_contact = ~dataset["contacts"][selected].any(axis=(1, 2)) & (motion <= .002)
        drift = np.linalg.norm(predicted[..., 4:6] - actual[:, :1, 4:6], axis=-1).max(axis=1)[no_contact]
        broad_push = (dataset["source"][selected] == 0) & (np.linalg.norm(actual[:, -1, 4:6] - actual[:, 0, 4:6], axis=-1) > .02)
        endpoints = {str(sec): physical_errors(predicted[:, sec*10], actual[:, sec*10], actual[:, 0]) for sec in (1, 2, 4)}
        result = {"test_episodes": len(actual), "one_step": report["one_step"]["learned"],
                  "horizon_endpoints": endpoints, "prefix_averaged_rollout": report["rollout"]["learned"],
                  "no_contact": {"episodes": int(no_contact.sum()), "mean_max_drift_m": float(drift.mean()),
                                 "p95_max_drift_m": float(np.quantile(drift, .95)), "fraction_over_2mm": float((drift > .002).mean())},
                  "broad_push_endpoint": physical_errors(predicted[broad_push, -1], actual[broad_push, -1], actual[broad_push, 0]),
                  "boundary_clear_push": report["category_metrics"]["boundary_clear_push"],
                  "boundary_overhead": report["category_metrics"]["boundary_overhead"],
                  "boundary_near_miss": report["category_metrics"]["boundary_near_miss"]}
    (Path(directory) / "screen_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def run(dataset_path=DATASET, output=OUTPUT, epochs=12):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing benchmark: {output}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(2)
    dataset = load_boundary(dataset_path)
    reference = torch.load(BASELINE / "firstpass/model.pt", map_location="cpu", weights_only=True)
    train_arrays, val_arrays = prepare(dataset, reference)
    if len(dataset["states"]) * 40 != 99200 or np.bincount(dataset["split"]).tolist() != [1736, 372, 372]:
        raise ValueError("Expected the original 99,200-transition mixed dataset and splits")
    output.mkdir(parents=True)
    config = {"dataset": str(dataset_path), "dataset_sha256": file_hash(dataset_path),
              "dataset_metadata": dataset["metadata"], "split_episodes": [1736, 372, 372],
              "widths": [128, 256, 512], "epoch_cap": epochs, "updates_per_epoch": (len(train_arrays[0]) + 255) // 256,
              "total_updates_per_model": epochs * ((len(train_arrays[0]) + 255) // 256),
              "batch_size": 256, "threads": 2, "init_seed": 91, "shuffle_seed": 91,
              "budget_definition": "Identical epochs, samples and Adam updates; wider models consume more FLOPs and wall time",
              "normalization_checkpoint": str(BASELINE / "firstpass/model.pt"),
              "baseline_sha256": {name: file_hash(BASELINE / name / "model.pt") for name in ("firstpass", "recursive")},
              "no_collection": True, "no_recursive_training": True}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = {}
    for width, name in ((128, "short_128"), (256, "medium_256"), (512, "large_512")):
        stats = train_width(width, reference, train_arrays, val_arrays, output / name, epochs)
        runs[name] = {"checkpoint": str(output / name / "model.pt"), "training": stats}
    for name, directory, method in (("baseline_one_step", "firstpass", "historical one-step; 150 epochs, selected 136"),
                                     ("baseline_recursive", "recursive", "historical one-step + 80 recursive epochs, selected refinement 25")):
        checkpoint = BASELINE / directory / "model.pt"
        runs[name] = {"checkpoint": str(checkpoint), "training": {"hidden_width": 128, "parameters": 36620,
                      "method": method, "training_wall_seconds": None, "mean_epoch_seconds": None,
                      "timing_note": "Historical timing was not recorded; baseline is not retrained"}}
    summary = {"config": config, "models": runs}
    for name, info in runs.items():
        checkpoint = torch.load(info["checkpoint"], map_location="cpu", weights_only=True)
        model = make_model(checkpoint.get("hidden_width", 128)).eval()
        model.load_state_dict(checkpoint["weights"])
        info["selected_checkpoint_losses"] = {"train": normalized_losses(model, train_arrays), "validation": normalized_losses(model, val_arrays)}
        directory = output / "evaluations" / name
        report = evaluate_boundary(dataset, info["checkpoint"], directory)
        info["evaluation"] = summarize_evaluation(dataset, directory, report)
        print(f"Evaluated {name}: one-step XY RMSE={info['evaluation']['one_step']['displacement_vector_rmse_m']:.5f} m, 4s endpoint={info['evaluation']['horizon_endpoints']['4']['displacement_vector_rmse_m']:.5f} m", flush=True)
    # This shared-axis plot compares endpoint errors, not prefix averages.
    figure, axes = new_plot()
    for name, info in runs.items():
        axes.plot([1, 2, 4], [100 * info["evaluation"]["horizon_endpoints"][str(sec)]["displacement_vector_rmse_m"] for sec in (1, 2, 4)], marker="o", label=name)
    axes.set(xlabel="Autoregressive prediction horizon (s)", ylabel="Cube XY endpoint RMSE (cm)")
    axes.legend(fontsize=8)
    figure.savefig(output / "endpoint_comparison.png")
    figure, axes = new_plot()
    names = list(runs)
    axes.bar(range(len(names)), [1000 * runs[name]["evaluation"]["no_contact"]["mean_max_drift_m"] for name in names])
    axes.set_xticks(range(len(names)), names, rotation=20)
    axes.set(ylabel="Mean maximum no-contact drift (mm)")
    figure.savefig(output / "drift_comparison.png")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Capacity screen complete: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    run(args.dataset, args.output, args.epochs)


if __name__ == "__main__":
    main()
