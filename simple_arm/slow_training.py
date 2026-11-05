"""Collect 0.2 rad/s slow interactions and refine the frozen stationary-20 model."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .contact_boundary import load_boundary, save_dataset
from .contact_learning import new_plot
from .cube_mpc_experiment import file_hash, source_hashes, write_json
from .slow_collection import collect
from .stationary_experiment import INDEPENDENT, evaluate
from .stationary_training import TrainingConfig, train


SOURCE = Path("runs/arm_cube_stationary_experiment/stationary_20/model.pt")
ORIGINAL = Path("data/arm_cube_contact_boundary/mixed.npz")
OUTPUT = Path("runs/arm_cube_slow_02_expanded")


def replay_mix(slow, original, seed=95):
    """Equal new-slow/replay episode counts per training and validation split."""
    rng = np.random.default_rng(seed)
    sources, indices, splits = [], [], []
    for split in (0, 1):
        slow_ids = np.flatnonzero(slow["split"] == split)
        old_ids = rng.choice(np.flatnonzero(original["split"] == split), len(slow_ids), replace=False)
        sources.extend([0] * len(slow_ids) + [1] * len(old_ids))
        indices.extend(slow_ids.tolist() + old_ids.tolist())
        splits.extend([split] * (2 * len(slow_ids)))
    # Preserve new-data test episodes, never use old test data in fine tuning.
    test = np.flatnonzero(slow["split"] == 2)
    sources.extend([0] * len(test))
    indices.extend(test.tolist())
    splits.extend([2] * len(test))
    sources, indices = np.asarray(sources), np.asarray(indices)
    parts = [slow, original]
    result = {key: np.stack([parts[src][key][idx] for src, idx in zip(sources, indices)]) for key in (
        "states", "actions", "contacts", "starting_contact", "kinds")}
    result.update(split=np.asarray(splits), source=sources, source_episode=indices,
                  group_ids=np.array([parts[src]["group_ids"][idx] + src * 1000000 for src, idx in zip(sources, indices)]))
    metadata = dict(original["metadata"], seed=slow["metadata"]["seed"], episodes=len(indices),
                    collector="slow_with_replay_v1", kinds=["source_case_0", "source_case_1", "source_case_2", "source_case_3"],
                    parents=[original["metadata"], *original["metadata"].get("parents", []), slow["metadata"]],
                    slow_rate_limit_rad_s=slow["metadata"]["command_rate_limit_rad_s"],
                    replay_fraction_train=.5, replay_fraction_validation=.5)
    metadata.pop("data_sha256", None)
    result["metadata"] = metadata
    return result


def plot_results(models, output):
    for dataset in ("slow_independent", "boundary", "broad"):
        figure, axes = new_plot()
        for name in models:
            with np.load(output / "evaluations" / dataset / name / "test_rollouts.npz") as saved:
                error = saved["learned"][:, 1:, 4:6] - saved["actual"][:, 1:, 4:6]
                rmse = np.sqrt(np.mean(np.sum(error**2, axis=-1), axis=0))
            axes.plot(np.arange(1, 41) * .1, rmse * 1000, label=name)
        axes.set(xlabel="Recursive horizon (s)", ylabel="Cube XY vector RMSE (mm)", title=dataset)
        axes.legend()
        figure.savefig(output / f"{dataset}_rollout.png")


def run(output=OUTPUT, source=SOURCE, rate=.2, groups=320, independent_groups=40, workers=4, epochs=10):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    config = TrainingConfig(epochs=epochs, seed=95, learning_rate=.0001)
    torch.set_num_threads(2)
    before = source_hashes(source, ORIGINAL)
    original = load_boundary(ORIGINAL)
    checkpoint = torch.load(source, weights_only=True, map_location="cpu")
    if checkpoint.get("hidden_width") != 256 or checkpoint["metadata"]["model_hash"] != original["metadata"]["model_hash"]:
        raise ValueError("Expected compatible Medium source")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"source": str(source), "source_hashes_before": before, "rate_limit_rad_s": rate,
               "training_groups": groups, "independent_groups": independent_groups, "training": asdict(config),
               "horizon_steps": 20, "collection_seeds": [394, 395], "replay_fraction": .5,
               "actuators": "Unchanged original position actuators; commanded rate bound, actual joint speeds measured"})
    slow, collection = collect(output / "slow_train.npz", groups, 394, rate, workers)
    independent, independent_collection = collect(output / "slow_independent.npz", independent_groups, 395, rate, workers)
    mixed = replay_mix(slow, original)
    save_dataset(output / "training_mix.npz", mixed, {"split_episodes": np.bincount(mixed["split"]).tolist(),
                 "source_labels": ["slow", "original_replay"], "normalization": "Frozen from source checkpoint"})
    # This new initialization artifact has exactly the source weights and
    # normalization. Its metadata describes the forthcoming fine-tuning corpus;
    # preserve explicit weight provenance so it cannot be confused with training.
    initialization = dict(checkpoint, metadata=mixed["metadata"], initialization_only=True,
                          source_checkpoint=str(source), source_checkpoint_sha256=file_hash(source),
                          source_metadata=checkpoint["metadata"])
    torch.save(initialization, output / "initialization.pt")
    stats = train(mixed, output / "initialization.pt", output / "refined", 20, config.stationary_weight, config)
    # Remove the initialization-only flag from newly trained artifacts.
    for filename in ("model.pt", "last.pt"):
        path = output / "refined" / filename
        saved = torch.load(path, weights_only=True, map_location="cpu")
        saved["initialization_only"] = False
        saved["training_method"] = "slow_rate_02_stationary_recursive"
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            if not torch.equal(saved[key], checkpoint[key]):
                raise RuntimeError("Normalization changed")
        torch.save(saved, path)
    models = {"baseline": {"checkpoint": str(source)}, "slow_refined": {"checkpoint": str(output / "refined/model.pt")}}
    datasets = {"slow_independent": independent, **{name: load_boundary(path) for name, path in INDEPENDENT.items()}}
    for name, info in models.items():
        info["evaluation"] = {d: evaluate(data, info["checkpoint"], output / "evaluations" / d / name) for d, data in datasets.items()}
        for d, result in info["evaluation"].items():
            print(f"{name}/{d}: one_step={result['one_step_overall_rmse_m']:.5f} "
                  f"drift={result['stationary_drift']['stationary_no_contact']['autoregressive']} push={result['push_4s']}", flush=True)
    summary = {"models": models, "training": stats, "collection": collection, "independent_collection": independent_collection,
               "source_hashes_after": {p: file_hash(p) for p in before},
               "new_dataset_hashes": {name: file_hash(output / name) for name in ("slow_train.npz", "slow_independent.npz", "training_mix.npz")}}
    summary["sources_unchanged"] = before == summary["source_hashes_after"]
    write_json(output / "summary.json", summary)
    plot_results(models, output)
    if not summary["sources_unchanged"]:
        raise RuntimeError("Existing datasets/checkpoints changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--rate", type=float, default=.2)
    parser.add_argument("--groups", type=int, default=320)
    parser.add_argument("--independent-groups", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run(args.output, args.source, args.rate, args.groups, args.independent_groups, args.workers, args.epochs)


if __name__ == "__main__":
    main()
