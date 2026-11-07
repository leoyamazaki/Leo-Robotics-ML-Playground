"""Base-relative residual world model: saved-data one-step training and refinement."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from .arm_learning import normalization
from .contact_boundary import load_boundary
from .contact_learning import differentiable_next, new_plot, training_arrays
from .contact_models import (INPUT_FLOOR, TARGET_FLOOR, STATE_SCALES,
                             RESIDUAL_ARCHITECTURE, model_from_checkpoint)
from .portable_training import cpu_tree, select_device


DATASET = Path("data/arm_cube_contact_boundary/mixed.npz")
OUTPUT = Path("runs/arm_cube_base_residual")
NORMALIZATION_KEYS = ("input_mean", "input_scale", "target_mean", "target_scale")


@dataclass(frozen=True)
class Config:
    one_step_epochs: int = 100
    recursive_epochs: int = 80
    batch_size: int = 256

    def __post_init__(self):
        for key, minimum in (("one_step_epochs", 1), ("recursive_epochs", 0), ("batch_size", 1)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def code_hashes():
    root = Path(__file__).resolve().parent
    return {path.name: file_hash(path) for path in sorted(root.glob("*.py"))}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved dataset: {path}. Copy the original data/ artifacts to this machine; this experiment collects no data.")
    dataset = load_boundary(path)
    if dataset["contacts"].shape != (len(dataset["states"]), 200, 10):
        raise ValueError("Expected saved 200-step contact trajectories with ten physics substeps")
    if dataset["metadata"].get("paired_groups") and "group_ids" not in dataset:
        raise ValueError("Paired dataset is missing group_ids; restore the original archive")
    return dataset


def fit_normalization(dataset):
    inputs, targets, _ = training_arrays(dataset, 5, 0, "base_relative")
    input_mean, input_scale = normalization(inputs)
    target_mean, target_scale = normalization(targets)
    return {"input_mean": input_mean, "input_scale": torch.maximum(input_scale, torch.tensor(INPUT_FLOOR)),
            "target_mean": target_mean, "target_scale": torch.maximum(target_scale, torch.tensor(TARGET_FLOOR))}


def one_step_arrays(dataset, checkpoint, split):
    x, y, weights = training_arrays(dataset, 5, split, "base_relative")
    return ((x - checkpoint["input_mean"]) / checkpoint["input_scale"],
            (y - checkpoint["target_mean"]) / checkpoint["target_scale"], weights)


def episode_arrays(dataset, split):
    selected = dataset["split"] == split
    states = torch.tensor(dataset["states"][selected, ::5], dtype=torch.float32)
    actions = torch.tensor(dataset["actions"][selected, ::5], dtype=torch.float32)
    contact = torch.tensor(dataset["contacts"][selected].reshape(len(states), 40, 50).any(-1))
    return states, actions, contact


def recursive_windows(episodes):
    return torch.cartesian_prod(torch.arange(episodes), torch.arange(31)).reshape(-1, 2)


def recursive_batch(arrays, windows, device):
    states, actions, contacts = arrays
    episode, start = windows.T
    offsets = start[:, None] + torch.arange(10)
    return (states[episode[:, None], start[:, None] + torch.arange(11)].to(device),
            actions[episode[:, None], offsets].to(device), contacts[episode[:, None], offsets].to(device))


def recursive_loss(model, states, actions, contacts, checkpoint):
    current = states[:, 0]
    scales = current.new_tensor(STATE_SCALES)
    loss = current.new_zeros(())
    for offset in range(actions.shape[1]):
        current = differentiable_next(model, current, actions[:, offset], checkpoint)
        error = ((current - states[:, offset + 1]) / scales).square().mean(-1)
        loss = loss + (error * (1 + 2 * contacts[:, offset].to(error.dtype))).mean() / actions.shape[1]
    return loss


def validation_score(model, arrays, checkpoint, stage, device, batch_size):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(arrays[0]), batch_size):
            if stage == "one_step":
                x, y = (item[start:start + batch_size].to(device) for item in arrays[:2])
                loss = (model(x) - y).square().mean()
                samples = len(x)
            else:
                states, actions = (item[start:start + batch_size].to(device) for item in arrays[:2])
                # Unweighted, full 40-step predicted-state feedback from measured initial state.
                contact = torch.zeros(actions.shape[:2], dtype=torch.bool, device=device)
                loss = recursive_loss(model, states, actions, contact, checkpoint)
                samples = len(states)
            total += loss.item() * samples
            count += samples
    score = total / count
    if not np.isfinite(score):
        raise RuntimeError(f"Nonfinite {stage} validation score; no checkpoint selected. Inspect rollout stability and normalization.")
    return score


def device_checkpoint(checkpoint, device):
    return {**checkpoint, **{key: checkpoint[key].to(device) for key in NORMALIZATION_KEYS}}


def atomic_save(value, path):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_stage(dataset, reference, output, stage, config, provenance, device, resume=None):
    """Save at epoch boundaries. Best weights travel inside last.pt for safe continuation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    learning_rate, seed = (.001, 91) if stage == "one_step" else (.0002, 93)
    torch.manual_seed(91)
    model = model_from_checkpoint(reference).to(device)
    if stage == "recursive":
        model.load_state_dict(reference["weights"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffle = torch.Generator().manual_seed(seed)
    start, best, best_checkpoint, history = 0, float("inf"), None, []
    if resume is not None:
        model.load_state_dict(resume["weights"])
        optimizer.load_state_dict(resume["optimizer"])
        shuffle.set_state(resume["shuffle_state"])
        start, best = resume["epoch"], resume["best_score"]
        best_checkpoint, history = resume["best_checkpoint"], list(resume["history"])
        atomic_save(best_checkpoint, output / "model.pt")
    runtime = device_checkpoint(reference, device)
    arrays_fn = one_step_arrays if stage == "one_step" else None
    train = arrays_fn(dataset, reference, 0) if arrays_fn else episode_arrays(dataset, 0)
    validation = arrays_fn(dataset, reference, 1) if arrays_fn else episode_arrays(dataset, 1)
    windows = recursive_windows(len(train[0])) if stage == "recursive" else None
    count = len(windows) if windows is not None else len(train[0])
    epochs = config.one_step_epochs if stage == "one_step" else config.recursive_epochs

    def snapshot(epoch, score):
        return {**{key: reference[key] for key in (*NORMALIZATION_KEYS, "metadata", "stride", "architecture", "input_coordinate_frame")},
                "weights": cpu_tree(model.state_dict()), "stage": stage, "epoch": epoch,
                "validation_score": score, "configuration": asdict(config), "provenance": provenance,
                "training_method": "base_relative_residual", "parameter_count": 4746252}

    def save(epoch, score, training, elapsed):
        nonlocal best, best_checkpoint
        current = snapshot(epoch, score)
        if score < best:
            best, best_checkpoint = score, current
            atomic_save(best_checkpoint, output / "model.pt")
        history.append({"epoch": epoch, "training_loss": training, "validation": score, "seconds": elapsed})
        last = dict(current, optimizer=cpu_tree(optimizer.state_dict()), shuffle_state=shuffle.get_state(),
                    best_score=best, best_checkpoint=best_checkpoint, history=history,
                    initialization_sha256=provenance.get("one_step_selected_sha256"))
        atomic_save(last, output / "last.pt")
        write_json(output / "history.json", history)

    if resume is not None and start == epochs:
        atomic_save(dict(resume, configuration=asdict(config), provenance=provenance), output / "last.pt")
        write_json(output / "history.json", history)
    batches = (count + config.batch_size - 1) // config.batch_size
    with tqdm(total=epochs * batches, initial=start * batches, desc=stage, leave=False,
              disable=not sys.stderr.isatty() or os.environ.get("TERM") == "dumb", dynamic_ncols=True, unit="batch") as progress:
        if stage == "recursive" and resume is None:
            progress.set_description(f"{stage} epoch 0/{epochs} validating")
            started = time.perf_counter()
            score = validation_score(model, validation, runtime, stage, device, config.batch_size)
            save(0, score, None, time.perf_counter() - started)
            progress.set_postfix(loss="n/a", val=f"{score:.5g}")
        for epoch in range(start + 1, epochs + 1):
            started, total = time.perf_counter(), 0.
            model.train()
            progress.set_description(f"{stage} epoch {epoch}/{epochs}")
            for batch, indices in enumerate(torch.randperm(count, generator=shuffle).split(config.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                if stage == "one_step":
                    x, y, weight = (item[indices].to(device) for item in train)
                    loss = ((model(x) - y).square().mean(-1) * weight).mean()
                else:
                    states, actions, contacts = recursive_batch(train, windows[indices], device)
                    loss = recursive_loss(model, states, actions, contacts, runtime)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {stage} training loss at epoch {epoch}, batch {batch}; last.pt retains the preceding complete epoch")
                loss.backward()
                if stage == "recursive":
                    try:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    except RuntimeError as error:
                        raise RuntimeError(f"Nonfinite recursive gradient at epoch {epoch}, batch {batch}; inspect last.pt") from error
                elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f"Nonfinite one_step gradient at epoch {epoch}, batch {batch}")
                optimizer.step()
                total += loss.item() * len(indices)
                progress.set_postfix(batch=f"{batch + 1}/{batches}", loss=f"{loss.item():.5g}",
                                     val=f"{history[-1]['validation']:.5g}" if history else "pending", refresh=False)
                progress.update(1)
            progress.set_description(f"{stage} epoch {epoch}/{epochs} validating")
            score = validation_score(model, validation, runtime, stage, device, config.batch_size)
            progress.set_postfix(batch=f"{batches}/{batches}", loss=f"{total / count:.5g}", val=f"{score:.5g}")
            save(epoch, score, total / count, time.perf_counter() - started)
    figure, axes = new_plot()
    axes.plot([row["epoch"] for row in history], [row["validation"] for row in history], label="Unweighted validation")
    axes.plot([row["epoch"] for row in history], [row["training_loss"] for row in history], label="Weighted training")
    axes.set(xlabel="Stage epoch", ylabel="Normalized displacement MSE" if stage == "one_step" else "State-scaled rollout MSE")
    axes.legend()
    figure.savefig(output / "loss.png")
    print(f"{stage} complete: epoch {epochs}, selected {best_checkpoint['epoch']}, validation {best:.7g}, "
          f"recorded epoch time {sum(row['seconds'] for row in history):.2f}s", flush=True)
    return output / "model.pt"


def verify_resume(saved, config, provenance, stage):
    if saved.get("training_method") != "base_relative_residual" or saved.get("stage") != stage or "optimizer" not in saved:
        raise ValueError(f"Resume requires {stage}/last.pt from this experiment")
    model_from_checkpoint(saved)  # rejects unsupported contracts
    if saved["architecture"] != RESIDUAL_ARCHITECTURE or saved["input_coordinate_frame"] != "base_relative":
        raise ValueError("Resume architecture/frame differs")
    for key in ("dataset_sha256", "code_hashes", "threads", "dtype", "init_seed",
                "one_step_shuffle_seed", "recursive_shuffle_seed", "one_step_learning_rate",
                "recursive_learning_rate", "state_scales", "gradient_norm_clip", "contact_weights",
                "window_starts", "normalization_sha256", "objectives", "base_world_xy"):
        if saved["provenance"].get(key) != provenance[key]:
            raise ValueError(f"Resume {key} differs; restore matching dataset and training code")
    if saved["configuration"]["batch_size"] != config.batch_size:
        raise ValueError("Resume batch_size differs")
    target = config.one_step_epochs if stage == "one_step" else config.recursive_epochs
    if target < saved["epoch"]:
        raise ValueError("Requested epoch target is less than completed resume epoch")
    if stage == "recursive" and saved["configuration"]["one_step_epochs"] != config.one_step_epochs:
        raise ValueError("Cannot extend one-step training after recursive refinement has begun")


def run(dataset_path=DATASET, output=OUTPUT, config=Config(), device="auto", resume=None,
        evaluate=True, historical=None):
    torch.set_num_threads(2)
    device = select_device(device)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Require a new output directory: {output}")
    dataset = load_data(dataset_path)
    if historical is not None and not Path(historical).is_file():
        raise FileNotFoundError(f"Missing historical checkpoint: {historical}; restore the original runs/ artifact")
    resume_root, normalization_source = None, None
    if resume:
        resume = Path(resume)
        if not resume.exists():
            raise FileNotFoundError(f"Missing resume artifact: {resume}; copy the complete run folder")
        if resume.is_file() and resume.name != "last.pt":
            raise ValueError("Resume requires a run directory or a stage last.pt, not a selected model.pt")
        resume_root = resume if resume.is_dir() else resume.parent.parent
        one_last = resume_root / "one_step/last.pt"
        if not one_last.is_file():
            raise FileNotFoundError(f"Missing resume artifact {one_last}; copy the complete run folder")
        normalization_source = torch.load(one_last, map_location="cpu", weights_only=True)
        # Retain the original fitted tensors across machines; CPU reductions can
        # differ slightly between processor architectures, so do not fit them again.
        norm = {key: normalization_source[key].clone() for key in NORMALIZATION_KEYS}
    else:
        norm = fit_normalization(dataset)
    provenance = {"dataset": str(dataset_path), "dataset_sha256": file_hash(dataset_path),
                  "code_hashes": code_hashes(), "device": str(device), "torch_version": str(torch.__version__),
                  "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                  "threads": 2, "dtype": "float32", "init_seed": 91, "one_step_shuffle_seed": 91,
                  "base_world_xy": [0., 0.],
                  "objectives": {"one_step": "mean normalized 12-component displacement MSE, weighted per transition",
                                 "recursive": "mean state-scaled MSE over ten predicted-feedback steps and batch, weighted per contact step",
                                 "one_step_selector": "lowest unweighted normalized one-step validation MSE after each epoch",
                                 "recursive_selector": "lowest unweighted 40-step validation state-scaled MSE, including epoch zero"},
                  "recursive_shuffle_seed": 93, "one_step_learning_rate": .001, "recursive_learning_rate": .0002,
                  "state_scales": STATE_SCALES, "gradient_norm_clip": 5, "contact_weights": {"one_step": 5, "recursive": 3},
                  "window_starts": list(range(31)), "normalization_split": "training episodes only (split == 0)",
                  "input_scale_floors": INPUT_FLOOR, "target_scale_floors": TARGET_FLOOR,
                  "normalization_sha256": {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in norm.items()},
                  "normalization_training_episodes": np.flatnonzero(dataset["split"] == 0).tolist(),
                  "split_episodes": np.bincount(dataset["split"]).tolist(), "no_collection": True,
                  "resume": str(resume) if resume else None}
    reference = dict(norm, stride=5, metadata=dataset["metadata"], architecture=RESIDUAL_ARCHITECTURE,
                     input_coordinate_frame="base_relative")
    resumes = {}
    if resume:
        # A one_step/last.pt explicitly requests restarting recursive training from that stage.
        stages = ("one_step",) if resume.is_file() and resume.parent.name == "one_step" else ("one_step", "recursive")
        for stage in stages:
            path = resume_root / stage / "last.pt"
            if not path.is_file():
                if stage == "one_step":
                    raise FileNotFoundError(f"Missing resume artifact {path}; copy the complete run folder")
                continue
            saved = normalization_source if stage == "one_step" else torch.load(path, map_location="cpu", weights_only=True)
            verify_resume(saved, config, provenance, stage)
            for key in NORMALIZATION_KEYS:
                torch.testing.assert_close(saved[key], norm[key], rtol=0, atol=0)
            resumes[stage] = saved
        if "recursive" in resumes and resumes["one_step"]["epoch"] != config.one_step_epochs:
            raise ValueError("Recursive resume requires the completed, unchanged one-step stage")
    output.mkdir(parents=True)
    write_json(output / "config.json", {"configuration": asdict(config), "provenance": provenance,
                                       "architecture": RESIDUAL_ARCHITECTURE, "input_coordinate_frame": "base_relative", "parameter_count": 4746252})
    started = time.perf_counter()
    one = train_stage(dataset, reference, output / "one_step", "one_step", config, provenance, device, resumes.get("one_step"))
    selected = torch.load(one, map_location="cpu", weights_only=True)
    provenance = dict(provenance, one_step_selected_sha256=file_hash(one))
    # Byte-level hashes can change when reserializing an unchanged selected checkpoint;
    # tensor equality is the compatibility contract for recursive initialization.
    if "recursive" in resumes:
        old = resumes["one_step"]["best_checkpoint"]
        for key, value in selected["weights"].items():
            torch.testing.assert_close(value, old["weights"][key], rtol=0, atol=0)
    recursive = train_stage(dataset, selected, output / "recursive", "recursive", config, provenance, device, resumes.get("recursive"))
    write_json(output / "timing.json", {"training_wall_seconds": time.perf_counter() - started})
    if evaluate:
        from .residual_evaluation import evaluate_run
        evaluate_run(dataset_path, output, output / "evaluation", historical=historical)
    return one, recursive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"), nargs="?", default="train")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--one-step-epochs", type=int, default=100)
    parser.add_argument("--recursive-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", type=Path, help="Previous run directory or stage last.pt; copy the full run folder")
    parser.add_argument("--run", type=Path, default=OUTPUT, help="Run to evaluate")
    parser.add_argument("--historical", type=Path, help="Optional historical reference, not an isolated architecture control")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            from .residual_evaluation import evaluate_run
            evaluate_run(args.dataset, args.run, args.output / "evaluation", historical=args.historical)
        else:
            run(args.dataset, args.output, Config(args.one_step_epochs, args.recursive_epochs, args.batch_size),
                args.device, args.resume, not args.skip_evaluation, args.historical)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
