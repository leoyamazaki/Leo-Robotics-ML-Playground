"""Fresh random physics + bounded replay, unweighted residual-model training."""

import argparse
from dataclasses import asdict, dataclass
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from .contact_models import RESIDUAL_ARCHITECTURE, STATE_SCALES, model_from_checkpoint
from .online_data import EpisodeStream, ReplayBuffer, dataset_from_rows, load_archive, metadata, save_archive
from .one_step_stationary import stationary_intervals
from .portable_training import cpu_tree, select_device
from .residual_evaluation import subset
from .residual_training import (NORMALIZATION_KEYS, atomic_save, code_hashes, device_checkpoint,
                               episode_arrays, file_hash, fit_normalization, load_data, new_plot,
                               one_step_arrays, recursive_batch, recursive_loss, validation_score, write_json)


OUTPUT = Path("runs/arm_cube_online")
BOUNDARY = Path("data/arm_cube_contact_boundary/focused.npz")


@dataclass(frozen=True)
class Config:
    one_step_epochs: int = 20
    recursive_epochs: int = 20
    one_step_new: int = 1000
    recursive_new: int = 250
    one_step_updates: int = 256
    recursive_updates: int = 64
    capacity: int = 5000
    batch_size: int = 256
    boundary_fraction: float = .1
    heldout_episodes: int = 120
    seed: int = 791

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "boundary_fraction":
                if not 0 <= value < 1:
                    raise ValueError("boundary_fraction must be in [0,1)")
            elif type(value) is not int or value < (0 if key in ("recursive_epochs", "seed") else 1):
                raise ValueError(f"Invalid {key}")
        if self.capacity < self.one_step_new or self.capacity < self.recursive_new:
            raise ValueError("Replay capacity must fit a complete collection chunk")


def status_write(output, value):
    tmp = Path(output) / "status.tmp.json"
    write_json(tmp, value)
    tmp.replace(Path(output) / "status.json")
    # Local-file dashboard: refresh the document itself; no server or fetch needed.
    fields = "".join(f"<tr><th>{escape(key.replace('_', ' '))}</th><td>{escape(str(item))}</td></tr>"
                     for key, item in value.items() if not isinstance(item, dict))
    document = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
                '<title>Robot training status</title><style>body{background:#101820;color:#edf5fa;font:18px system-ui;'
                'max-width:1000px;margin:3rem auto}h1{color:#65dcc2}table{width:100%;border-collapse:collapse}'
                'td,th{text-align:left;padding:.6rem;border-bottom:1px solid #33414c}th{width:45%}'
                'small{color:#a8bbc8}</style><h1>Robot training status</h1>'
                '<small>Refreshes every five seconds. Detailed metrics and checkpoints are saved alongside this page.</small>'
                f'<table>{fields}</table>')
    page = Path(output) / "status.tmp.html"
    page.write_text(document, encoding="utf-8")
    page.replace(Path(output) / "status.html")


def curves(path, history, stage):
    figure, axes = new_plot()
    axes.plot([r["epoch"] for r in history], [r["validation"] for r in history], label="Fixed validation")
    axes.plot([r["epoch"] for r in history], [r["training_loss"] for r in history], label="Unweighted training")
    axes.set(xlabel="Collection/training epoch", ylabel="Normalized delta MSE" if stage == "one_step" else "State-scaled rollout MSE")
    axes.legend()
    figure.savefig(path)


def one_step_metrics(model, dataset, reference, device, batch_size):
    arrays = one_step_arrays(dataset, reference, 0)
    stationary = stationary_intervals(dataset["states"]).reshape(-1)
    contacts = dataset["contacts"].reshape(len(dataset["states"]), 40, 50).any(-1).reshape(-1)
    masks = {"stationary": stationary, "moving": ~stationary, "contact": contacts}
    sums = {name: 0. for name in masks}
    mse = 0.
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays[0]), batch_size):
            x, y = (a[start:start + batch_size].to(device) for a in arrays[:2])
            error = (model(x) - y).square()
            mse += error.mean(-1).sum().item()
            xy = (error[:, 4:6] * reference["target_scale"][4:6].to(device).square()).sum(-1).cpu().numpy()
            for name, mask in masks.items():
                sums[name] += float(xy[mask[start:start + len(x)]].sum())
    return {"one_step_mse": mse / len(arrays[0]),
            **{name + "_xy_rmse_mm": float(np.sqrt(sums[name] / mask.sum()) * 1000) if mask.any() else None for name, mask in masks.items()},
            "stationary_fraction": float(stationary.mean()), "contact_fraction": float(contacts.mean())}


def validate(model, datasets, reference, runtime, stage, device, config):
    details = {}
    for name, dataset in datasets.items():
        details[name] = one_step_metrics(model, dataset, reference, device, config.batch_size)
        if stage == "recursive":
            arrays = episode_arrays(dataset, 0)
            details[name]["rollout_mse"] = validation_score(model, arrays, runtime, stage, device, config.batch_size)
    key = "one_step_mse" if stage == "one_step" else "rollout_mse"
    score = ((1 - config.boundary_fraction) * details["random"][key]
             + config.boundary_fraction * details["boundary"][key])
    if not np.isfinite(score):
        raise RuntimeError(f"Nonfinite {stage} validation; preceding last.pt is retained")
    return score, details


def coverage(dataset):
    contact = dataset["contacts"].reshape(len(dataset["states"]), 40, 50).any(-1)
    motion = np.linalg.norm(np.diff(dataset["states"][:, ::5, 4:6], axis=1), axis=-1)
    return {"episodes": len(dataset["states"]), "contact_interval_fraction": float(contact.mean()),
            "stationary_interval_fraction": float(stationary_intervals(dataset["states"]).mean()),
            "motion_over_2mm_fraction": float((motion > .002).mean()),
            "families": np.bincount(dataset["kinds"], minlength=6).tolist(),
            "peak_saved_joint_speed_rad_s": float(np.abs(dataset["states"][..., 2:4]).max())}


def sample_batch(random_arrays, boundary_arrays, config, generator, recursive, device):
    boundary_count = int((torch.rand(config.batch_size, generator=generator) < config.boundary_fraction).sum())
    pieces = []
    for arrays, count in ((random_arrays, config.batch_size - boundary_count), (boundary_arrays, boundary_count)):
        indices = torch.randint(len(arrays[0]), (count,), generator=generator)
        if recursive:
            starts = torch.randint(31, (count,), generator=generator)
            piece = recursive_batch(arrays, torch.stack([indices, starts], -1), device)
        else:
            piece = tuple(a[indices].to(device) for a in arrays[:2])
        pieces.append(piece)
    return tuple(torch.cat([a, b], 0) for a, b in zip(*pieces)), boundary_count


def online_loss(model, values, reference, recursive):
    if recursive:
        states, actions, contacts = values
        return recursive_loss(model, states, actions, torch.zeros_like(contacts), reference)
    x, y = values
    return (model(x) - y).square().mean()


def run(output=OUTPUT, config=Config(), device="auto", workers=4, boundary=BOUNDARY,
        resume=None, evaluate=True):
    torch.set_num_threads(2)
    device, output = select_device(device), Path(output)
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists():
        raise FileExistsError(f"Require a new output directory: {output}")
    boundary_data = load_data(boundary)
    if boundary_data["metadata"]["seed"] in (config.seed, config.seed + 1, config.seed + 2):
        raise ValueError("Random training/validation/test seeds must differ from the boundary collection seed")
    if boundary_data["metadata"].get("command_rate_limit_rad_s") != .2:
        raise ValueError("Boundary supplement must use the 0.2 rad/s command protocol")
    boundary_train = subset(boundary_data, boundary_data["split"] == 0)
    boundary_validation = subset(boundary_data, boundary_data["split"] == 1)
    boundary_validation["split"][:] = 0  # Local view only; disjoint source IDs verified below.
    code = code_hashes()
    provenance = {"code_hashes": code, "boundary_sha256": file_hash(boundary), "device": str(device),
                  "torch_version": str(torch.__version__), "threads": 2, "dtype": "float32",
                  "initialization_seed": 91, "one_step_shuffle_seed": 91, "recursive_shuffle_seed": 93,
                  "one_step_learning_rate": .001, "recursive_learning_rate": .0002,
                  "state_scales": STATE_SCALES, "recursive_gradient_clip": 5,
                  "loss": "Unweighted in both stages; no contact or stationary multipliers",
                  "seed_training": config.seed, "seed_validation": config.seed + 1, "seed_test": config.seed + 2,
                  "selection": "Fixed validation only; weighted mixture of unweighted random/boundary validation scores",
                  "boundary_training_source_episodes": np.flatnonzero(boundary_data["split"] == 0).tolist(),
                  "boundary_validation_source_episodes": np.flatnonzero(boundary_data["split"] == 1).tolist(),
                  "boundary_test_source_episodes": np.flatnonzero(boundary_data["split"] == 2).tolist()}
    previous, previous_root = {}, None
    if resume:
        resume = Path(resume)
        if not resume.exists() or (resume.is_file() and resume.name != "last.pt"):
            raise ValueError("Resume requires an existing run directory or stage last.pt")
        previous_root = resume if resume.is_dir() else resume.parent.parent
        names = ("one_step",) if resume.is_file() and resume.parent.name == "one_step" else ("one_step", "recursive")
        for stage in names:
            path = previous_root / stage / "last.pt"
            if not path.is_file():
                if stage == "one_step":
                    raise FileNotFoundError(f"Missing continuation state: {path}")
                continue
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("training_method") != "online_uniform_residual" or saved.get("stage") != stage:
                raise ValueError("Resume requires an online-training stage last.pt")
            if saved.get("architecture") != RESIDUAL_ARCHITECTURE or saved.get("input_coordinate_frame") != "base_relative":
                raise ValueError("Resume architecture/coordinate frame differs")
            if saved["provenance"].get("loss") != provenance["loss"]:
                raise ValueError("Resume loss definition differs")
            for key, value in asdict(config).items():
                if key not in ("one_step_epochs", "recursive_epochs") and saved["configuration"][key] != value:
                    raise ValueError(f"Resume configuration differs: {key}")
            if saved["provenance"]["code_hashes"] != code or saved["provenance"]["boundary_sha256"] != provenance["boundary_sha256"]:
                raise ValueError("Resume source code or boundary supplement differs")
            if saved["epoch"] > getattr(config, stage + "_epochs"):
                raise ValueError("Cannot reduce completed epoch count")
            previous[stage] = saved
        if "recursive" in previous and previous["recursive"]["configuration"]["one_step_epochs"] != config.one_step_epochs:
            raise ValueError("Recursive continuation must preserve the one-step budget")
    output.mkdir(parents=True)
    started = time.perf_counter()
    status_write(output, {"stage": "initializing", "device": str(device), "configuration": asdict(config)})
    data_dir = output / "data"
    data_dir.mkdir()
    cache = []
    if previous:
        final = previous.get("recursive", previous["one_step"])
        for record in final["archives"]:
            src = previous_root / record["path"]
            if file_hash(src) != record["sha256"]:
                raise ValueError(f"Replay archive hash differs: {src}")
            shutil.copy2(src, output / record["path"])
            cache.append(record)
        for name in ("validation.npz", "test.npz"):
            source = previous_root / "data" / name
            if file_hash(source) != final["heldout_hashes"][name]:
                raise ValueError("Fixed held-out data changed")
            shutil.copy2(source, data_dir / name)
        reference = {key: previous["one_step"][key] for key in (*NORMALIZATION_KEYS, "metadata", "stride", "architecture", "input_coordinate_frame")}
        for key in ("normalization_random_episode_ids", "normalization_boundary_source_episodes"):
            provenance[key] = previous["one_step"]["provenance"][key]
        generated = final["generated_episodes"]
    else:
        for name, seed, split in (("validation.npz", config.seed + 1, 1), ("test.npz", config.seed + 2, 2)):
            with EpisodeStream(seed, 0, config.heldout_episodes, workers, config.heldout_episodes) as stream:
                save_archive(data_dir / name, dataset_from_rows(stream.take(config.heldout_episodes), seed, split))
        generated, reference = 0, None
    heldout_hashes = {name: file_hash(data_dir / name) for name in ("validation.npz", "test.npz")}
    if previous:
        write_json(output / "config.json", {"configuration": asdict(config), "provenance": provenance,
                                           "planned_unique_episodes": config.one_step_epochs * config.one_step_new + config.recursive_epochs * config.recursive_new})
    validation_random = load_archive(data_dir / "validation.npz")
    validation_random["split"][:] = 0
    valsets = {"random": validation_random, "boundary": boundary_validation}
    replay = ReplayBuffer(config.capacity)
    for record in cache:
        replay.append(load_archive(output / record["path"]))
    total_episodes = config.one_step_epochs * config.one_step_new + config.recursive_epochs * config.recursive_new
    if generated > total_episodes:
        raise ValueError("Resume generation budget is smaller than already generated data")
    stage_results = {}
    with EpisodeStream(config.seed, generated, total_episodes, workers, max(config.one_step_new, config.recursive_new)) as stream:
        for stage in ("one_step", "recursive"):
            stage_dir = output / stage
            stage_dir.mkdir()
            recursive = stage == "recursive"
            old = previous.get(stage)
            epochs = getattr(config, stage + "_epochs")
            updates = getattr(config, stage + "_updates")
            fresh_count = getattr(config, stage + "_new")
            shuffle = torch.Generator().manual_seed(93 if recursive else 91)
            history, best, best_checkpoint, start = [], float("inf"), None, 0
            model, optimizer = None, None
            if old:
                reference = {key: old[key] for key in (*NORMALIZATION_KEYS, "metadata", "stride", "architecture", "input_coordinate_frame")}
            elif recursive:
                selected = torch.load(output / "one_step/model.pt", map_location="cpu", weights_only=True)
                reference = {**reference, "weights": selected["weights"]}
            if reference is not None:
                torch.manual_seed(91)
                model = model_from_checkpoint(reference).to(device)
                if recursive or old:
                    model.load_state_dict(old["weights"] if old else reference["weights"])
                optimizer = torch.optim.Adam(model.parameters(), lr=.0002 if recursive else .001)
            if old:
                optimizer.load_state_dict(old["optimizer"])
                shuffle.set_state(old["shuffle_state"])
                history, best, best_checkpoint, start = list(old["history"]), old["best_score"], old["best_checkpoint"], old["epoch"]
                atomic_save(best_checkpoint, stage_dir / "model.pt")
                if start == epochs:
                    atomic_save(dict(old, configuration=asdict(config), provenance=provenance), stage_dir / "last.pt")
            batches_done = start * updates

            def save(epoch, score, details, loss, seconds, distribution, supplement_fraction):
                nonlocal best, best_checkpoint
                current = {**{key: reference[key] for key in (*NORMALIZATION_KEYS, "metadata", "stride", "architecture", "input_coordinate_frame")},
                           "weights": cpu_tree(model.state_dict()), "stage": stage, "epoch": epoch,
                           "validation_score": score, "configuration": asdict(config), "provenance": provenance,
                           "training_method": "online_uniform_residual", "parameter_count": 4746252,
                           "generated_episodes": generated}
                if score < best:
                    best, best_checkpoint = score, current
                    atomic_save(current, stage_dir / "model.pt")
                history.append({"epoch": epoch, "validation": score, "validation_strata": details,
                                "training_loss": loss, "seconds": seconds, "fresh_episodes_total": generated,
                                "replay_distribution": distribution, "actual_boundary_sample_fraction": supplement_fraction})
                atomic_save(dict(current, optimizer=cpu_tree(optimizer.state_dict()), shuffle_state=shuffle.get_state(),
                                 best_score=best, best_checkpoint=best_checkpoint, history=history,
                                 archives=list(cache), heldout_hashes=heldout_hashes), stage_dir / "last.pt")
                write_json(stage_dir / "history.json", history)
                curves(stage_dir / "loss.png", history, stage)

            with tqdm(total=epochs * updates, initial=batches_done, desc=stage, leave=False,
                      disable=not sys.stderr.isatty() or os.environ.get("TERM") == "dumb", dynamic_ncols=True, unit="batch") as bar:
                if recursive and not old:
                    t = time.perf_counter()
                    runtime = device_checkpoint(reference, device)
                    score, details = validate(model, valsets, reference, runtime, stage, device, config)
                    save(0, score, details, None, time.perf_counter() - t, coverage(replay.data), None)
                for epoch in range(start + 1, epochs + 1):
                    t = time.perf_counter()
                    status_write(output, {"stage": stage, "phase": "collecting", "epoch": epoch, "epochs": epochs,
                                          "generated_episodes": generated, "target_episodes": total_episodes, "device": str(device)})
                    bar.set_description(f"{stage} epoch {epoch}/{epochs} collecting")
                    dataset = dataset_from_rows(stream.take(fresh_count), config.seed)
                    generated += fresh_count
                    path = data_dir / f"{stage}_{epoch:04}.npz"
                    save_archive(path, dataset)
                    cache.append({"path": path.relative_to(output).as_posix(), "sha256": file_hash(path), "episodes": fresh_count})
                    replay.append(dataset)
                    if reference is None:
                        # Fit once on initial random training episodes plus ~10% boundary examples.
                        ids = torch.randperm(len(boundary_train["states"]), generator=torch.Generator().manual_seed(91))[:min(len(boundary_train["states"]), round(fresh_count * config.boundary_fraction / (1 - config.boundary_fraction)))].numpy()
                        norm_data = dict(dataset, **{key: np.concatenate([dataset[key], boundary_train[key][ids]]) for key in ("states", "actions", "contacts", "starting_contact", "kinds", "split")})
                        provenance["normalization_random_episode_ids"] = dataset["episode_ids"].tolist()
                        provenance["normalization_boundary_source_episodes"] = np.flatnonzero(boundary_data["split"] == 0)[ids].tolist()
                        reference = dict(fit_normalization(norm_data), metadata=dict(metadata(config.seed, total_episodes), parents=[boundary_data["metadata"]]),
                                         stride=5, architecture=RESIDUAL_ARCHITECTURE, input_coordinate_frame="base_relative")
                        torch.manual_seed(91)
                        model = model_from_checkpoint(reference).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                        write_json(output / "config.json", {"configuration": asdict(config), "provenance": provenance,
                                                           "planned_unique_episodes": total_episodes, "planned_unique_transitions": total_episodes * 40})
                    runtime = device_checkpoint(reference, device)
                    random_arrays = episode_arrays(replay.data, 0) if recursive else one_step_arrays(replay.data, reference, 0)
                    boundary_arrays = episode_arrays(boundary_train, 0) if recursive else one_step_arrays(boundary_train, reference, 0)
                    distribution = coverage(replay.data)
                    model.train()
                    total_loss, boundary_samples, last_status = 0., 0, 0.
                    bar.set_description(f"{stage} epoch {epoch}/{epochs}")
                    for batch in range(updates):
                        values, number_boundary = sample_batch(random_arrays, boundary_arrays, config, shuffle, recursive, device)
                        boundary_samples += number_boundary
                        optimizer.zero_grad(set_to_none=True)
                        loss = online_loss(model, values, runtime, recursive)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite {stage} loss at epoch {epoch}, batch {batch + 1}")
                        loss.backward()
                        if recursive:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                        elif not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                            raise RuntimeError(f"Nonfinite one-step gradient at epoch {epoch}, batch {batch + 1}")
                        optimizer.step()
                        total_loss += loss.item()
                        batches_done += 1
                        latest = history[-1]["validation"] if history else None
                        bar.set_postfix(batch=f"{batch + 1}/{updates}", loss=f"{loss.item():.5g}", val=f"{latest:.5g}" if latest is not None else "pending", refresh=False)
                        bar.update(1)
                        now = time.perf_counter()
                        if now - last_status >= 5 or batch + 1 == updates:
                            elapsed = now - t
                            status_write(output, {"stage": stage, "phase": "training", "epoch": epoch, "epochs": epochs,
                                                  "batch": batch + 1, "batches_per_epoch": updates, "loss": loss.item(),
                                                  "latest_validation": latest, "generated_episodes": generated,
                                                  "target_episodes": total_episodes, "replay_episodes": len(replay),
                                                  "device": str(device), "stage_eta_seconds": elapsed / (batch + 1) * ((epochs - epoch) * updates + updates - batch - 1)})
                            last_status = now
                    bar.set_description(f"{stage} epoch {epoch}/{epochs} validating")
                    status_write(output, {"stage": stage, "phase": "validating", "epoch": epoch, "epochs": epochs,
                                          "generated_episodes": generated, "target_episodes": total_episodes,
                                          "latest_validation": history[-1]["validation"] if history else None, "device": str(device)})
                    score, details = validate(model, valsets, reference, runtime, stage, device, config)
                    save(epoch, score, details, total_loss / updates, time.perf_counter() - t, distribution,
                         boundary_samples / (updates * config.batch_size))
                    bar.set_postfix(loss=f"{total_loss / updates:.5g}", val=f"{score:.5g}")
                    status_write(output, {"stage": stage, "phase": "epoch_complete", "epoch": epoch, "epochs": epochs,
                                          "generated_episodes": generated, "target_episodes": total_episodes,
                                          "latest_validation": score, "validation_strata": details, "device": str(device)})
            stage_results[stage] = {"selected_epoch": best_checkpoint["epoch"], "validation": best,
                                    "completed_epochs": epochs, "recorded_seconds": sum(row["seconds"] for row in history)}
            print(f"{stage} complete: selected epoch {best_checkpoint['epoch']}, validation {best:.6g}", flush=True)
    write_json(output / "training_summary.json", {"stages": stage_results, "generated_episodes": generated,
                                                  "unique_transitions": generated * 40, "seconds_this_invocation": time.perf_counter() - started})
    status_write(output, {"stage": "evaluation" if evaluate else "complete", "generated_episodes": generated, "stages": stage_results})
    if evaluate:
        from .online_evaluation import evaluate_run
        evaluate_run(output, boundary)
        status_write(output, {"stage": "complete", "generated_episodes": generated, "stages": stage_results})
    return stage_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("train", "status", "evaluate"), default="train")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    for key, value in asdict(Config()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=float if isinstance(value, float) else int, default=value)
    args = parser.parse_args()
    if args.command == "status":
        print((args.output / "status.json").read_text(encoding="utf-8"))
    elif args.command == "evaluate":
        from .online_evaluation import evaluate_run
        evaluate_run(args.output, args.boundary)
    else:
        try:
            run(args.output, Config(**{key: getattr(args, key) for key in asdict(Config())}), args.device,
                args.workers, args.boundary, args.resume, not args.skip_evaluation)
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
            if (args.output / "status.json").exists():
                status_write(args.output, {"stage": "error", "error": str(error), "continuation": "Resume the last complete epoch into a new directory"})
            parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
