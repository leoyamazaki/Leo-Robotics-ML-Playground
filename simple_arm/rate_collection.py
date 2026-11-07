"""Regenerate fixed-rate broad/contact-boundary datasets without starting training."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .collection_workers import DEFAULT_WORKERS
from .contact_boundary import collect, load_boundary, merge
from .contact_varied import collect_varied
from .residual_training import code_hashes, file_hash, write_json


def check_rate(dataset, rate=.2):
    initial = dataset["states"][:, :1, :2]
    commands = dataset["actions"][:, ::5]
    peak = float(np.abs(np.diff(np.concatenate([initial, commands], 1), axis=1)).max() / .1)
    if peak > rate + 1e-9:
        raise ValueError(f"Command slew exceeds {rate} rad/s: {peak}")
    return {"peak_command_rate_rad_s": peak,
            "maximum_saved_joint_speed_rad_s": np.abs(dataset["states"][..., 2:4]).max(axis=(0, 1)).tolist(),
            "p95_saved_joint_speed_rad_s": np.quantile(np.abs(dataset["states"][..., 2:4]), .95, axis=(0, 1)).tolist(),
            "contact_interval_fraction": float(dataset["contacts"].reshape(len(initial), 40, 50).any(-1).mean()),
            "cube_endpoint_over_2cm_fraction": float((np.linalg.norm(dataset["states"][:, -1, 4:6] - dataset["states"][:, 0, 4:6], axis=-1) > .02).mean()),
            "split_episodes": np.bincount(dataset["split"], minlength=3).tolist()}


def generate(root, pilot=False, workers=DEFAULT_WORKERS):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    previous = json.loads((root / "rate_collection.json").read_text(encoding="utf-8")) if (root / "rate_collection.json").exists() else {}
    specs = [("broad", 120 if pilot else 1200, 490 if pilot else 491, "broad.npz"),
             ("boundary", 40 if pilot else 320, 590 if pilot else 591, "focused.npz")]
    if not pilot:
        specs += [("boundary", 40, 592, "independent.npz"), ("broad", 120, 493, "broad_independent.npz")]
    manifest = {"command_rate_limit_rad_s": .2, "actual_joint_speeds": "measured, not constrained",
                "pilot": pilot, "workers": workers, "source_code_hashes": code_hashes(), "datasets": {}}
    for kind, count, seed, name in specs:
        path = root / name
        if path.exists():
            dataset = load_boundary(path)
            if (dataset["metadata"].get("command_rate_limit_rad_s") != .2 or dataset["metadata"]["seed"] != seed
                    or len(dataset["states"]) != count * (4 if kind == "boundary" else 1)):
                raise ValueError(f"Existing {path} does not match this generation protocol; choose another root")
        elif kind == "broad":
            dataset = collect_varied(path, count, seed, workers=workers, command_rate=.2)
        else:
            dataset = collect(path, count, seed, workers=workers, command_rate=.2)
        prior = previous.get("datasets", {}).get(name, {})
        if prior and prior["sha256"] != file_hash(path):
            raise ValueError(f"Existing {path} changed since its recorded collection manifest")
        manifest["datasets"][name] = {"seed": seed, "episodes": len(dataset["states"]), "sha256": file_hash(path),
                                      "source_code_hashes": prior.get("source_code_hashes", previous.get("source_code_hashes", manifest["source_code_hashes"])) if prior else manifest["source_code_hashes"],
                                      **check_rate(dataset)}
        write_json(root / "rate_collection.json", manifest)
    mixed = root / "mixed.npz"
    dataset = load_boundary(mixed) if mixed.exists() else merge(root / "focused.npz", root / "broad.npz", mixed)
    expected_parents = [load_boundary(root / name)["metadata"] for name in ("broad.npz", "focused.npz")]
    if dataset["metadata"].get("parents") != expected_parents:
        raise ValueError("Existing mixed dataset does not match the saved source datasets")
    manifest["datasets"]["mixed.npz"] = {"episodes": len(dataset["states"]), "sha256": file_hash(mixed), **check_rate(dataset)}
    manifest["wall_seconds_this_invocation"] = time.perf_counter() - started
    write_json(root / "rate_collection.json", manifest)
    print(f"{'Pilot' if pilot else 'Dataset regeneration'} complete: {root}; {len(dataset['states'])} mixed episodes", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "collect"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    root = args.root or Path("data/arm_cube_rate_02_pilot" if args.phase == "pilot" else "data/arm_cube_contact_boundary")
    generate(root, args.phase == "pilot", args.workers)


if __name__ == "__main__":
    main()
