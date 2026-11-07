"""Evaluate online-stage winners on fixed untouched random and boundary tests."""

import json
from pathlib import Path

import torch

from .contact_models import model_from_checkpoint
from .online_data import load_archive
from .residual_evaluation import cpu_latency, evaluate_checkpoint, subset, write_report
from .residual_training import file_hash, load_data, write_json
from .stationary_experiment import INDEPENDENT


def evaluate_run(run, boundary):
    torch.set_num_threads(2)
    run = Path(run)
    output = run / "evaluation"
    if output.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output}")
    fixed_test = load_archive(run / "data/test.npz")
    boundary_data = load_data(boundary)
    datasets = {"random_test": fixed_test, "boundary_test": subset(boundary_data, boundary_data["split"] == 2)}
    omitted = {}
    for name, path in INDEPENDENT.items():
        if path.is_file():
            datasets[name] = load_data(path)
        else:
            omitted[name] = str(path)
    summary = {"datasets": {name: {"episodes": len(data["states"]), "metadata": data["metadata"]} for name, data in datasets.items()},
               "models": {}, "omitted_independent": omitted,
               "selection": "Online random training with FIFO replay, expected 10% boundary supplement; unweighted loss in both stages. Winners selected on fixed validation only. Test sets are evaluation only. Controller defaults unchanged."}
    for stage in ("one_step", "recursive"):
        checkpoint = run / stage / "model.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        last = torch.load(run / stage / "last.pt", map_location="cpu", weights_only=True)
        if file_hash(boundary) != saved["provenance"]["boundary_sha256"]:
            raise ValueError("Boundary supplement/test archive differs from training")
        if file_hash(run / "data/test.npz") != last["heldout_hashes"]["test.npz"]:
            raise ValueError("Fixed random test archive changed")
        if fixed_test["metadata"]["seed"] != saved["configuration"]["seed"] + 2:
            raise ValueError("Unexpected fixed random test seed")
        for name in datasets:
            if name in INDEPENDENT and datasets[name]["metadata"]["seed"] in (saved["configuration"]["seed"], boundary_data["metadata"]["seed"]):
                raise ValueError("Independent data seed overlaps training")
        history = json.loads((run / stage / "history.json").read_text(encoding="utf-8"))
        info = {"checkpoint": str(checkpoint), "selected_epoch": saved["epoch"],
                "parameter_count": sum(p.numel() for p in model_from_checkpoint(saved).parameters()),
                "training": {"epochs_completed": history[-1]["epoch"], "recorded_seconds": sum(row["seconds"] for row in history)},
                "latency": cpu_latency(checkpoint, fixed_test["states"][:, :-1:5], fixed_test["actions"][:, ::5]),
                "evaluation": {}}
        for name, data in datasets.items():
            info["evaluation"][name] = evaluate_checkpoint(data, checkpoint, output / name / stage)
        summary["models"][stage] = info
    write_json(output / "summary.json", summary)
    write_report(summary, output / "REPORT.md")
    print(f"Evaluation complete: {output / 'REPORT.md'}", flush=True)
    return summary
