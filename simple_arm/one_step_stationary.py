"""Matched stationary XY weighting on saved one-step transitions only.

No collection or recursive feedback occurs in training. Independent rollouts
are evaluation diagnostics and never select a checkpoint.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .capacity_screen import DATASET, NORMALIZATION_KEYS, prepare
from .contact_boundary import load_boundary
from .contact_learning import load_predictor, make_model, new_plot, rollout
from .cube_mpc_experiment import file_hash, source_hashes, write_json
from .stationary_experiment import INDEPENDENT, case_masks, prediction_metrics, vector_rmse


SOURCE = Path("runs/arm_cube_capacity_screen/medium_256/model.pt")
OUTPUT = Path("runs/arm_cube_one_step_stationary")


@dataclass(frozen=True)
class Config:
    epochs: int = 12
    seed: int = 91
    learning_rate: float = .001
    batch_size: int = 256
    speed_tolerance: float = .005
    excursion_tolerance: float = .0005

    def __post_init__(self):
        for key in ("epochs", "batch_size", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid {key}")
        if (not np.isfinite([self.learning_rate, self.speed_tolerance, self.excursion_tolerance]).all()
                or self.learning_rate <= 0 or min(self.speed_tolerance, self.excursion_tolerance) < 0):
            raise ValueError("Invalid learning rate or stationarity tolerances")


def stationary_intervals(states, speed_tolerance=.005, excursion_tolerance=.0005):
    """Inclusive bounds across the actual start and five following 20 ms states."""
    states = np.asarray(states)
    if (states.ndim != 3 or states.shape[1:] != (201, 13) or not np.isfinite(states).all()
            or not np.isfinite([speed_tolerance, excursion_tolerance]).all()
            or min(speed_tolerance, excursion_tolerance) < 0):
        raise ValueError("Expected finite saved states and nonnegative tolerances")
    samples = states[:, np.arange(40)[:, None] * 5 + np.arange(6)]
    speed = np.linalg.norm(samples[..., 9:11], axis=-1).max(-1)
    excursion = np.linalg.norm(samples[..., 4:6] - samples[..., :1, 4:6], axis=-1).max(-1)
    return (speed <= speed_tolerance) & (excursion <= excursion_tolerance)


def objective(predicted, target, contacts, stationary, multiplier):
    """Fixed 12-component denominator, original 5x contact weights.

    Multiplication by one preserves the original operation order at 1x.
    Only stationary columns 4 and 5 receive an additional coefficient.
    """
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Multiplier must be finite and >=1")
    return weighted_error((predicted - target).square(), contacts, stationary, multiplier)


def weighted_error(error, contacts, stationary, multiplier):
    coefficients = torch.ones_like(error)
    coefficients[:, 4:6] += stationary[:, None] * (multiplier - 1)
    return ((error * coefficients).mean(-1) * contacts).mean()


@torch.no_grad()
def losses(model, arrays, stationary, multiplier):
    model.eval()
    x, y, contact = arrays
    error = torch.cat([(model(part) - y[i:i + len(part)]).square()
                       for i, part in zip(range(0, len(x), 4096), x.split(4096))])
    def xy(mask):
        return float(error[mask, 4:6].mean()) if mask.any() else None
    # Evaluate the same objective without a second model forward.
    weighted = weighted_error(error, contact, stationary, multiplier)
    return {"weighted_objective": float(weighted), "unweighted": float(error.mean()),
            "stationary_xy": xy(stationary), "moving_xy": xy(~stationary),
            "unrelated_outputs": float(error[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]].mean()),
            "stationary_transitions": int(stationary.sum()), "moving_transitions": int((~stationary).sum())}


def train(dataset, source, output, multiplier, config=Config()):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Multiplier must be finite and >=1")
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    reference = torch.load(source, weights_only=True, map_location="cpu")
    training, validation = prepare(dataset, reference)
    masks = stationary_intervals(dataset["states"], config.speed_tolerance, config.excursion_tolerance)
    still_train, still_val = [torch.tensor(masks[dataset["split"] == split].reshape(-1)) for split in (0, 1)]
    model = make_model(reference.get("hidden_width", 128))
    model.load_state_dict(reference["weights"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    order_hash = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history, best, selected, updates = [], float("inf"), 0, 0
    def save(path, epoch):
        torch.save(dict(reference, weights=model.state_dict(), epoch=epoch,
                        training_method="one_step_stationary", stationary_multiplier=multiplier,
                        stationary_config=asdict(config)), path)
    for epoch in range(config.epochs + 1):
        tick = time.perf_counter()
        if epoch:
            model.train()
            order = torch.randperm(len(training[0]), generator=generator)
            order_hash.update(order.numpy().tobytes())
            x, y, contacts = training
            for indices in order.split(config.batch_size):
                optimizer.zero_grad()
                loss = objective(model(x[indices]), y[indices], contacts[indices], still_train[indices], multiplier)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite one-step training loss")
                loss.backward()
                optimizer.step()
                updates += 1
        train_loss = losses(model, training, still_train, multiplier)
        val_loss = losses(model, validation, still_val, multiplier)
        if not np.isfinite(val_loss["unweighted"]):
            raise RuntimeError("Nonfinite validation loss")
        if val_loss["unweighted"] < best:
            best, selected = val_loss["unweighted"], epoch
            save(output / "best.pt", epoch)
        history.append({"epoch": epoch, "updates": updates, "train": train_loss,
                        "validation": val_loss, "epoch_seconds": time.perf_counter() - tick})
        print(f"{output.name} epoch={epoch}/{config.epochs} val={val_loss['unweighted']:.6g}", flush=True)
    save(output / "final.pt", config.epochs)
    report = {"config": asdict(config), "multiplier": multiplier, "history": history,
              "optimizer_updates": updates, "transitions_per_epoch": len(training[0]),
              "batch_order_sha256": order_hash.hexdigest(), "selected_best_epoch": selected,
              "primary_epoch": config.epochs, "wall_seconds": time.perf_counter() - started,
              "source_sha256": file_hash(source), "no_recursive_training": True}
    write_json(output / "training.json", report)
    return report


def validate_matching(models, reference, config):
    """Audit saved primary/secondary checkpoints and schedules before evaluation."""
    reports = [m["training"] for m in models.values() if "training" in m]
    expected = config.epochs * ((reports[0]["transitions_per_epoch"] + config.batch_size - 1) // config.batch_size)
    if len({r["batch_order_sha256"] for r in reports}) != 1:
        raise RuntimeError("Batch orders differ")
    for info in models.values():
        if "training" not in info:
            continue
        report = info["training"]
        if report["optimizer_updates"] != expected or report["primary_epoch"] != config.epochs:
            raise RuntimeError("Update budget or primary epoch differs")
        for filename, epoch in (("final.pt", config.epochs), ("best.pt", report["selected_best_epoch"])):
            saved = torch.load(Path(info["checkpoint"]).parent / filename, weights_only=True, map_location="cpu")
            if saved["epoch"] != epoch:
                raise RuntimeError("Saved checkpoint epoch differs")
            for key in NORMALIZATION_KEYS:
                if not torch.equal(saved[key], reference[key]):
                    raise RuntimeError("Normalization changed")
        for split in ("train", "validation"):
            zero = report["history"][0][split]
            first = reports[0]["history"][0][split]
            for key in ("unweighted", "stationary_xy", "moving_xy", "unrelated_outputs"):
                if zero[key] != first[key]:
                    raise RuntimeError("Initialization losses differ")


def displacement_metrics(dataset, teacher, config=Config()):
    actual = dataset["states"][:, ::5]
    truth = np.diff(actual[..., 4:6], axis=1)
    predicted = teacher[..., 4:6] - actual[:, :-1, 4:6]
    error = predicted - truth
    stationary = stationary_intervals(dataset["states"], config.speed_tolerance, config.excursion_tolerance)
    contact = dataset["contacts"].reshape(len(actual), 40, 50).any(-1)
    magnitude, predicted_magnitude = [np.linalg.norm(a, axis=-1) for a in (truth, predicted)]
    def cohort(mask):
        return {"samples": int(mask.sum()), "rmse_m": vector_rmse(error[mask])}
    def distribution(values):
        return ({"samples": len(values), "mean_m": float(values.mean()), "median_m": float(np.median(values)),
                 "p95_m": float(np.quantile(values, .95)), "max_m": float(values.max())} if len(values)
                else {"samples": 0, "mean_m": None, "median_m": None, "p95_m": None, "max_m": None})
    report = {"stationary": cohort(stationary), "moving": cohort(~stationary),
              "moving_contact": cohort(~stationary & contact),
              "cases": {k: cohort(np.broadcast_to(v[:, None], stationary.shape)) for k, v in case_masks(dataset).items()},
              "stationary_predicted_magnitude": distribution(predicted_magnitude[stationary]),
              "rates": {}}
    for threshold in (.0005, .002):
        moving_contact = contact & (magnitude > threshold)
        false_count = int((stationary & (predicted_magnitude > threshold)).sum())
        missed_count = int((moving_contact & (predicted_magnitude <= threshold)).sum())
        report["rates"][str(threshold)] = {
            "stationary_samples": int(stationary.sum()), "false_motion_count": false_count,
            "false_motion_rate": false_count / int(stationary.sum()) if stationary.any() else None,
            "contact_motion_samples": int(moving_contact.sum()), "missed_motion_count": missed_count,
            "missed_motion_rate": missed_count / int(moving_contact.sum()) if moving_contact.any() else None}
    moving = contact & (magnitude > .002)
    directed = moving & (predicted_magnitude > 1e-12)
    cosine = (predicted * truth).sum(-1)[directed] / (predicted_magnitude[directed] * magnitude[directed])
    report["contact_motion"] = {**cohort(moving),
        "magnitude_rmse_m": float(np.sqrt(np.mean((predicted_magnitude[moving] - magnitude[moving])**2))) if moving.any() else None,
        "predicted_magnitude": distribution(predicted_magnitude[moving]),
        "truth_magnitude": distribution(magnitude[moving]), "direction_samples": int(directed.sum()),
        "mean_angle_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()) if len(cosine) else None,
        "within_30deg_and_above_2mm_rate": float(((cosine >= np.cos(np.pi / 6)) & (predicted_magnitude[directed] > .002)).sum() / moving.sum()) if moving.any() else None,
        "wrong_direction_rate": float((cosine < 0).sum() / moving.sum()) if moving.any() else None}
    return report


def evaluate(dataset, checkpoint, output, config=Config()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actual, actions = dataset["states"][:, ::5], dataset["actions"][:, ::5]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(parent["seed"] == dataset["metadata"]["seed"] for parent in [saved["metadata"], *saved["metadata"].get("parents", [])]):
        raise ValueError("Independent evaluation seed overlaps training")
    predict = load_predictor(checkpoint, 5)
    teacher = predict(actual[:, :-1], actions)
    ar = rollout(predict, actual[:, 0], actions)
    accumulated = actual.copy()
    accumulated[:, 1:, 4:6] = actual[:, :1, 4:6] + np.cumsum(teacher[..., 4:6] - actual[:, :-1, 4:6], axis=1)
    report = {"one_step": displacement_metrics(dataset, teacher, config),
              "rollout_diagnostics": prediction_metrics(dataset, teacher, ar, accumulated)}
    np.savez_compressed(output / "predictions.npz", actual=actual, actions=actions, teacher=teacher,
                        autoregressive=ar, teacher_accumulated=accumulated)
    write_json(output / "metrics.json", report)
    return report


def recommendation(models):
    """Independent metrics compare fixed final epochs against the 1x control."""
    baseline = models["weight_1"]["evaluation"]
    decisions = {}
    for name, info in models.items():
        if name in ("initialization", "weight_1"):
            continue
        reasons, gains = [], []
        for dataset, base_eval in baseline.items():
            base, candidate = base_eval["one_step"], info["evaluation"][dataset]["one_step"]
            b, c = base["stationary"]["rmse_m"], candidate["stationary"]["rmse_m"]
            if b is None or c is None or b <= 0 or c >= b:
                reasons.append(f"{dataset}: stationary RMSE did not improve")
            else:
                gains.append(1 - c / b)
            for cohort in ("moving_contact", "clear_push"):
                b, c = [(m["cases"][cohort] if cohort == "clear_push" else m[cohort])["rmse_m"] for m in (base, candidate)]
                if b is None or c is None:
                    reasons.append(f"{dataset}: {cohort} unavailable")
                elif c > b * 1.1:
                    reasons.append(f"{dataset}: {cohort} RMSE regression {100*(c/b-1):.2f}% exceeds 10%" if b else f"{dataset}: {cohort} regressed from zero")
            b, c = [m["rates"]["0.002"]["missed_motion_rate"] for m in (base, candidate)]
            if b is None or c is None:
                reasons.append(f"{dataset}: missed motion unavailable")
            elif c > b + .05:
                reasons.append(f"{dataset}: missed motion increase {100*(c-b):.2f}pp exceeds 5pp")
        decisions[name] = {"eligible": not reasons, "reasons": reasons,
                           "average_relative_stationary_improvement": float(np.mean(gains)) if len(gains) == len(baseline) else None}
    eligible = [name for name in decisions if decisions[name]["eligible"]]
    winner = min(eligible, key=lambda name: (-decisions[name]["average_relative_stationary_improvement"], models[name]["multiplier"])) if eligible else "weight_1"
    return {"recommended": winner, "candidates": decisions, "deployment_endorsement": False}


def plots(summary, output):
    for split in ("train", "validation"):
        figure, axes = new_plot()
        for name, info in summary["models"].items():
            if "training" not in info:
                continue
            history = info["training"]["history"]
            line, = axes.semilogy([r["epoch"] for r in history], [r[split]["unweighted"] for r in history], label=f"{name} unweighted")
            axes.semilogy([r["epoch"] for r in history], [r[split]["weighted_objective"] for r in history], "--", color=line.get_color(), label=f"{name} objective")
        axes.set(xlabel="Additional epoch", ylabel="Normalized MSE", title=split)
        axes.legend(fontsize=6)
        figure.savefig(output / f"{split}_learning_curves.png")
    for dataset in INDEPENDENT:
        figure, axes = new_plot()
        for name, info in summary["models"].items():
            with np.load(output / "evaluations" / dataset / name / "predictions.npz") as saved:
                actual, ar, teacher = saved["actual"], saved["autoregressive"], saved["teacher"]
                line, = axes.plot(np.arange(1, 41) / 10, np.sqrt(np.mean(np.sum((ar[:, 1:, 4:6] - actual[:, 1:, 4:6])**2, -1), 0)) * 1000, label=f"{name} AR")
                axes.plot(np.arange(1, 41) / 10, np.sqrt(np.mean(np.sum((teacher[..., 4:6] - actual[:, 1:, 4:6])**2, -1), 0)) * 1000, ":", color=line.get_color(), label=f"{name} TF")
        axes.set(xlabel="Time (s)", ylabel="XY vector RMSE (mm)", title=dataset)
        axes.legend(fontsize=5)
        figure.savefig(output / f"{dataset}_four_second_errors.png")
        for threshold in ("0.0005", "0.002"):
            figure, axes = new_plot()
            for name, info in summary["models"].items():
                rate = info["evaluation"][dataset]["one_step"]["rates"][threshold]
                if rate["false_motion_rate"] is not None and rate["missed_motion_rate"] is not None:
                    axes.scatter(rate["false_motion_rate"] * 100, rate["missed_motion_rate"] * 100, label=name)
            rate = summary["zero_xy_reference"][dataset]["rates"][threshold]
            axes.scatter(rate["false_motion_rate"] * 100, rate["missed_motion_rate"] * 100, marker="x", label="zero XY (evaluation only)")
            axes.set(xlabel="Stationary false motion (%)", ylabel="Contact missed motion (%)", title=f"{dataset}: {float(threshold)*1000:g} mm")
            axes.legend(fontsize=6)
            figure.savefig(output / f"{dataset}_tradeoff_{threshold}.png")


def run(output=OUTPUT, source=SOURCE, config=Config(), multipliers=(1., 5., 20., 100.)):
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment: {output}")
    if (1. not in multipliers or len(set(multipliers)) != len(multipliers)
            or not np.isfinite(multipliers).all() or min(multipliers) < 1):
        raise ValueError("Require unique finite multipliers >=1 including 1")
    torch.set_num_threads(2)
    before = source_hashes(source, DATASET)
    data = load_boundary(DATASET)
    reference = torch.load(source, map_location="cpu", weights_only=True)
    if reference.get("hidden_width") != 256 or reference.get("training_method") != "one_step_only_capacity_screen":
        raise ValueError("Expected the one-step-only Medium capacity checkpoint")
    if len(data["states"]) * 40 != 99200 or np.bincount(data["split"]).tolist() != [1736, 372, 372]:
        raise ValueError("Expected original mixed data and unchanged splits")
    prepare(data, reference)
    independent = {name: load_boundary(path) for name, path in INDEPENDENT.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"config": asdict(config), "multipliers": multipliers,
        "source": str(source), "source_hashes_before": before, "dataset": str(DATASET),
        "independent_datasets": {k: str(v) for k, v in INDEPENDENT.items()},
        "primary": "Fixed final epoch; best unweighted validation including epoch zero is diagnostic only",
        "recommendation_rule": "Improve stationary RMSE on both datasets; moving-contact and clear-push RMSE <=1.1x control on each; 2mm missed motion <=control+0.05 on each; maximize mean relative stationary improvement, tie toward smaller weight; else 1x",
        "stationarity": "All six saved samples: speed <= tolerance and XY excursion from actual interval start <= tolerance",
        "loss": "Mean over 12 normalized squared residuals; stationary columns 4,5 multiplied; then existing 5x contact weight",
        "threads": 2, "no_collection": True, "no_recursive_training": True})
    models = {"initialization": {"checkpoint": str(source)}}
    for multiplier in multipliers:
        name = f"weight_{multiplier:g}"
        stats = train(data, source, output / name, multiplier, config)
        models[name] = {"checkpoint": str(output / name / "final.pt"), "multiplier": multiplier, "training": stats}
    validate_matching(models, reference, config)
    orders = {info["training"]["batch_order_sha256"] for info in models.values() if "training" in info}
    input_hashes, zero = {}, {}
    for name, info in models.items():
        saved = torch.load(info["checkpoint"], weights_only=True, map_location="cpu")
        for key in NORMALIZATION_KEYS:
            if not torch.equal(reference[key], saved[key]):
                raise RuntimeError("Normalization changed")
        info["evaluation"] = {}
        for dataset, independent_data in independent.items():
            directory = output / "evaluations" / dataset / name
            info["evaluation"][dataset] = evaluate(independent_data, info["checkpoint"], directory, config)
            with np.load(directory / "predictions.npz") as saved_predictions:
                digest = hashlib.sha256(saved_predictions["actual"].tobytes() + saved_predictions["actions"].tobytes()).hexdigest()
            if dataset in input_hashes and input_hashes[dataset] != digest:
                raise RuntimeError("Evaluation states/actions differ")
            input_hashes[dataset] = digest
    for name, independent_data in independent.items():
        zero[name] = displacement_metrics(independent_data, independent_data["states"][:, :-1:5].copy(), config)
        write_json(output / f"{name}_zero_xy_reference.json", zero[name])
    after = {p: file_hash(p) for p in before}
    summary = {"models": models, "zero_xy_reference": zero, "recommendation": recommendation(models),
               "source_hashes_after": after, "sources_unchanged": before == after,
               "evaluation_input_hashes": input_hashes, "identical_batch_orders": len(orders) == 1}
    write_json(output / "summary.json", summary)
    if before != after:
        raise RuntimeError("Existing datasets/checkpoints changed")
    plots(summary, output)
    write_report(summary, output)
    print(f"Complete: {output}; recommendation: {summary['recommendation']['recommended']}", flush=True)
    return summary


def write_report(summary, output):
    control = summary["models"]["weight_1"]["training"]
    config = control["config"]
    lines = ["# One-step stationary-weighted Medium experiment", "",
             f"Recommendation: **{summary['recommendation']['recommended']}**. This is a one-step comparison, not a deployment endorsement.", "",
             f"Each model starts from runs/arm_cube_capacity_screen/medium_256/model.pt and uses {config['epochs']} additional epochs, batch size {config['batch_size']}, fresh Adam at learning rate {config['learning_rate']}, seed {config['seed']}, and two CPU threads. Original 99,200-transition data and episode splits 1,736/372/372 remain unchanged. No collection or recursive training occurs.", "",
             "Features, architecture, normalization, 12 displacement targets, and 100 ms interval are preserved. Only stationary squared normalized residuals in columns 4 and 5 receive the multiplier. The denominator stays 12, then the original 5x contact-transition weight is applied. Small actual displacements within tolerance remain the target. Histories include epoch zero and post-epoch train/validation weighted objective, unweighted overall error, stationary/moving XY error, and unrelated-output error.", "",
             "All primary comparisons use the fixed final epoch. Best unweighted-validation checkpoints include epoch zero and remain secondary diagnostics. Independent data never select checkpoints.", "",
             "A multiplier above 1x must improve stationary XY RMSE on both datasets; on each, moving-contact and clear-push one-step RMSE must stay within 10% of 1x, and missed motion at 2mm may increase at most 5 percentage points. Among eligible weights, maximize average relative stationary improvement, breaking ties toward the smaller weight; if none qualify, retain 1x.", "",
             "Stationary intervals require planar speed <=5 mm/s and excursion <=0.5 mm across the actual start and all five subsequent saved samples (unless overridden in config.json). XY errors below are vector RMSE. Rates use strictly greater than the reporting threshold; missed motion requires actual contact. Empty cohorts are unavailable.", ""]
    for dataset in INDEPENDENT:
        lines += [f"## {dataset}", "", "|Model|Stationary mm|Moving mm|Moving contact mm|Clear push mm|False 0.5mm %|False 2mm %|Missed 2mm %|4s endpoint cm|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(value, scale=1):
            return "unavailable" if value is None else f"{value*scale:.4f}"
        for name, info in summary["models"].items():
            one = info["evaluation"][dataset]["one_step"]
            ar = info["evaluation"][dataset]["rollout_diagnostics"]
            values = [fmt(one[key]["rmse_m"], 1000) for key in ("stationary", "moving", "moving_contact")]
            values += [fmt(one["cases"]["clear_push"]["rmse_m"], 1000),
                       fmt(one["rates"]["0.0005"]["false_motion_rate"], 100), fmt(one["rates"]["0.002"]["false_motion_rate"], 100),
                       fmt(one["rates"]["0.002"]["missed_motion_rate"], 100),
                       fmt(ar["teacher_forced_vs_autoregressive"]["all"]["autoregressive_4s_endpoint_rmse_m"], 100)]
            lines.append("|" + "|".join([name, *values]) + "|")
        lines += ["", "Full metrics and predictions are saved separately per model and dataset. Teacher-forced XY errors are displacement errors; no prediction is fed back during training. Contact-motion direction requires >2 mm actual displacement and is measured against the actual displacement vector. Zero-length predictions have undefined angle and fail the joint magnitude/direction accuracy measure.", "",
                  "### Cohorts, case errors, and interval errors", "",
                  "|Model|Stationary n|Moving n|Moving contact n|Light case n / RMSE mm|Clear case n / RMSE mm|No-contact interval n / mm|Light interval n / mm|Active-push interval n / mm|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, info in summary["models"].items():
            one = info["evaluation"][dataset]["one_step"]
            diagnostic = info["evaluation"][dataset]["rollout_diagnostics"]
            case = [f"{one['cases'][key]['samples']} / {fmt(one['cases'][key]['rmse_m'], 1000)}" for key in ("light_contact", "clear_push")]
            intervals = [f"{diagnostic['one_step_intervals'][key]['transitions']} / {fmt(diagnostic['one_step_intervals'][key]['cube_displacement_rmse_m'], 1000)}" for key in ("no_contact", "light_contact", "active_push")]
            lines.append("|" + "|".join([name, *[str(one[key]["samples"]) for key in ("stationary", "moving", "moving_contact")], *case, *intervals]) + "|")
        lines += ["", "Boundary case labels are saved collector kinds. Broad light contact requires >0 and <=0.2s contact and <=2cm excursion; broad clear push requires >=0.1s contact and >=5cm excursion. Interval light contact means 1..10 active substeps out of 50; active push means contact and >2mm net motion. These interval cohorts may overlap. Whole-episode rollout stationarity uses <=2mm excursion from the episode's initial position, distinct from the six-sample transition label.", "",
                  "### False and missed motion, including evaluation-only zero XY", "",
                  "|Model|Threshold mm|Stationary n|False count / %|Contact-motion n|Missed count / %|", "|---|---:|---:|---:|---:|---:|"]
        for name, one in [(name, info["evaluation"][dataset]["one_step"]) for name, info in summary["models"].items()] + [("zero_xy_reference", summary["zero_xy_reference"][dataset])]:
            for threshold, rate in one["rates"].items():
                lines.append(f"|{name}|{float(threshold)*1000:g}|{rate['stationary_samples']}|{rate['false_motion_count']} / {fmt(rate['false_motion_rate'], 100)}|{rate['contact_motion_samples']}|{rate['missed_motion_count']} / {fmt(rate['missed_motion_rate'], 100)}|")
        lines += ["", "### Predicted displacement magnitudes and contact direction", "",
                  "|Model|Stationary predicted mean / median / p95 / max mm|Contact n|Predicted / actual mean mm|Magnitude RMSE mm|Direction n / mean angle deg|Wrong direction %|Within 30deg and >2mm %|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, one in [(name, info["evaluation"][dataset]["one_step"]) for name, info in summary["models"].items()] + [("zero_xy_reference", summary["zero_xy_reference"][dataset])]:
            magnitude = one["stationary_predicted_magnitude"]
            contact = one["contact_motion"]
            stationary = " / ".join(fmt(magnitude[key], 1000) for key in ("mean_m", "median_m", "p95_m", "max_m"))
            values = [name, stationary, str(contact["samples"]), f"{fmt(contact['predicted_magnitude']['mean_m'], 1000)} / {fmt(contact['truth_magnitude']['mean_m'], 1000)}",
                      fmt(contact["magnitude_rmse_m"], 1000), f"{contact['direction_samples']} / {fmt(contact['mean_angle_error_deg'])}",
                      fmt(contact["wrong_direction_rate"], 100), fmt(contact["within_30deg_and_above_2mm_rate"], 100)]
            lines.append("|" + "|".join(values) + "|")
        lines += ["", "Wrong-direction rate divides opposite predictions by all contact-motion samples. Zero predictions have undefined direction; the joint >2mm/30deg measure counts them as failures. The zero-XY reference is never part of inference or deployment.", "",
                  "### Four-second diagnostics", "", "|Model|Cohort|Episodes|TF one-step mm|AR prefix mm|AR increment mm|AR 4s endpoint cm|TF summed 4s cm|",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, info in summary["models"].items():
            diagnostic = info["evaluation"][dataset]["rollout_diagnostics"]
            for cohort, metric in diagnostic["teacher_forced_vs_autoregressive"].items():
                values = [name, cohort, str(metric["episodes"]),
                          fmt(metric["teacher_forced_one_step_rmse_m"], 1000),
                          fmt(metric["autoregressive_prefix_position_rmse_m"], 1000),
                          fmt(metric["autoregressive_increment_rmse_m"], 1000),
                          fmt(metric["autoregressive_4s_endpoint_rmse_m"], 100),
                          fmt(metric["teacher_forced_accumulated_4s_diagnostic_rmse_m"], 100)]
                lines.append("|" + "|".join(values) + "|")
            push = diagnostic["push_4s"]
            lines.append(f"|{name}|actual push (>2cm endpoint, contact)|{push['episodes']}|unavailable|unavailable|unavailable|{fmt(push['cube_endpoint_rmse_m'], 100)}|unavailable|")
        lines += ["", "The TF summed displacement is an error-accumulation diagnostic, not an autoregressive physical rollout.", "",
                  "|Model|Stationary cohort|Episodes|AR mean / p95 / max drift mm|AR >2mm %|TF next mean mm|TF summed mean mm|",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, info in summary["models"].items():
            for cohort, drift in info["evaluation"][dataset]["rollout_diagnostics"]["stationary_drift"].items():
                ar = drift["autoregressive"]
                values = [name, cohort, str(ar["episodes"]),
                          " / ".join(fmt(ar.get(key), 1000) for key in ("mean_max_drift_m", "p95_max_drift_m", "worst_max_drift_m")),
                          fmt(ar.get("fraction_over_2mm"), 100), fmt(drift["teacher_forced_next"].get("mean_max_drift_m"), 1000),
                          fmt(drift["teacher_forced_accumulated_diagnostic"].get("mean_max_drift_m"), 1000)]
                lines.append("|" + "|".join(values) + "|")
        lines += ["",
                  "### Rollout regressions and initialization comparisons", ""]
        for name, info in summary["models"].items():
            if name == "initialization":
                continue
            for reference_name in ("initialization", "weight_1"):
                if reference_name == name:
                    continue
                ref = summary["models"][reference_name]["evaluation"][dataset]
                current = info["evaluation"][dataset]
                lines.append(f"{name} versus {reference_name}:")
                for cohort in ("stationary", "moving", "moving_contact"):
                    a, b = [m["one_step"][cohort]["rmse_m"] for m in (ref, current)]
                    lines.append(f"- One-step {cohort} RMSE change: {fmt(None if a is None or b is None else b-a, 1000)} mm.")
                for cohort, metrics in current["rollout_diagnostics"]["teacher_forced_vs_autoregressive"].items():
                    a = ref["rollout_diagnostics"]["teacher_forced_vs_autoregressive"][cohort]["autoregressive_4s_endpoint_rmse_m"]
                    b = metrics["autoregressive_4s_endpoint_rmse_m"]
                    lines.append(f"- {cohort} 4s endpoint RMSE change: {fmt(None if a is None or b is None else b-a, 100)} cm (positive is regression).")
                for cohort, drift in current["rollout_diagnostics"]["stationary_drift"].items():
                    a = ref["rollout_diagnostics"]["stationary_drift"][cohort]["autoregressive"].get("mean_max_drift_m")
                    b = drift["autoregressive"].get("mean_max_drift_m")
                    lines.append(f"- {cohort} mean maximum drift change: {fmt(None if a is None or b is None else b-a, 1000)} mm.")
                lines.append("")
        lines += ["### Every rollout regression", "",
                  "All scalar rollout errors/drift measures are compared in comparisons.json, including improvements. Positive changes below are regressions; units follow the metric suffix (m or dimensionless fraction). These diagnostics do not enter the recommendation rule.", "",
                  "|Model|Reference|Metric|Increase|", "|---|---|---|---:|"]
        comparisons = diagnostic_comparisons(summary["models"])
        for name, references in comparisons[dataset].items():
            for reference_name, metrics in references.items():
                for key, delta in metrics.items():
                    if delta["change"] is not None and delta["change"] > 0:
                        lines.append(f"|{name}|{reference_name}|{key}|{delta['change']:.7g}|")
    lines += ["## Eligibility", ""]
    for name, decision in summary["recommendation"]["candidates"].items():
        lines.append(f"- {name}: {'eligible' if decision['eligible'] else '; '.join(decision['reasons'])}.")
    lines += ["", "## Verification", "", f"Existing datasets/checkpoints unchanged: {summary['sources_unchanged']}. Identical batch-order hashes: {summary['identical_batch_orders']}. Normalization and saved evaluation states/actions checked for every model. See config.json and summary.json for SHA-256 evidence.", ""]
    for name, info in summary["models"].items():
        if "training" in info:
            t = info["training"]
            lines.append(f"- {name}: {t['optimizer_updates']} updates; final epoch {t['primary_epoch']}; diagnostic best epoch {t['selected_best_epoch']}; {t['wall_seconds']:.2f} seconds.")
    write_json(output / "comparisons.json", diagnostic_comparisons(summary["models"]))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostic_comparisons(models):
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, path))
            elif key not in ("episodes", "transitions", "stationary_episodes"):
                result[path] = item
        return result
    comparisons = {}
    for dataset in INDEPENDENT:
        comparisons[dataset] = {}
        for name, info in models.items():
            if name == "initialization":
                continue
            current = flatten(info["evaluation"][dataset]["rollout_diagnostics"])
            comparisons[dataset][name] = {}
            for reference_name in ("initialization", "weight_1"):
                if name == reference_name:
                    continue
                reference = flatten(models[reference_name]["evaluation"][dataset]["rollout_diagnostics"])
                comparisons[dataset][name][reference_name] = {
                    key: {"reference": reference[key], "current": value,
                          "change": value - reference[key] if value is not None and reference[key] is not None else None}
                    for key, value in current.items()}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1., 5., 20., 100.])
    for name, value in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    run(args.output, args.source, Config(**{name: getattr(args, name) for name in asdict(Config())}), args.multipliers)


if __name__ == "__main__":
    main()
