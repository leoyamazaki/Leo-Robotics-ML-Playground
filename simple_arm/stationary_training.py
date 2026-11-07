"""Ground-truth stationary supervision, used only during recursive training."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from .contact_learning import differentiable_next, make_model
from .cube_mpc_experiment import write_json


SCALES = torch.tensor([1, 1, 3, 3, .1, .1, .01, .5, .5, .5, .5, .1, 2])


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    stationary_threshold: float = .002
    stationary_weight: float = 1.
    seed: int = 94
    batch_size: int = 256
    learning_rate: float = .0002

    def __post_init__(self):
        for name in ("epochs", "seed", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        if (not np.isfinite([self.stationary_threshold, self.stationary_weight, self.learning_rate]).all()
                or self.stationary_threshold < 0 or self.stationary_weight < 0 or self.learning_rate <= 0):
            raise ValueError("Invalid stationary threshold, weight or learning rate")


def stationary_intervals(states, threshold):
    """All saved 20 ms samples within a 100 ms interval must stay near its start.

    Output is per transition, not per episode or recursive-window anchor. A
    stationary interval following a push uses the cube's new actual location.
    """
    if states.ndim != 3 or states.shape[1:] != (201, 13) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Require 201 recorded states and a nonnegative threshold")
    xy = np.asarray(states)[..., 4:6]
    following = xy[:, 1:].reshape(len(xy), 40, 5, 2)
    anchor = xy[:, :-1:5, None]
    return np.linalg.norm(following - anchor, axis=-1).max(axis=-1) <= threshold


def stationary_penalty(predicted, actual_start, stationary):
    """Scale XY by 0.1 m, matching the original recursive position scale.

    Mean over the batch including zero terms for moving intervals. No contact
    labels enter this term. The anchor is truth, never the drifting prediction.
    """
    return (((predicted[..., 4:6] - actual_start[..., 4:6]) / .1).square().mean(-1) * stationary).mean()


def recursive_loss(model, truth, actions, contacts, stationary, checkpoint, weight):
    current = truth[:, 0]
    base, extra = current.new_zeros(()), current.new_zeros(())
    horizon = actions.shape[1]
    for offset in range(horizon):
        current = differentiable_next(model, current, actions[:, offset], checkpoint)
        base = base + ((((current - truth[:, offset + 1]) / SCALES).square().mean(-1))
                       * (1 + 2 * contacts[:, offset])).mean() / horizon
        extra = extra + stationary_penalty(current, truth[:, offset], stationary[:, offset]) / horizon
    return base + weight * extra, base, extra


def split_tensors(dataset, split, threshold):
    selected = dataset["split"] == split
    return (torch.tensor(dataset["states"][selected, ::5], dtype=torch.float32),
            torch.tensor(dataset["actions"][selected, ::5], dtype=torch.float32),
            torch.tensor(dataset["contacts"][selected].reshape(-1, 40, 50).any(-1), dtype=torch.float32),
            torch.tensor(stationary_intervals(dataset["states"][selected], threshold), dtype=torch.float32))


@torch.no_grad()
def rollout_losses(model, tensors, checkpoint, stationary_weight, batch_size=256):
    """Common 40-step unweighted diagnostic for every condition and split."""
    model.eval()
    states, actions, _, stationary = tensors
    base_total, stationary_total = 0., 0.
    for first in range(0, len(states), batch_size):
        actual = states[first:first + batch_size]
        current = actual[:, 0]
        for offset in range(40):
            current = differentiable_next(model, current, actions[first:first + batch_size, offset], checkpoint)
            base_total += ((current - actual[:, offset + 1]) / SCALES).square().mean(-1).sum().item()
            stationary_total += stationary_penalty(current, actual[:, offset], stationary[first:first + batch_size, offset]).item() * len(actual)
    base, extra = base_total / (len(states) * 40), stationary_total / (len(states) * 40)
    if not np.isfinite([base, extra]).all():
        raise RuntimeError("Nonfinite diagnostic rollout loss")
    return {"state_scaled_mse": base, "stationary_scaled_mse": extra, "combined": base + stationary_weight * extra}


def train(dataset, source, output, horizon, stationary_weight, config=TrainingConfig()):
    """Identical 21 valid 20-step window starts for all matched conditions."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing training: {output}")
    if horizon not in (10, 20) or not np.isfinite(stationary_weight) or stationary_weight < 0:
        raise ValueError("Require horizon 10 or 20 and nonnegative finite weight")
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint["stride"] != 5 or checkpoint["metadata"] != dataset["metadata"]:
        raise ValueError("Training requires the exact source dataset and 100 ms interval")
    from .contact_models import model_from_checkpoint
    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["weights"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training = split_tensors(dataset, 0, config.stationary_threshold)
    validation = split_tensors(dataset, 1, config.stationary_threshold)
    states, actions, contacts, stationary = training
    windows = torch.cartesian_prod(torch.arange(len(states)), torch.arange(21))
    shuffle = torch.Generator().manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=False)
    history, best, selected, updates = [], float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(config.epochs + 1):
        epoch_started = time.perf_counter()
        totals = np.zeros(3)
        if epoch:
            model.train()
            for indices in torch.randperm(len(windows), generator=shuffle).split(config.batch_size):
                episode, start = windows[indices].T
                offsets = start[:, None] + torch.arange(horizon + 1)
                truth = states[episode[:, None], offsets]
                control = actions[episode[:, None], offsets[:, :-1]]
                active = contacts[episode[:, None], offsets[:, :-1]]
                still = stationary[episode[:, None], offsets[:, :-1]]
                loss, base, extra = recursive_loss(model, truth, control, active, still, checkpoint, stationary_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite stationary training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                totals += np.array([loss.item(), base.item(), extra.item()]) * len(indices)
                updates += 1
        train_loss = rollout_losses(model, training, checkpoint, stationary_weight)
        validation_loss = rollout_losses(model, validation, checkpoint, stationary_weight)
        # Preserve the original, unweighted state validation selector. Stationary
        # metrics never select a checkpoint on their own; test data never select.
        if validation_loss["state_scaled_mse"] < best:
            best, selected = validation_loss["state_scaled_mse"], epoch
            torch.save(dict(checkpoint, weights=model.state_dict(), stationary_refinement_epoch=epoch,
                            training_method="stationary_experiment_recursive", training_horizon_steps=horizon,
                            stationary_threshold=config.stationary_threshold, stationary_weight=stationary_weight,
                            stationary_training_config=asdict(config)), output / "model.pt")
        history.append({"epoch": epoch, "online_combined_loss": float(totals[0] / len(windows)) if epoch else None,
                        "online_base_loss": float(totals[1] / len(windows)) if epoch else None,
                        "online_stationary_loss": float(totals[2] / len(windows)) if epoch else None,
                        "train_40_step": train_loss, "validation_40_step": validation_loss,
                        "epoch_seconds": time.perf_counter() - epoch_started})
        print(f"{output.name} epoch={epoch}/{config.epochs} val={validation_loss['state_scaled_mse']:.5f} "
              f"stationary={validation_loss['stationary_scaled_mse']:.5f}", flush=True)
    torch.save(dict(checkpoint, weights=model.state_dict(), stationary_refinement_epoch=config.epochs,
                    training_method="stationary_experiment_recursive", training_horizon_steps=horizon,
                    stationary_threshold=config.stationary_threshold, stationary_weight=stationary_weight,
                    stationary_training_config=asdict(config)), output / "last.pt")
    report = {"config": asdict(config), "horizon": horizon, "stationary_weight": stationary_weight,
              "selected_epoch": selected, "history": history, "optimizer_updates": updates,
              "windows_per_epoch": len(windows), "updates_per_epoch": (len(windows) + config.batch_size - 1) // config.batch_size,
              "supervised_window_steps": len(windows) * horizon * config.epochs,
              "wall_seconds": time.perf_counter() - started}
    write_json(output / "training.json", report)
    return report
