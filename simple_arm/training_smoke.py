"""Verify CPU learning on synthetic data, independently of robot physics."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn


def make_model():
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)).cpu()


def make_samples(generator, count):
    """Known relationship: output = 2 * first input - second input + 0.5."""
    inputs = generator.uniform(-1, 1, (count, 2)).astype(np.float32)
    outputs = 2 * inputs[:, :1] - inputs[:, 1:] + 0.5
    return torch.from_numpy(inputs).cpu(), torch.from_numpy(outputs).cpu()


def run_smoke(output_dir=Path("runs/training_smoke")):
    """Train, validate, save artifacts, and raise if environment checks fail."""
    torch.manual_seed(42)
    generator = np.random.default_rng(42)
    train_inputs, train_targets = make_samples(generator, 512)
    validation_inputs, validation_targets = make_samples(generator, 128)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    training_losses = []
    validation_losses = []

    def record_losses():
        model.eval()
        with torch.no_grad():
            training_losses.append(loss_function(model(train_inputs), train_targets).item())
            validation_losses.append(loss_function(model(validation_inputs), validation_targets).item())

    record_losses()
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_inputs)
        loss = loss_function(predictions, train_targets)
        loss.backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise RuntimeError("Missing or nonfinite gradients")
        optimizer.step()
        record_losses()
        if (epoch + 1) % 50 == 0:
            print(f"epoch={epoch + 1} train MSE={training_losses[-1]:.6f} validation MSE={validation_losses[-1]:.6f}")

    if not np.isfinite([training_losses, validation_losses]).all():
        raise RuntimeError("Nonfinite loss")
    if validation_losses[-1] >= 0.25 * validation_losses[0]:
        raise RuntimeError("Validation MSE did not fall below 25% of its initial value")
    if not any(not torch.equal(before, after) for before, after in zip(initial_parameters, model.parameters())):
        raise RuntimeError("Optimizer did not update parameters")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    restored = make_model()
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(validation_inputs), model(validation_inputs), rtol=0, atol=0)

    metrics = {
        "task": "synthetic: y = 2*x0 - x1 + 0.5",
        "seed": 42,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "epochs": 200,
        "training_samples": 512,
        "validation_samples": 128,
        "training_mse": training_losses,
        "validation_mse": validation_losses,
        "checkpoint_verified": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    figure = Figure(figsize=(7, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.semilogy(training_losses, label="Training")
    axes.semilogy(validation_losses, label="Validation")
    axes.set(xlabel="Epoch (0 = before training)", ylabel="Mean squared error", title="Synthetic CPU training smoke test")
    axes.legend()
    figure.savefig(output_dir / "loss.png")
    print(f"PASS: CPU training and checkpoint reload; artifacts: {output_dir.resolve()}")
    return metrics


if __name__ == "__main__":
    run_smoke()
