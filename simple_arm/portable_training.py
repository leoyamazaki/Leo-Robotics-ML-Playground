"""Train wider one-step contact models on CUDA or CPU with portable Adam state."""

import argparse
import json
from pathlib import Path
import time

import torch

from .capacity_screen import DATASET, NORMALIZATION_KEYS, file_hash, prepare
from .contact_boundary import load_boundary
from .contact_learning import make_model


def select_device(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA PyTorch build or use --device cpu")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def fit(reference, train, validation, output, *, width=512, epochs=100,
        batch_size=256, device="auto", resume=None, dataset_sha256=None):
    """Fresh initialization, or resume this trainer's last.pt into a new directory.

    The objective is normalized one-step MSE with existing 5x contact weighting.
    Reference supplies normalization, not initial weights. No stationary multiplier.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    selected = select_device(device)
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved:
        if saved.get("training_method") != "portable_one_step":
            raise ValueError("Resume requires portable_training last.pt")
        if saved.get("dataset_sha256") != dataset_sha256 or saved["metadata"] != reference["metadata"]:
            raise ValueError("Resume dataset differs")
        for key in NORMALIZATION_KEYS:
            torch.testing.assert_close(saved[key], reference[key], rtol=0, atol=0)
        width, batch_size = saved["hidden_width"], saved["batch_size"]
    torch.manual_seed(91)
    model = make_model(width).to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    shuffle = torch.Generator().manual_seed(91)
    start, best = 0, float("inf")
    if saved:
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        shuffle.set_state(saved["shuffle_state"])
        start = saved["epoch"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Training on {selected}: width={width}, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if selected.type == "cuda":
        print(torch.cuda.get_device_name(), flush=True)
    # Keep the dataset in host memory; transfer only a batch so VRAM use is bounded.
    x, _, _ = train
    vx, vy, _ = validation
    history = []
    for epoch in range(start + 1, start + epochs + 1):
        started = time.perf_counter()
        model.train()
        for indices in torch.randperm(len(x), generator=shuffle).split(batch_size):
            bx, by, bw = (item[indices].to(selected) for item in train)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(bx) - by).square().mean(-1) * bw).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.
        with torch.no_grad():
            for offset in range(0, len(vx), batch_size):
                error = model(vx[offset:offset + batch_size].to(selected)) - vy[offset:offset + batch_size].to(selected)
                total += error.square().mean(-1).sum().item()
        score = total / len(vx)
        if not torch.isfinite(torch.tensor(score)):
            raise RuntimeError("Nonfinite validation loss")
        improved = score < best
        best = min(best, score)
        checkpoint = {key: reference[key] for key in (*NORMALIZATION_KEYS, "metadata", "stride")}
        checkpoint.update(weights=cpu_tree(model.state_dict()), hidden_width=width,
                          optimizer=cpu_tree(optimizer.state_dict()), shuffle_state=shuffle.get_state(),
                          epoch=epoch, best_validation=best, batch_size=batch_size,
                          dataset_sha256=dataset_sha256, training_method="portable_one_step",
                          training_device=str(selected), torch_version=str(torch.__version__),
                          objective="normalized_delta_mse_5x_contact", init_seed=91,
                          learning_rate=.001, resumed_from=str(resume) if resume else None)
        # Replace only checkpoints owned by this new run; retain the previous epoch on interruption.
        temporary = output / "last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "model.pt")
        history.append({"epoch": epoch, "validation": score, "seconds": time.perf_counter() - started})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} validation={score:.6f} seconds={history[-1]['seconds']:.2f}", flush=True)
    return output / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--reference", type=Path, required=True, help="Existing checkpoint supplying exact dataset normalization")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Continue last.pt; restores width, batch size, Adam and shuffle state")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100, help="Additional epochs when resuming")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    train, validation = prepare(load_boundary(args.dataset), reference)
    fit(reference, train, validation, args.output, width=args.width, epochs=args.epochs,
        batch_size=args.batch_size, device=args.device, resume=args.resume,
        dataset_sha256=file_hash(args.dataset))


if __name__ == "__main__":
    main()
