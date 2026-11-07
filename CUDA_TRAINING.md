# CUDA desktop and CPU laptop training

The portable trainer uses the existing 15-input, three-hidden-layer SiLU network
and 12 displacement outputs. Width 512 has 539,660 parameters; width 1024 has
2,127,884. Start with 512 and measure validation error and CPU epoch time before
increasing further. A wider network does not guarantee better predictions.

## Environment

Create a separate environment on each computer; do not copy `.venv` between them.
Python 3.12 or 3.13 works with the selected PyTorch 2.10.0 wheels.

```powershell
py -3.12 -m venv .venv
# Desktop (RTX 2060 Super):
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# Laptop: use this instead of the desktop installation command:
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
# Both machines:
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

These commands select matching PyTorch versions with different compute builds.
The historical `requirements-dev.txt` pins a different CPU-only PyTorch; use
`requirements-training.txt` for this workflow. Official installation guidance:
https://pytorch.org/get-started/locally/.

## Fresh larger model

Restore the original dataset and a matching normalization checkpoint first.
Neither `data/` nor `runs/` is tracked by Git. The reference below supplies
normalization only; the wider network starts with fresh weights. Its objective
is one-step normalized MSE with existing 5x **contact** weighting. It does not
implement the separate stationary XY weighting experiment.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.portable_training --dataset data/arm_cube_contact_boundary/mixed.npz --reference runs/arm_cube_contact_boundary/firstpass/model.pt --output runs/cuda_512 --width 512 --epochs 100 --device auto
```

`auto` selects CUDA when available and otherwise CPU. `--device cuda` fails clearly
if unavailable; `--device cpu` forces the laptop-compatible path on either machine.
`--batch-size` defaults to 256; `--threads` defaults to 2. Batches transfer from
host RAM to the device as needed. Training uses float32 on both devices.

Each epoch saves CPU tensors in `last.pt`, including Adam moments and shuffle
state. `model.pt` is the best unweighted validation checkpoint **within this run**.
`history.json` records validation loss and epoch time. Existing output directories
are refused. Existing prediction/viewer code can load either checkpoint on CPU.

## Continue on the laptop

Copy the dataset, reference checkpoint and run folder to the laptop along with
the code. Then use a new output directory:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.portable_training --dataset data/arm_cube_contact_boundary/mixed.npz --reference runs/arm_cube_contact_boundary/firstpass/model.pt --resume runs/cuda_512/last.pt --output runs/cpu_512_continued --epochs 20 --device cpu
```

Resume restores network width, batch size, weights, optimizer and shuffle state.
`--epochs` means additional epochs. Dataset hash and normalization must match.
Changing devices can change floating-point results; continuation need not be
bitwise identical across CPU and CUDA. CPU training takes longer, and laptop
timing must be measured there. No NVIDIA GPU is required on the laptop.

Run `python -m pytest -q tests/test_portable_training.py` using the project
interpreter for numerical CPU resume and, when available, CUDA-to-CPU checks.
These fixtures verify training portability, not model quality on robot datasets.
