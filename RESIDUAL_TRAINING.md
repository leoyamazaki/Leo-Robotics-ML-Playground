# Base-relative residual world model

This experiment is implemented but **has not yet been trained**. A new
[0.2 rad/s dataset](DATASET_02_REPORT.md) has now been generated at the default
dataset path, with independent evaluation sets. It replaces the missing original
collection and must be labeled as a new experiment. Numerical test fixtures
verify implementation only and are never substituted for experiment data.
The training CLI itself collects no simulation data.

## Environment and full run

Use the [CUDA/CPU environment instructions](CUDA_TRAINING.md), then install
the updated hardware-neutral requirements (including tqdm):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m simple_arm.residual_training train --dataset data/arm_cube_contact_boundary/mixed.npz --output runs/arm_cube_base_residual --device auto --one-step-epochs 100 --recursive-epochs 80 --batch-size 256
```

The saved mixed archive is required; missing archives produce a clear error.
A previous trained checkpoint is not required for fresh initialization. The
output directory must be new. Training uses two CPU threads and float32. `auto`
selects CUDA when available, otherwise CPU; `--device cuda` fails if unavailable.
The dataset stays in host RAM, with individual batches transferred to the device.

One tqdm bar per stage updates in place and is reused between epochs, showing
stage, epoch, current batch/total, training loss, latest validation score, and
remaining stage batch ETA. It closes with `leave=False`. Non-TTY and `TERM=dumb`
outputs disable it; only stage completion summaries, warnings, and errors print.
Validation/checkpoint time is included in observed throughput; ETA is an estimate.

## Exact experiment

Only the residual architecture is trained:

- `Linear(15,768) → SiLU`.
- Four blocks computing `SiLU(x + Linear2(SiLU(Linear1(x))))`, both biased
  linears mapping 768 to 768.
- Biased `Linear(768,12)`, no output activation, dropout or batch normalization.
- Exactly **4,746,252 parameters**.

Cube XY inputs are relative to the fixed world base (0,0), so they are recorded
world XY. All other inputs and all 12 displacement targets retain their previous
definitions. Input/target normalization is fitted only on training episodes,
using existing physical scale floors. Those tensors are preserved for refinement
and continuation. Splits and paired groups are read unchanged from the archive;
groups crossing splits are rejected.

Stage 1 initializes fresh weights with seed 91 and uses an independent shuffle
generator seeded 91. Adam runs at 0.001 for 100 epochs. Loss is normalized
12-component displacement MSE weighted 5x for contact-containing intervals,
otherwise 1x. There is no stationary penalty. Every epoch is validated using
unweighted normalized one-step MSE; the lowest score selects the checkpoint.

Stage 2 starts from the selected one-step checkpoint and a fresh Adam at 0.0002,
with shuffle seed 93. Each of 80 epochs visits every ten-step window starting at
indices 0–30 in each training episode (31 windows per episode). Predictions feed
back through all ten steps, without teacher forcing, clipping states or detaching
feedback. State-scaled MSE averages over batch, components and steps; contact
steps receive weight 3. The fixed scales are
`[1,1,3,3,0.1,0.1,0.01,0.5,0.5,0.5,0.5,0.1,2]`. Gradient norm is clipped to 5.
Validation uses unweighted 40-step rollouts from each validation episode's measured
initial state at epoch zero and every epoch. Epoch zero can win selection.

## Continue on either machine

Copy the complete run folder, original dataset and matching source code. Install
the CPU build of the same PyTorch version on the laptop; do not copy `.venv`.
Resume into a new directory:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.residual_training train --dataset data/arm_cube_contact_boundary/mixed.npz --resume runs/arm_cube_base_residual --output runs/arm_cube_base_residual_continued --device cpu --one-step-epochs 100 --recursive-epochs 80 --batch-size 256
```

Epoch options are **total stage targets**, not additional epochs. An interrupted
run continues the incomplete stage; completed one-step artifacts are retained.
To extend a finished recursive stage, increase `--recursive-epochs`, e.g. to 100.
To extend one-step training and restart refinement from its newly selected model,
explicitly resume `--resume runs/arm_cube_base_residual/one_step/last.pt` and
increase `--one-step-epochs`. A recursive continuation cannot change its source
one-step schedule. A recursive `last.pt` requires its sibling one-step artifacts.

Continuation restores weights, Adam state, shuffle state, history, stage and epoch.
The preceding selected checkpoint is embedded in `last.pt` and preserved even if
no new epoch improves validation. Dataset SHA-256, source hashes, batch size and
training contract must match. Changing device is allowed; CPU/CUDA floating-point
differences can produce different trajectories, despite preserved optimizer state.
Nonfinite training loss, gradients or validation stop with diagnostics. The last
complete epoch remains available; selected checkpoints are never replaced by
nonfinite candidates. Checkpoints and metadata do not change controller defaults.

## Evaluation and artifacts

The training command evaluates both selected models automatically. Use
`--skip-evaluation` to defer this, then run:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.residual_training evaluate --dataset data/arm_cube_contact_boundary/mixed.npz --run runs/arm_cube_base_residual --output runs/arm_cube_base_residual_reevaluation
```

This creates a new `evaluation/` directory under the chosen output path. Add
`--historical PATH_TO_CHECKPOINT` to include an explicitly labeled historical
reference; its differing inputs and training history preclude an isolated
architecture comparison. Missing supplied checkpoints fail clearly.

Both models use the identical mixed test episodes. If present, the saved
`data/arm_cube_contact_boundary/independent.npz` and `broad_independent.npz`
are evaluated on all episodes, with collection-seed overlap checks. Missing
independent datasets are recorded as omitted. No test data selects a checkpoint.

Artifacts include:

|Path|Contents|
|---|---|
|`config.json`|Architecture/frame, resolved settings, hashes, seeds, device, normalization provenance and parameter count|
|`one_step/model.pt`, `recursive/model.pt`|Stage winners selected exclusively on validation|
|`one_step/last.pt`, `recursive/last.pt`|CPU-portable continuation state and preceding best checkpoint|
|Each stage's `history.json`, `loss.png`|Per-epoch weighted training, unweighted validation and timings|
|`timing.json`|Wall time for the current training invocation|
|`evaluation/REPORT.md`, `summary.json`|Selected epochs, runtime, paired metrics, regressions, CPU latency and omissions|
|Per-dataset/model evaluation folder|Full physical errors, stationary/motion diagnostics, saved viewer-compatible rollouts and plots|

Evaluation reuses existing definitions for contact labels, stationary intervals
(speed <=5mm/s and excursion <=0.5mm at every saved 20ms sample), false/missed
motion (0.5mm and 2mm), and stationary episode drift (all saved XY within 2mm of
initial position). It reports overall/contact one-step errors and 1/2/4-second
prefix rollout and endpoint errors. CPU transition latency uses two threads,
10 warm-up calls and 50 measurements at batch sizes 1 and 256, with identical
recorded inputs and conditions for all models. Desktop CPU timings do not measure
the laptop's throughput.

Checkpoints explicitly record `architecture` and `input_coordinate_frame`.
Missing fields retain legacy plain three-layer MLP and tip-relative behavior;
unknown configurations are rejected. NumPy prediction, differentiable transitions,
the replay viewer and MPC share construction/feature selection. To preview a new
checkpoint, pass its path explicitly to `simple_arm.cube_mpc view --checkpoint ...`.

## Validation and future comparison

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_residual_training.py tests/test_contact_learning.py tests/test_portable_training.py tests/test_capacity_screen.py tests/test_cube_mpc.py
```

The new numerical tests cover architecture/gradients, residual behavior, features,
legacy predictions, MPC loading, training-only normalization, windows/contact
weights, differentiable feedback, both stages, selection preservation, CPU resume
equivalence, paired evaluation artifacts and CUDA-to-CPU continuation when hardware
is available. Some existing MPC tests require the original ignored artifacts;
their missing-file failures cannot verify historical checkpoint integration here.

Local verification on 2026-11-07: **67 relevant tests passed**, including CUDA
portability and numerical paired evaluation; five artifact-dependent existing
MPC tests were excluded. The initial complete MPC attempt failed four tests
with missing-file errors. A later disk-space failure was resolved for verification
by placing temporary test files on E:. No real robot-data training or scientific
comparison results are claimed. `pip check` and `git diff --check` passed.

Future work: train the parameter-matched plain MLP with three 1,536-wide hidden
layers (**4,764,684 parameters**, about 0.39% more). Match base-relative inputs,
normalization, splits, objectives, budgets and seeds to isolate architecture.
That model is not trained by this experiment.
