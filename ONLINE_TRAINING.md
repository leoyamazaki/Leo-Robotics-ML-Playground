# Online random-data residual training

The first full million-transition run is complete; see
[results and limitations](ONLINE_TRAINING_REPORT.md).

This experiment generates new MuJoCo episodes while the GPU trains the existing
4,746,252-parameter base-relative residual MLP. It retains complete episodes in
a bounded FIFO replay buffer and archives every consumed episode for continuation.
It is a new training protocol, not an equal-compute comparison with the earlier
100/80-epoch saved-dataset experiment.

## Default budget and distribution

|Stage|Collection epochs|New episodes per epoch|Updates per epoch|Learning rate|
|---|---:|---:|---:|---:|
|One-step|20|1,000|256|0.001|
|Ten-step recursive|20|250|64|0.0002|

Total: **25,000 unique four-second episodes / 1,000,000 saved 100ms transitions**.
An epoch here means a fresh collection chunk followed by a fixed update budget.
The number of SGD samples differs from the number of unique generated transitions:
replay sampling can revisit an example or leave some examples unused. Defaults
use float32, batches of 256, two training CPU threads, four simulation workers,
and automatic CUDA/CPU selection. Replay retains the latest 5,000 complete
episodes. Bounded asynchronous prefetch overlaps CPU simulation and training;
results are consumed in episode-ID order, independently of worker scheduling.

Random generation includes:

- Initial arm angles uniformly within joint limits with a 0.15rad margin;
  stationary arm starts.
- Cube XY sampled uniformly from `[-0.55,0.55]²` meters, excluding radius <0.12m
  near the base and invalid physical starts. Uniform cube yaw over a full turn.
- A random half of cube starts have planar speed 0.02–0.15m/s and yaw rate
  within ±1rad/s; the others start stationary.
- Six equally sampled command families: idle, shoulder-only, elbow-only,
  independent persistent goals, coordinated motion and reversals.
- Every joint target change, including the first, capped at **0.2rad/s**.

Only invalid physical starts are retried. Misses, idle episodes, tiny cube motion
and unsuccessful pushes are retained. Nonfinite simulation stops with diagnostics;
it is not silently retried. Actual joint speeds and limit excursions are recorded
without selecting episodes by them. Simulation remains at 2ms with states saved
every 20ms. Geometry, mass and friction remain fixed. This is randomized coverage
of a specified regime, not uniform sampling over all possible physical states.

Each training sample independently has a **10% probability** of coming from the
existing focused dataset's training split; the other 90% comes from random replay.
The actual sampled fraction and replay contact/stationary distributions are logged.
This explicit sampling mixture is the only contact-enrichment mechanism.

**Both stages use unweighted MSE.** There is no 5x contact multiplier, 3x contact
multiplier, stationary penalty or stationary XY multiplier. One-step loss averages
the 12 normalized displacement errors. Recursive loss averages state-scaled errors
over all ten feedback steps, components and batch, with scales
`[1,1,3,3,0.1,0.1,0.01,0.5,0.5,0.5,0.5,0.1,2]`. Recursive gradient norm is clipped
to 5; predicted states are never clipped or detached.

## Normalization, validation and selection

Training generation uses seed 791, fixed random validation seed 792, and fixed
random test seed 793. Network initialization and one-step sampling use seed 91;
recursive sampling uses seed 93. Normalization is fitted once from the first
1,000 random training episodes plus 111 randomly selected focused **training**
episodes, retaining the existing physical floors. It is frozen for the entire run.

Fixed random validation/test sets each contain 120 episodes. Focused validation
and test use only their original disjoint splits, preserving paired groups.
Validation logs stationary, moving and contact XY RMSE separately for random and
focused cases. Selection uses `0.9 * random_score + 0.1 * focused_score`, reflecting
the expected training mixture; each score itself is unweighted. One-step selection
uses normalized displacement MSE. Recursive selection uses complete unweighted
40-step rollouts and includes epoch zero. Test/independent metrics never select
weights. The test archive is generated in advance but is not used for fitting or
checkpoint selection.

## Run and monitor

Install [hardware-specific PyTorch and training requirements](CUDA_TRAINING.md).
The 0.2rad/s focused supplement must already exist.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.online_training train --device auto --workers 4 --output runs/arm_cube_online
```

The output directory must be new. The terminal uses one in-place tqdm bar per
stage, reused across epochs with `leave=False`; it shows epoch, batch, loss,
latest validation and ETA. Non-interactive terminals disable it and print only
stage/evaluation completion summaries or errors. Per-epoch metrics are saved
to JSON and plots, not printed.

The trainer refreshes `status.json` every five seconds during updates and at
phase changes. To request a single readable status snapshot:

Opening `status.html` inside the run folder provides an automatically refreshing
local status page without adding terminal output.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.online_training status --output runs/arm_cube_online
```

`one_step/history.json` and `recursive/history.json` contain detailed validation
strata, coverage, actual supplement fraction, epoch timings and training loss.
Each stage has `loss.png`, selected `model.pt`, and CPU-portable `last.pt`.
`training_summary.json` records the consumed generation budget and selected epochs.
All consumed random episodes and fixed holdouts are retained in `data/` inside
the run folder as compressed NPZ chunks. Per-episode initial-start rejection counts,
2ms peak speed and limit-excursion diagnostics are retained in chunk metadata.

## Continue on the laptop

Copy the complete run folder (including its data chunks), matching code and the
focused supplement. Install matching CPU PyTorch locally; do not copy `.venv`.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.online_training train --device cpu --workers 2 --resume runs/arm_cube_online --output runs/arm_cube_online_cpu
```

Epoch counts are total targets, not additional epochs. All non-epoch training
settings, source hashes, fixed holdout hashes, supplement hash, model weights,
Adam state, replay archives and sampling state are checked/restored. Worker count
and device may change. Resume into a new directory; earlier best checkpoints
survive even if later validation never improves. To extend one-step training
and restart refinement, explicitly resume `one_step/last.pt`; a recursive
continuation must retain its original one-step budget. CPU continuation is
supported, but CPU/CUDA arithmetic need not produce bitwise identical trajectories.

## Evaluation

Evaluation runs automatically at completion, or can be deferred with
`--skip-evaluation` and invoked later:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.online_training evaluate --output runs/arm_cube_online
```

The two selected models are compared on the identical random test, focused test,
and available independent boundary/broad sets. Reports reuse existing stationary,
false/missed motion, contact-conditioned, 1/2/4-second rollout and endpoint metrics.
CPU inference timing uses two threads, 10 warm-up calls, 50 measurements and batch
sizes 1/256. `evaluation/REPORT.md` lists regressions and reports epoch-zero selection
honestly. Controller defaults remain unchanged. The future 1,536-wide plain-MLP
comparison remains untrained.

The whole run lives under Git-ignored `runs/`; retain it separately when moving
machines. Small real-simulator integration tests are in `tests/test_online_training.py`.

## Visualize the selected model in MPC

```powershell
.\.venv\Scripts\python.exe -m simple_arm.cube_mpc view --checkpoint runs/arm_cube_online/recursive/model.pt --dataset data/arm_cube_contact_boundary/independent.npz --episode 23 --device cuda --command-rate-limit 0.2 --candidates 128 --iterations 3
```

The viewer accepts the regenerated boundary scenarios while historical benchmark
commands retain their seed-292 guard. CUDA inference is optional; `--device cpu`
supports the laptop. The reduced CEM search is explicit for interactive speed;
controller defaults remain unchanged. Green marks the goal, magenta the predicted
cube trajectory and endpoint. Keys 6/7 change target X, 8/9 change target Y,
Enter resets, and R selects another nearby cube spawn. Timing/fallback warnings
are limited to one every five seconds. Close the MuJoCo window to stop.
